""" Probability of logical error (failure of memory) after thermal dynamics in long-range toric code
Starts from a ground state, runs t = 200 monte carlo steps at temperature T, saves final snapshot of system
"""
import numpy as np
from numba import njit
import pickle
import sys
import os
from longtoric import temps_list

ALPHA = float(sys.argv[1])
SUPTRIAL = int(sys.argv[2])
print(f"Pfail: ALPHA = {ALPHA}, SUPTRIAL = {SUPTRIAL}")
TEMPS = temps_list(ALPHA, which='pfail')
print(f'TEMPS: {TEMPS}')
print(f'Starting super-trial {SUPTRIAL}')

# syst[d, x, y]: d=0 horizontal link (x,y) -> (x+1,y), d=1 vertical (x,y) -> (x,y+1)
# flat index: d * L * L + x * L + y (same as pymatching?)
@njit(fastmath=True, inline='always')
def link_idx(d, x, y, L):
    return d * L * L + x * L + y

# return link indices for links comprising an R x R square at (x0, y0) 
@njit(fastmath=True, inline='always')
def square_links(x0, y0, R, L):
    lks = np.empty(4*R, dtype=np.int32)
    for k in range(R):
        i = 4*k
        lks[i]   = link_idx(0, (x0+k)%L, y0%L, L)
        lks[i+1] = link_idx(1, (x0+R)%L, (y0+k)%L, L)
        lks[i+2] = link_idx(0, (x0+k)%L, (y0+R)%L, L)
        lks[i+3] = link_idx(1, x0%L, (y0+k)%L, L)
    return lks

# set Rmax = L//2
@njit(fastmath=True)
def build_tables(L, alpha, Rmax, J1=1.0):
    n_links = 2*L*L
    n_sq = L*L*(Rmax-1)
    sq_per_link = Rmax * (Rmax-1)
    max_lk = 4 * (Rmax-1)
    norm = np.pi**2/6 # zeta(2) fixed for everyone

    J = np.zeros(n_sq) # interaction strength per square
    sq_lk_flat = np.zeros((n_sq, max_lk), dtype=np.int32) # links in each square
    sq_lk_len = np.zeros(n_sq, dtype=np.int32) # number of links in each square
    lk_sq = np.zeros((n_links, sq_per_link), dtype=np.int32) # squares (interactions) in each link
    lk_sq_count = np.zeros(n_links, dtype=np.int32) # number of squares added so far (count) in each link 

    si = 0 # square index
    for R in range(1, Rmax):
        JR = (J1 if R == 1 else R**(-alpha)) / norm # power law coeff
        for x0 in range(L):
            for y0 in range(L):
                lks = square_links(x0, y0, R, L) # all the links inside this specific square
                J[si] = JR # strength of this particular interaction
                sq_lk_len[si] = 4*R
                for i in range(4*R):  # run through links of square
                    li = lks[i]
                    sq_lk_flat[si, i] = li
                    lk_sq[li, lk_sq_count[li]] = si
                    lk_sq_count[li] += 1
                si += 1

    return J, sq_lk_flat, sq_lk_len, lk_sq

# link inds of (x, y)-Wilson loop (on lattice, NOT dual)
# measures presence of non-contractible (y, x)-loop on dual
def logical_inds(L):
    Wx = np.array([link_idx(0, x, 0, L) for x in range(L)])
    Wy = np.array([link_idx(1, 0, y, L) for y in range(L)])
    return Wx, Wy

# link inds of (x, y)-t Hooft loop (on dual NOT lattice)
# used to flip non-contractible (x, y)-loop on dual
def thooft_inds(L):
    Tx = np.array([link_idx(1, x, 0, L) for x in range(L)])
    Ty = np.array([link_idx(0, 0, y, L) for y in range(L)])
    return Tx, Ty

# make list of signs of each square interaction given a syst of links
@njit(inline='always', fastmath=True)
def init_signs(syst, sq_lk_flat, sq_lk_len, n_sq):
    signs = np.empty(n_sq, dtype=np.int8)
    # loop over squares
    for square in range(n_sq):
        sign_square = np.int8(1)
        # loop over links in square
        for link in range(sq_lk_len[square]):
            link_index = sq_lk_flat[square, link]
            sign_square *= syst[link_index]
        signs[square] = sign_square
    return signs
 
# update step: attempt to flip one link
# update system (links) and signs of interactions in-place
@njit(inline='always', fastmath=True)
def mc_update(syst, J, signs, lk_sq, n_links, beta):
    link = np.random.randint(0, n_links)
    squares = lk_sq[link]  # indices of squares containing this link
    dE = 0.0
    for sq in squares:
        dE += 2 * J[sq] * signs[sq]
    if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
        syst[link] *= -1
        for sq in squares:
            signs[sq] *= -1
        return True
    return False


@njit(fastmath=True)
def evolve(syst, J, signs, lk_sq, n_links, beta, savetimes):
    # savetimes must be a sorted 1D int array; sweep t = one attempt per link
    n_save = len(savetimes)
    T_tot = savetimes[-1] + 1
    config_data = np.zeros((n_save, n_links), dtype=np.int8)
    energies = np.zeros(n_save, dtype=np.float64)
    accepted = 0

    config_data[0, :] = syst.copy()
    count = 1
    for t in range(1, T_tot):
        
        for _ in range(n_links):
            acc = mc_update(syst, J, signs, lk_sq, n_links, beta)
            if acc:
                accepted += 1

        if count < n_save and t == savetimes[count]:
            config_data[count, :] = syst.copy()
            energies[count] = np.sum(-J * signs)
            count += 1

    # print("Acceptance ratio: ", accepted/(T_tot * n_links))
    return config_data, energies

# L x L array of magnetic anyons (plaquette defects on dual lattice)
def get_anyons(syst, L):
    syst_grid = syst.reshape(2, L, L)
    flipped = (syst_grid == -1).astype(np.int8)

    # how many links flipped at each mini plaquette 
    count_ylinks = flipped[0] + np.roll(flipped[0], -1, axis=1) # left, right
    count_xlinks = flipped[1] + np.roll(flipped[1], -1, axis=0) # btm, top
    count = count_xlinks + count_ylinks
    
    return (count % 2) == 1

# can start from either partially decohered state or logical state
# input_tables can pre-loaded: (J, sq_lk_flat, sq_lk_len, lk_sq)
def thermalize(L, alpha, temp, trial=0, p_noise=0, logicals=(1, 1), input_tables=None):
    # decoherence threshhold p_noise ~ 0.1 at temp = 0 (lower at nonzero temp)
    n_links = 2 * L * L
    beta = 1/temp
    syst = np.random.choice(np.array([-1, 1], 
        dtype=np.int8), size=n_links, p=[p_noise, 1 - p_noise])
    
    # flip logicals if desired 
    Tx, Ty = thooft_inds(L)
    syst[Ty] *= logicals[0] # sets Wx = +/- 1
    syst[Tx] *= logicals[1] # sets Wy = +/- 1

    # build tables of interactions
    if input_tables is not None:
        J, sq_lk_flat, sq_lk_len, lk_sq = input_tables
    else:
        J, sq_lk_flat, sq_lk_len, lk_sq = build_tables(L, alpha, Rmax=L//2)
    
    # get signs of all interactions in input system
    if p_noise == 0:
        signs = np.ones(len(J), dtype=np.int8)
    else:
        signs = init_signs(syst.copy(), sq_lk_flat, sq_lk_len, len(J))

    # takes only about 50 steps max to thermalize at low temp
    savetimes = np.array([0, 50, 100, 200])
    data, energies = evolve(syst, J, signs, lk_sq, n_links, beta, savetimes)
    return data

# ----- GENERATE MONTE CARLO DATA ------ 

def generate_pfail_data():
    for L in [16, 24, 32, 40]:
        print(f'L = {L}')
        print("Building interaction tables...")
        tab = build_tables(L, ALPHA, Rmax=L//2)
        print("Starting simulations ")
        for tempind, temp in enumerate(TEMPS):
            fname = f'data/pfail_L{L}_alpha{ALPHA}_tempind{tempind}_1000trials_suptrial{SUPTRIAL}.p'
            if os.path.exists(fname):
                print(f'File exists, skipping temp = {temp}')
                continue
            print(f'\nL={L}, temp = {temp}')
            thousand_trials = []
            for trial in range(1000):
                data = thermalize(L, ALPHA, temp, trial, p_noise=0, logicals=(1,1), input_tables=tab)
                thousand_trials.append(data[-1])
            with open(f'data/pfail_L{L}_alpha{ALPHA}_tempind{tempind}_1000trials_suptrial{SUPTRIAL}.p', 'wb') as h:
                pickle.dump(thousand_trials, h)
            print("Dumped")


if __name__ == "__main__":
    generate_pfail_data() # designed to be run on cluster for a single ALPHA, SUPTRIAL
