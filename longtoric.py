"""
Contains helper functions to perform Monte Carlo evolution of long-range toric code model with square loops
"""

# general imports
import numpy as np
from numba import njit
import pickle
from scipy.special import zeta
import sys
import os

# binary encoded file w/ systems: 'L{L}_alpha{alpha}_temp{temp}_10ktrials.npz'
# how to remove padding bits (only needed if 2*L**2 isn't divisible by 8):
# systems = systems[:, :data["shape"][1]]
def read_compressed_data(file):
    data = np.load(file)
    packed = data["packed"]
    systems = np.unpackbits(packed, axis=1) # unpack bits
    return systems
    
# temperature ranges used in simulations
def temps_list(alpha, which='pfail'):
    if which == 'pfail' or which == 'wilson':
        num_temps = 30
    elif which == 'tmem':
        num_temps = 15
    else:
        print("Quantity not supported")
        return None
    if alpha == 2.5:
        T_min, T_max = 0.5, 1.1
    elif alpha == 3.0:
        T_min, T_max = 0.35, 0.7
    elif alpha == 4.0:
        T_min, T_max = 0.2, 0.6
    elif alpha == 1.8:
        T_min, T_max = 1.0, 2.0
    else:
        print("Alpha not supported")
        return None
    return np.linspace(T_min, T_max, num_temps)

# syst[d, x, y]: d=0 horizontal link (x,y) -> (x+1,y), d=1 vertical (x,y) -> (x,y+1)
# flat index: d * L * L + x * L + y (same as pymatching?)
@njit(fastmath=True, inline='always')
def link_idx(d, x, y, L):
    return d * L * L + x * L + y

# convert my own system of links to pymatching state/error vector (note horiz and vert different!)
# my default values are 1/-1, plaquette/magnetic anyons, horiz links x -> x+1, vertical y -> y+1
# pymatching values are 0/1, star terms/electric anyons, horiz links x-1 -> x, vertical y -> y+1
def links_to_pymatching(syst, L):
    syst0 = ((1 - syst.copy()) // 2).astype(np.uint8) # convert to 0/1
    
    # convert links to dual lattice, origin is (0.5, 0.5)
    syst0 = syst0.reshape((2, L, L))   
    horiz_old, vert_old = syst0[0], syst0[1]
    horiz_new = np.roll(vert_old, -1, axis=0)
    vert_new = np.roll(horiz_old, -1, axis=1)

    # change my x --> x+1 to x - 1 --> x
    horiz_new = np.roll(horiz_new, +1, axis=0)

    # flatten
    return np.array([horiz_new, vert_new]).flatten()

# return link indices already
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
 
# flip 4 links around random vertex
# unimportant for plaquette (classical) MC but helps explore all configs in "quantum" gnd state
@njit(inline='always', fastmath=True)
def flip_random_star(syst):
    L = int(np.sqrt(syst.shape[0]/2))
    rand_site = np.random.randint(0, L**2)
    x, y = rand_site % L, rand_site // L

    # links around vertex (x, y): E, N, W, S
    star = [link_idx(0, x, y, L), link_idx(1, x, y, L), 
                    link_idx(0, (x-1)%L, y, L), link_idx(1, x, (y-1)%L, L)]
    for i in star:
        syst[i] *= -1

# monte carlo update flips two adjacent links, possibly creating four anyons 
@njit(inline='always', fastmath=True)
def mc_update_4anyon(syst, J, signs, lk_sq, beta, L):
    link1 = np.random.randint(2 * L * L)
    d = link1 // (L * L)
    x, y = (link1 // L) % L, link1 % L
    rand_neigh = 2 * np.random.randint(2) - 1
    if d == 0:
        x = (x + rand_neigh) % L
    else:
        y = (y + rand_neigh) % L
    link2 = link_idx(d, x, y, L)

    dE = 0.0
    squares1, squares2 = lk_sq[link1], lk_sq[link2]
    for sq in squares1:
        dE += 2 * J[sq] * signs[sq]
        signs[sq] *= -1
    for sq in squares2:
        dE += 2 * J[sq] * signs[sq]
        signs[sq] *= -1

    # flip links
    if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
        syst[link1] *= -1
        syst[link2] *= -1
        return True

    # update signs of squares
    for sq in squares1:
        signs[sq] *= -1
    for sq in squares2:
        signs[sq] *= -1
    
    return False

# evolution where bath just flips single links
@njit(fastmath=True)
def evolve(syst, J, signs, lk_sq, n_links, beta, savetimes):
    # savetimes must be a sorted 1D int array; sweep t = one attempt per link
    L = int(np.sqrt(n_links // 2))
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

# evolution where bath flips single links and pairs of links (creates anyon quadrupoles)
@njit(fastmath=True)
def evolve_ultralong(syst, J, signs, lk_sq, n_links, beta, savetimes):
    # savetimes must be a sorted 1D int array; sweep t = one attempt per link
    L = int(np.sqrt(n_links // 2))
    n_save = len(savetimes)
    T_tot = savetimes[-1] + 1
    config_data = np.zeros((n_save, n_links), dtype=np.int8)
    energies = np.zeros(n_save, dtype=np.float64)
    accepted = 0

    config_data[0, :] = syst.copy()
    count = 1
    for t in range(1, T_tot):
        
        for _ in range(n_links):
            if np.random.rand() < 0.5:
                acc = mc_update(syst, J, signs, lk_sq, n_links, beta)
            else:
                acc = mc_update_4anyon(syst, J, signs, lk_sq, beta, L)
            if acc:
                accepted += 1

        if count < n_save and t == savetimes[count]:
            config_data[count, :] = syst.copy()
            energies[count] = np.sum(-J * signs)
            count += 1

    # print("Acceptance ratio: ", accepted/(T_tot * n_links))
    return config_data, energies

# L x L boolean array of magnetic anyons (plaquette defects)
# syst[d,x,y], d=0 horizontal, d=1 vertical.
def get_anyons(syst, L, magnetic=True, binary=True):
    grid = syst.reshape(2, L, L)
    if not binary:
        grid = (grid == -1).astype(np.int8)

    shift = -1 if magnetic else 1
    axis_x, axis_y = (1, 0) if magnetic else (0, 1)

    count_xlinks = grid[0] + np.roll(grid[0], shift, axis=axis_x)
    count_ylinks = grid[1] + np.roll(grid[1], shift, axis=axis_y)
    count = count_xlinks + count_ylinks

    return (count % 2) == 1


# can start from either partially decohered state or logical state
# input_tables can pre-loaded: (J, sq_lk_flat, sq_lk_len, lk_sq)
# if alpha is low, both anyon dipoles and quadrupoles are created by bath 
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
    if alpha <= 2.0:
        data, energies = evolve_ultralong(syst, J, signs, lk_sq, n_links, beta, savetimes)
    else:
        data, energies = evolve(syst, J, signs, lk_sq, n_links, beta, savetimes)

    # with open(f'../data/squareloop_mc/L{L}_alpha{alpha}_temp{temp}_trial{trial}.p', 'wb') as h:
    #     pickle.dump(data[-1], h)

    return data

def main():
    dummyvariable = 1
    
if __name__ == "__main__":
    main()
