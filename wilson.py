""" Computes thermal expectation values of Wilson loops in long-range toric code """
import numpy as np
from numba import njit
import sys
import pickle
from joblib import Parallel, delayed
from timeit import default_timer as timer
from longtoric import link_idx, square_links, build_tables, init_signs, evolve, temps_list

ALPHA = float(sys.argv[1])
SUPTRIAL = int(sys.argv[2])
print(f"Wilson: ALPHA = {ALPHA}, SUPTRIAL = {SUPTRIAL}")
TEMPS = temps_list(ALPHA, which='wilson')
print(f'TEMPS: {TEMPS}')

# equilibrate a system to T = temp using 200 MC steps
# cooling = True cools from T = infty, cooling = False heats from T = 0 (a specific ground state)
def equilibrate(L, alpha, temp, input_tables=None, cooling=True):
    n_links = 2 * L * L
    beta = 1/temp
    
    if cooling:
        # start in fully random state, i.e., at infinite temperature
        syst = np.random.choice(np.array([-1, 1], 
            dtype=np.int8), size=n_links, p=[0.5, 0.5])
    else:
        # start in ground state with all spins +1 (T = 0)
        syst = np.random.choice(np.array([-1, 1], 
            dtype=np.int8), size=n_links, p=[0, 1])

    # build tables of interactions
    if input_tables is not None:
        J, sq_lk_flat, sq_lk_len, lk_sq = input_tables
    else:
        J, sq_lk_flat, sq_lk_len, lk_sq = build_tables(L, alpha, Rmax=L//2)
    
    signs = np.ones(len(J), dtype=np.int8)
    # takes only about 50 steps max to thermalize at low temp
    savetimes = np.arange(0, 201, 10)
    data, energies = evolve(syst, J, signs, lk_sq, n_links, beta, savetimes)
    return data

# varying contractible wilson loop side length R (start from T = infty to sample all logical sectors)
def wilson_data_varyR(L, alpha, temp, n_trials):
    J, sq_lk_flat, sq_lk_len, lk_sq = build_tables(L, alpha, Rmax=L//2)
    data = []
    Rvals = np.arange(1, L//2, 1)
    data = np.zeros(len(Rvals))
        
    for t in range(n_trials):
        links = equilibrate(L, alpha, temp, input_tables=[J, sq_lk_flat, sq_lk_len, lk_sq])[-1]
        for ind, R in enumerate(Rvals):
            for x in range(L):
                for y in range(L):
                    wilson_links = links[square_links(x, y, R, L)]
                    data[ind] += np.prod(wilson_links)
    return data / (n_trials * L**2)

def run_varyR(alpha, n_trials, temps):
    def _run_temp(temp, L, alpha_, n_trials_):
        data = wilson_data_varyR(L, alpha_, temp, n_trials_)
        return data
    L = 64
    Rvals = np.arange(1, L//2, 1)
    wilsons = Parallel(n_jobs=-1)(delayed(_run_temp)(temp, L, alpha, n_trials)
        for temp in temps)
    des = 'description, temps, Rvals, <W> (2d array, temps x Rvals)'
    with open(f'data/wilson_varyR_L{L}_alpha{alpha}_suptrial{SUPTRIAL}.p', 'wb') as h:
        pickle.dump([des, temps, Rvals, np.array(wilsons)], h)
    print("DONE")

# average over all vertical noncontractible wilson loops Wy
# start from a specific logical sector so T = 0
def wilson_data_noncontractible(L, alpha, temp, n_trials):
    J, sq_lk_flat, sq_lk_len, lk_sq = build_tables(L, alpha, Rmax=L//2)
    data = []
    for t in range(n_trials):
        links = equilibrate(L, alpha, temp, 
            input_tables=[J, sq_lk_flat, sq_lk_len, lk_sq], cooling=False)[-1]
        for x in range(L):
            wilson_line = np.array([link_idx(1, x, y, L) for y in range(L)])
            wilson_links = links[wilson_line]
            data.append(np.prod(wilson_links))
    return np.mean(data)

def run_noncontractible(alpha, n_trials, temps):
    def _run_temp(temp, L, alpha_, n_trials_):
        w = wilson_data_noncontractible(L, alpha_, temp, n_trials_)
        return w
    Lvals = np.array([16, 24, 32, 40])
    for L in Lvals:
        print(f'L = {L}')
        wilsons = Parallel(n_jobs=-1)(delayed(_run_temp)(temp, L, alpha, n_trials)
            for temp in temps)
        wilsons = np.array(wilsons)
        with open(f'data/wilson_noncontractible_L{L}_alpha{alpha}_suptrial{SUPTRIAL}.p', 'wb') as h:
            pickle.dump(['description, temps, wilson loop expectations', temps, wilsons], h)
    print("DONE")

def main():
    run_noncontractible(alpha=ALPHA, n_trials=1000, temps=TEMPS)
    
if __name__ == '__main__':
    main()

