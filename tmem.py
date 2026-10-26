""" Directly simulates memory time of long-range toric code for different system sizes, temperatures.
Memory time = Monte Carlo time at which MWPM decoding does not recover original toric code ground state
Requires: PyMatching package to do MWPM
"""

# general imports
import numpy as np
from numba import njit
from joblib import Parallel, delayed
import pickle
import sys
from timeit import default_timer as timer
import os
from longtoric import *

# pymatching
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
from pymatching import Matching

ALPHA = float(sys.argv[1])
TEMP_IND = int(sys.argv[2])
LVALS = [8, 12, 16, 20]
N_JOBS = 16 # 16 tasks per node
TEMPS = temps_list(ALPHA, which='tmem')
print(f'TEMPS: {TEMPS}')
TEMP = TEMPS[TEMP_IND]

# --------- MEMORY TIME EXPERIMENT ----------- #

# --------- MWPM DECODING ----------- #

# convert my own system of links to pymatching error vector (note hz and vert different!)
# my default values are 1/-1, plaquette/magnetic anyons, horiz links x -> x+1, vertical y -> y+1
# pymatching values are 0/1, star terms/electric anyons, horiz links x-1 -> x, vertical y -> y+1
def links_to_pymatching(syst, L, binary=False):
    # if syst is not already in binary form with +1 -> 0 and -1 -> 1
    if not binary:
        syst = (syst == -1).astype(np.uint8) # convert to 0/1

    # convert links to dual lattice 
    syst0 = syst.reshape((2, L, L))   
    horiz_old, vert_old = syst0[0], syst0[1]
    horiz_new = np.roll(vert_old, -1, axis=0)
    vert_new = np.roll(horiz_old, -1, axis=1)

    # change my x --> x+1 to x - 1 --> x
    horiz_new = np.roll(horiz_new, +1, axis=0)

    # flatten
    return np.array([horiz_new, vert_new]).flatten()

# returns Hx (parity check matrix) and logX (logicals) for TC
# partiy checks are star terms, anyons on vertices 
def toric_pymatching(L):
    # Repetition code parity checks in 1D    
    row_ind, col_ind = zip(*((i, j) for i in range(L) for j in (i, (i+1)%L)))
    data = np.ones(2*L, dtype=np.uint8)
    Hr = csc_matrix((data, (row_ind, col_ind)))

    # Toric X-stabilizers (star terms) from hypergraph product of repetition codes
    H = hstack( [kron( Hr, eye(Hr.shape[1]) ), kron( eye(Hr.shape[0]), Hr.T )], dtype=np.uint8)
    H.data = H.data % 2
    H.eliminate_zeros()
    Hx = csc_matrix(H)

    # Toric X-logicals
    H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = block_diag( [kron(H1, H0), kron(H0, H1)] )
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    logX = csc_matrix(x_logicals)

    return Hx, logX

# create matching object outside
def mwpm(syst, Hx, logX, matching=None):
    N = syst.size
    L = int(np.sqrt(N//2))
    if matching is None:
        matching = Matching.from_check_matrix(Hx, weights=1.0, faults_matrix=logX)
    
    noise = links_to_pymatching(syst, L, binary=False)
    syndrome = (Hx @ noise) % 2      
    
    actual_logicals = (logX @ noise) % 2 # actual logical error of system  
    predicted_logicals = matching.decode(syndrome) # correction in logical errors from mwpm

    # want (error + correction) = 0, i.e. in trivial homology class
    if not np.array_equal(predicted_logicals, actual_logicals):
        return False # error (failure)
    else:
        return True # no error (success)

@njit(fastmath=True)
def evolve_until_check(syst, J, signs, lk_sq, n_links, beta, n_sweeps):
    """Run n_sweeps sweeps (one attempt per link per sweep) in place, no saving."""
    for _ in range(n_sweeps):
        for _ in range(n_links):
            mc_update(syst, J, signs, lk_sq, n_links, beta)


# first failure time of single monte carlo run
def run_single_trial(L, alpha, temp, max_sweeps, Hx, logX, input_tables,
                    logicals=(1, 1), base_interval=1, max_interval=1000):

    n_links = 2 * L * L
    J, sq_lk_flat, sq_lk_len, lk_sq = input_tables
    matching = Matching.from_check_matrix(Hx, weights=1.0, faults_matrix=logX)

    syst = np.ones(n_links, dtype=np.int8)
    Tx, Ty = thooft_inds(L)
    syst[Tx] *= logicals[1]
    syst[Ty] *= logicals[0]
    signs = np.ones(len(J), dtype=np.int8)

    t_elapsed = 0
    interval = base_interval
    while t_elapsed < max_sweeps:
        step = min(interval, max_sweeps - t_elapsed)
        evolve_until_check(syst, J, signs, lk_sq, n_links, beta=1/temp,
                            n_sweeps=step)
        t_elapsed += step

        success = mwpm(syst.copy(), Hx, logX, matching=matching)
        if not success:
            return float(t_elapsed)
    
    # print(f"Final interval: {interval}")
    return np.nan  # never failed within allotted time

# first failure time, averaged over many trials
def first_failure_time(L, alpha, temp, max_sweeps, n_trials, Hx, logX,
                        input_tables=None, logicals=(1, 1), base_interval=1, 
                        max_interval=None, n_jobs=N_JOBS):
    if input_tables is None:
        input_tables = build_tables(L, alpha, Rmax=L // 2)

    fail_times = Parallel(n_jobs=n_jobs)(delayed(run_single_trial)(L, alpha, temp, max_sweeps, Hx, logX, input_tables,
                                    logicals=logicals, base_interval=base_interval, max_interval=max_interval) 
                                for trial in range(n_trials) )
    return np.array(fail_times)


# For each L in Lvals, compute mean memory time at a fixed temp.
# doesn't record value if more than nan_cutoff fraction of trials never failed
def gen_data(temp, Lvals, alpha, max_sweeps, n_trials=1000,
             logicals=(1, 1), nan_cutoff=0.1, base_interval=1,
             max_interval=None, n_jobs=N_JOBS, verbose=True):
    
    data = np.zeros((len(Lvals), n_trials))
    for i, L in enumerate(Lvals):
        Hx, logX = toric_pymatching(L)
        input_tables = build_tables(L, alpha, Rmax=L // 2)

        if verbose:
            print(f"L={L}, T={temp:.3f}")

        fail_times = first_failure_time(
            L, alpha, temp, max_sweeps, n_trials, Hx, logX,
            input_tables=input_tables, logicals=logicals,
            base_interval=base_interval, max_interval=max_interval, n_jobs=n_jobs)

        data[i, :] = fail_times
        frac_nan = np.mean(np.isnan(fail_times))
        if verbose:
            print(f"mean memory time = {np.mean(data[i])}, "
                  f"frac never failed = {frac_nan:.2f}")

    return data


def main():
    fname = f'data/tmem_alpha{ALPHA}_tempind{TEMP_IND}.p'
    if os.path.exists(fname):
        print(f'File exists, skipping alpha = {ALPHA}, tempind = {TEMP_IND}')
    else:
        Tmax = int(1.4e4)
        t0 = timer()
        data = gen_data(TEMP, LVALS, ALPHA, max_sweeps=Tmax, n_trials=1000,
                     base_interval=1, max_interval=2)
        print(data)
        print(f"time elapsed: {timer() - t0}")

        with open(f'data/tmem_alpha{ALPHA}_tempind{TEMP_IND}.p', 'wb') as h:
            pickle.dump(data, h)
        print("DUmped")
    print("DONE")


if __name__ == '__main__':
    main()
