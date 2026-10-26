""" Process pfail data (monte carlo snapshots of long-range toric code after thermal dynamics)
Applies MWPM decoding to see if original ground state is returned, calculates prob of logical failure
"""

# IMPORTANT: requires PyMatching package
import numpy as np
from numba import njit
import pickle
import sys
import os
from pathlib import Path
from longtoric import temps_list
from joblib import Parallel, delayed
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
from pymatching import Matching

ALPHA = 2.5 # put whatever alpha(s) your generated data uses

# ------ COMBINE AND COMPRESS RAW MONTE CARLO DATA -------

def combine_and_compress(alpha):
    # =========================
    # COMBINE 10 sets of 1000 trials
    # =========================
    Lvals = [16, 24, 32, 40]
    temps = temps_list(alpha)
    # Load and combine all 10 batches
    for L in Lvals:
        for tempind, temp in enumerate(temps):
            all_trials = []
            # 10 super-trials to combine to get 10k total trials
            for i in range(10):
                fname = f"data/pfail_L{L}_alpha{alpha}_tempind{tempind}_1000trials_suptrial{i}.p"
                with open(fname, "rb") as f:
                    batch = pickle.load(f)
                
                all_trials.extend(batch)   # assumes each file contains a list-like object

            print(len(all_trials))  # should be 10000

            # Save combined result
            outname = f"data/pfail_L{L}_alpha{alpha}_tempind{tempind}_10ktrials.p"

            with open(outname, "wb") as f:
                pickle.dump(all_trials, f)
            print(f"Dumped L = {L}")

    print("DONE COMBINE")
    # =========================
    # COMPRESS
    # =========================

    # Directory containing pickle files
    INPUT_DIR = Path("./data")
    # Where compressed files will go
    OUTPUT_DIR = Path("./compressed")

    # Filename example L32_alpha3.5_temp0.7_10ktrials.p
    # file contains python list with 10k entries
    # each entry is np.ndarray with int8 dtype of size 2*L*L, entries are -1 and +1

    for L in Lvals:
        for tempind, temp in enumerate(temps):
            fname = f"pfail_L{L}_alpha{alpha}_tempind{tempind}_10ktrials"

            input_path = INPUT_DIR / f"{fname}.p"
            # add extension if needed
            if not input_path.exists():
                print(f"Missing file: {fname}")
                continue
            print(f"Loading: {input_path}")
            with open(input_path, "rb") as f:
                configs = pickle.load(f)
            # Stack list of arrays into one ndarray
            # shape = (n_configs, 2*L*L)
            X = np.stack(configs)
            if X.dtype != np.int8:
                X = X.astype(np.int8)
            # Optional validation
            unique_vals = np.unique(X)
            if not np.all(np.isin(unique_vals, [-1, 1])):
                raise ValueError(f"Unexpected values in {fname}: {unique_vals}")
            # Convert {1,-1} -> {0,1}
            bits = (X == -1).astype(np.uint8)
            # Pack bits along config dimension
            packed = np.packbits(bits, axis=1)
            # Save compressed
            output_path = OUTPUT_DIR / f"{fname}.npz"
            np.savez_compressed(
                output_path,
                packed=packed,
                shape=np.array(X.shape, dtype=np.int64))

            original_size_mb = X.nbytes / 1e6
            packed_size_mb = packed.nbytes / 1e6
            print(
                f"Saved: {output_path.name} | "
                f"raw={original_size_mb:.2f} MB | "
                f"packed={packed_size_mb:.2f} MB"
            )
    print("DONE COMPRESS")
    print("DONE.")


# ------- MWPM HELPERS ------- #

# convert my own system of links to pymatching error vector (note hz and vert different!)
# my default values are 1/-1, plaquette/magnetic anyons, horiz links x -> x+1, vertical y -> y+1
# pymatching values are 0/1, star terms/electric anyons, horiz links x-1 -> x, vertical y -> y+1
def links_to_pymatching(syst, L, binary=True):
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

# does not require initial logical sector to be (1, 1), works on batch of systems
def mwpm_vectorized(systems, Hx, logX, matching=None, binary=True):
    # convert systems (n_trials x N) to pymatching convention 
    N = systems[0].size
    L = int(np.sqrt(N//2))
    noise = np.array([links_to_pymatching(syst, L, binary) for syst in systems])

    if matching is None:
        matching = Matching.from_check_matrix(Hx, weights=1.0, faults_matrix=logX)

    syndromes = (noise @ Hx.T) % 2
    actual_observables = (noise @ logX.T) % 2 # actual logical error of system
    predicted_observables = matching.decode_batch(syndromes) # decoder correction

    # want (error + correction) = 0, i.e. in trivial homology class
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    return num_errors

#------ PROCESS DATA USING MWPM ------ #

# binary encoded file w/ systems: 'pfail_L{L}_alpha{alpha}_tempind{tempind}_10ktrials.npz'
def read_compressed_data(file):
    data = np.load(file)
    packed = data["packed"]
    systems = np.unpackbits(packed, axis=1) # unpack bits
    return systems

# process the monte carlo data after it has been compressed
def process_pfail_data(alpha):
    # 30 temps for each L, alpha (values don't matter here, tracked internally)
    def _run_L_alpha(L, alpha, num_temps=30):
        Hx, logX = toric_pymatching(L)
        matching = Matching.from_check_matrix(Hx, weights=1.0, faults_matrix=logX)
        print(f"created matching object for L={L}, alpha={alpha}")
        failure_rates = []
        for tempind in range(num_temps):
            print(f'L = {L}, alpha = {alpha}, tempind = {tempind}')
            systems = read_compressed_data(f'compressed/pfail_L{L}_alpha{alpha}_tempind{tempind}_10ktrials.npz')
            num_errors = mwpm_vectorized(systems, Hx, logX, matching)
            failure_rates.append(num_errors / 10000)
        temps = temps_list(alpha, which='pfail')
        with open(f'pfail_plot_alpha{alpha}_L{L}.p', 'wb') as h:
            pickle.dump([temps, failure_rates], h)
        print(f"Dumped L={L}, alpha={alpha}")
        return L, alpha, failure_rates
    
    alphavals = [alpha] # put in whatever alpha you need
    Lvals = [16, 24, 32, 40]
    results = Parallel(n_jobs=-1)(delayed(_run_L_alpha)(L, alpha, num_temps=30)
                                    for alpha in alphavals for L in Lvals)
    print("DONE")

if __name__ == "__main__":
    combine_and_compress(alpha=ALPHA) # run on cluster after generating all pfail data 
    process_pfail_data(alpha=ALPHA) # run locallly after pulling data from cluster