""" Generates 1000 MC Snapshots of long-range toric code, for several temperatures
MC Snapshots can come from either cooling from T = infinity or heating from T = 0"""
import numpy as np
from numba import njit
import pickle
import sys
import os
from pathlib import Path
from longtoric import *
from joblib import Parallel, delayed

L = int(sys.argv[1])
ALPHA = float(sys.argv[2])
SUPTRIAL = int(sys.argv[3])
print(f"ALPHA = {ALPHA}, SUPTRIAL = {SUPTRIAL}")
TEMPS = temps_list(ALPHA, which='pfail')
print(f'TEMPS: {TEMPS}')
print(f'Starting super-trial {SUPTRIAL}')
COOLING = True # by default, cool from infinite temperature

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

def main():
    print(f'L = {L}')
    print("Building interaction tables...")
    tab = build_tables(L, ALPHA, Rmax=L//2)
    print("Starting simulations ")

    def _run_temp(tempind):
        temp = TEMPS[tempind]
        fname = f'data/mc_snapshots_L{L}_alpha{ALPHA}_tempind{tempind}_1000trials_suptrial{SUPTRIAL}.p'
        if os.path.exists(fname):
            print(f'File exists, skipping temp = {temp}')
            return
        print(f'\nL={L}, temp = {temp}')
        thousand_trials = []
        for trial in range(1000):
            all_data = equilibrate(L, ALPHA, temp, input_tables=tab, cooling=True)
            syst = all_data[-1]
            thousand_trials.append(syst)

        with open(f'data/mc_snapshots_L{L}_alpha{ALPHA}_tempind{tempind}_1000trials_suptrial{SUPTRIAL}.p', 'wb') as h:
            pickle.dump(thousand_trials, h)
        print("Dumped")

    # run the 30 temperatures in parallel
    Parallel(n_jobs=-1)(delayed(_run_temp)(tempind) for tempind in range(len(TEMPS)))

# ------ COMBINE AND COMPRESS RAW DATA -------- #
# This must be done to analyze the raw data for specific_heat.py or rg_decoder.py
# Best done after using main() to generate all the raw data

def combine_and_compress(alpha):
    # =========================
    # COMBINE 10 sets of 1000 trials
    # =========================
    Lvals = [16, 24, 32, 40] # whichever L you have data for (e.g. RG decoder uses L = 128)
    temps = temps_list(alpha)
    # Load and combine all 10 batches
    for L in Lvals:
        for tempind, temp in enumerate(temps):
            all_trials = []
            # 10 super-trials to combine to get 10k total trials
            for i in range(10):
                fname = f"data/mc_snapshots_L{L}_alpha{alpha}_tempind{tempind}_1000trials_suptrial{i}.p"
                with open(fname, "rb") as f:
                    batch = pickle.load(f)
                
                all_trials.extend(batch)   # assumes each file contains a list-like object

            print(len(all_trials))  # should be 10000

            # Save combined result
            outname = f"data/mc_snapshots_L{L}_alpha{alpha}_tempind{tempind}_10ktrials.p"

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
            fname = f"mc_snapshots_L{L}_alpha{alpha}_tempind{tempind}_10ktrials"

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


if __name__ == "__main__":
    main() # only generates the snapshots, need to separately do the combine_and_compress()
