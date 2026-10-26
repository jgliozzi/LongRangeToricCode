""" Calculate specific heat of long-range toric code at different temperatures from monte carlo snapshots
"""
import numpy as np
from numba import njit
import pickle
from longtoric import *
from joblib import Parallel, delayed

# read my own compressed finite temp data
def read_compressed_data(file):
    data = np.load(file)
    packed = data["packed"]
    systems = np.unpackbits(packed, axis=1) # unpack bits
    return systems

# assumes system data are in binary format 0/1 -> 1/-1
@njit
def get_energies(systems, J, sq_lk_flat, sq_lk_len):
    n_trials = systems.shape[0]
    n_sq = len(J)
    energies = np.empty(n_trials)

    for i in range(n_trials):
        syst = 1 - 2*systems[i]  # binary 0,1 -> spins +1,-1
        signs = init_signs(syst, sq_lk_flat, sq_lk_len, n_sq)
        energies[i] = np.sum(-J * signs)

    return energies

def process_specific_heat_data():
    alphavals = [2.5]
    Lvals = [16, 24, 32, 40]

    def _run_temp(tempind, temps, alpha, L, J, sq_lk_flat, sq_lk_len):
        print(f'L = {L}, alpha = {alpha}, tempind = {tempind}')
        systems = read_compressed_data(f'compressed/mc_snapshots_L{L}_alpha{alpha}_tempind{tempind}_10ktrials.npz')

        # remove padding bits
        systems = systems[:, :2*L**2]
        energies = get_energies(systems, J, sq_lk_flat, sq_lk_len)
        Cv = (np.mean(energies**2) - np.mean(energies)**2) / (temps[tempind]**2 * L**2)
        return Cv

    for alpha in alphavals:
        for L in Lvals:
            J, sq_lk_flat, sq_lk_len, _ = build_tables(L, alpha, Rmax=L//2)
            specific_heats = []
            temps = temps_list(alpha, which='pfail')

            specific_heats = Parallel(n_jobs=-1)(delayed(_run_temp)(tempind, temps, alpha, L, J , sq_lk_flat, sq_lk_len)
                        for tempind in range(len(temps)))

            with open(f'specific_heat_plot/specific_heat_alpha{alpha}_L{L}.p', 'wb') as h:
                pickle.dump([temps, specific_heats], h)
            print(f'Dumped L={L}, alpha={alpha}')


if __name__ == '__main__':
    process_specific_heat_data()


