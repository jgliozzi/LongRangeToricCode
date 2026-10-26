""" Apply RG decoder to configurations of long-range toric code at different temperatures 
(generated from monte carlo snapshots)
"""
import numpy as np
from numba import njit
import pickle
from longtoric import *

# read my own compressed finite temp data
def read_compressed_data(file):
    data = np.load(file)
    packed = data["packed"]
    systems = np.unpackbits(packed, axis=1) # unpack bits
    return systems

"""
One Sang-Zou-Hsieh RG step on an L x L anyon array.

1. Odd 2x2 blocks:
   annihilate all anyons in the block if its parity is even;
   otherwise leave the block unchanged.

2. Even 2x2 blocks (coarse-grain):
   replace each even block by its parity.

Returns an (L/2) x (L/2) anyon array.
"""
@njit(inline='always', fastmath=True)
def rg_step(anyons, skip_odd=False):
    L = anyons.shape[0]
    a = anyons.copy()

    if not skip_odd:
        # odd blocks: shifted 2x2 tiling with PBC wrapping
        for x in range(1, L, 2):
            xp = (x + 1) if x < L-1 else 0
            for y in range(1, L, 2):
                yp = (y + 1) if y < L-1 else 0
                block_parity = (a[x, y] + a[xp, y] + a[x, yp] + a[xp, yp]) & 1
                # annihilate all anyons if parity is even
                if block_parity == 0:
                    a[x, y] = 0
                    a[xp, y] = 0
                    a[x, yp] = 0
                    a[xp, yp] = 0

    # even blocks: compute parity and coarse grain
    L2 = L // 2
    coarse = np.empty((L2, L2), dtype=np.bool_)

    # coarse graining means making new block 2n x 2n contain parity number of anyons
    for i in range(L2):
        x = 2 * i
        for j in range(L2):
            y = 2 * j
            coarse[i, j] = (a[x, y] + a[x + 1, y] + a[x, y + 1] + a[x + 1, y + 1]) & 1
    return coarse

@njit(fastmath=True)
def rg_flow(anyons, L, n_steps, skip_odd=False):
    configs = [anyons.copy()]
    densities = np.empty(n_steps + 1)
    densities[0] = np.mean(anyons)

    for step in range(n_steps):
        anyons = rg_step(anyons, skip_odd)
        configs.append(anyons.copy())
        densities[step + 1] = np.mean(anyons)

    return densities, configs

# do RG on finite temp config (hardcode some parameters)
def finite_temp_rg(n_trials=10000):
    L, alpha = 128, 2.5
    temps = temps_list(alpha, which='pfail')
    print(temps)
    tmin, tmax = 10, 30
    n_steps = int(np.log2(L) - np.log2(8))
    data = np.zeros((len(temps), n_steps+1)) # ending system must be bigger than 4 x 4

    for tempind, temp in enumerate(temps):
        if tempind >= tmin and tempind < tmax:
            print(f'Temperature: {temps[tempind]}')
            filename = f'compressed/mc_snapshots_L{L}_alpha{alpha}_tempind{tempind}_10ktrials.npz'
            systems = read_compressed_data(filename)
            for i, syst in enumerate(systems[:n_trials]):
                anyons = get_anyons(syst, L, magnetic=True, binary=True)
                dens, configs = rg_flow(anyons, L, n_steps, skip_odd=False) 
                data[tempind, :] += dens/n_trials

    des = 'description, temps (30,), data (30, 5) [anyon density after rg steps 0-4]'
    with open(f'rg_decoder/rg_L{L}_alpha{alpha}_20ktrials.p', 'wb') as h:
        pickle.dump([des, temps, data], h)
    print("Dumped")
    
if __name__ == '__main__':
    finite_temp_rg(10**4)
