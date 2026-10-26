# Make plots for long-range toric code
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# uncomment for nice production-grade plots
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 20
})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['text.usetex'] = True

# --------- PROCESSING RESULTS + PLOTTING----------- #

# THESE ARE THE TEMPERATURES USED IN THE DATA!!!
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

# binary encoded file w/ systems: 'L{L}_alpha{alpha}_temp{temp}_10ktrials.npz'
# how to remove padding bits (only needed if 2*L**2 isn't divisible by 8):
# systems = systems[:, :data["shape"][1]]
def read_compressed_data(file):
    data = np.load(file)
    packed = data["packed"]
    systems = np.unpackbits(packed, axis=1) # unpack bits
    return systems


# paper quality figure (adjust with inkscape)
def wilson_plot_contractible():
    Lvals = np.array([16, 24, 32, 40])
    alphavals = [1.8, 2.5, 4.0]
    fig2, ax2 = plt.subplots(2, 3, figsize=(18/1.5, 10/1.5))
    # xticks = np.arange(0.35, 0.701, 0.05)
    # xticklabels = ['', r'$0.4$', '', r'$0.5$', '', r'$0.6$', '', r'$0.7$']

    for aind, alpha in enumerate(alphavals):
        L = 64
        print(f'L = {L}')
        # requires user to combine raw data from diff suptrials into one file
        with open(f'../data/wilson_raw/wilson_varyR_L{L}_alpha{alpha}.p', 'rb') as h:
            data = pickle.load(h)

        des, temps, Rvals, wilson = data
        print(des)
        print(Rvals)
        inds_Rvals = np.arange(7, L//2-1, 1)
        Rvals = Rvals[inds_Rvals]
        wilson = wilson[:, inds_Rvals]

        for Rind, R in enumerate(Rvals):
            w = wilson[:, Rind]
            color = (Rind-.5)/(len(Rvals) - 3 )
            if aind < 2:
                ax2[0, aind].plot(temps[0::2], w[0::2], 'o-', ms=6, lw=2, c=plt.get_cmap('RdYlBu')(color), label=fr'$R = {R}$' if Rind % 8 == 0 or R == 31 else None, zorder=2)
                ax2[1, aind].plot(temps[0::2], -np.log(w[0::2])/R, 'o-', ms=6, lw=2, label=fr'$R = {R}$' if Rind % 4 == 0 else None, c=plt.get_cmap('RdYlBu')(color), zorder=2)
                ax2[1, aind].set_yticks([0, 0.3, 0.6])

            # ax2[2, aind].plot(temps[::2], w_log[::2]*R**alpha, 'o-', ms=6, lw=2, label=fr'$R = {R}$' if Rind % 4 == 2 else None, c=plt.get_cmap('RdYlBu')(color), zorder=2)
            else:
                if R < 30:
                    log_to_plot = -np.log(gaussian_filter1d(w, 0.5)) / R # apply gaussian filter to smooth very noise data near w=0 before taking log
                    for i, lw in enumerate(log_to_plot):
                        if i < 14:
                            log_to_plot[i] = -np.log(w[i]) / R
                        if np.isnan(lw):
                            log_to_plot[i] = np.nanmean(log_to_plot[i-1:i+2:2])
                    # for i in range(len(log_to_plot)):
                    #     if i < 15:
                    #         log_to_plot[i] = -np.log(w[i])/R 

                    ax2[0, aind].plot(temps[0::2], w[0::2], 'o-', ms=6, lw=2, c=plt.get_cmap('RdYlBu')(color), label=fr'$R = {R}$' if Rind % 4 == 0 else None, zorder=2)
                    ax2[1, aind].plot(temps[0::2], log_to_plot[0::2], 'o-', ms=6, lw=2, label=fr'$R = {R}$' if Rind % 4 == 0 else None, c=plt.get_cmap('RdYlBu')(color), zorder=2)
                    ax2[1, aind].set_yticks([0, 0.4, 0.8])
        
        ax2[1, aind].set_xlabel('Temperature', fontsize=24)
    ax2[0, 0].set_ylabel(r'$\langle W_R \rangle$', fontsize=26)
    ax2[1, 0].set_ylabel(r'$-\ln \langle W_R \rangle/R$', fontsize=26)

    for j in range(3):
        ax2[0, j].set_yticks([0, 0.5, 1.0])
    ax2[0,0].legend(fontsize=16, loc='lower left')
    ax2[0, 0].set_title(r'$\alpha = 1.8$', fontsize=26)
    ax2[0, 1].set_title(r'$\alpha = 2.5$', fontsize=26)
    ax2[0, 2].set_title(r'$\alpha = 4.0$', fontsize=26)

    fig2.tight_layout()
    plt.show()


# paper quality figure (adjust with inkscape)
# includes pfail (failure probability) and tmem (memory time)
# figure is plotted either with x-axis = 1/temp or temp
def memory_plots():
    prob_Lvals = [16, 24, 32, 40]
    time_Lvals = np.array([8, 12, 16, 20])
    alphavals = [1.8, 2.5, 4.0] # add 4.0 separately

    fig, ax = plt.subplots(2, 3, figsize=(18/1.5, 10/1.5))
    xticks = np.arange(0.35, 0.701, 0.05)
    xticklabels = ['', r'$0.4$', '', r'$0.5$', '', r'$0.6$', '', r'$0.7$']

    for aind, alpha in enumerate(alphavals):
        for Lind, L in enumerate(prob_Lvals):
            with open(f'../data/pfail_plot/pfail_plot_alpha{alpha}_L{L}.p', 'rb') as h:
                temps, failure_rates = pickle.load(h)

            color = np.array(plt.cm.tab10.colors[Lind])
            marker_face = 0.85*color + 0.15*np.ones(3)
            ax[0, aind].plot(temps, failure_rates, 'o-', markevery=2, label=fr'$L = {L}$', 
                color=color, markerfacecolor=marker_face, markeredgecolor=color, 
                markeredgewidth=1.5, markersize=8, linewidth=2.5)
        ax[0, aind].set_ylim([-0.02, 0.8])
        ax[0, aind].set_title(fr'$\alpha = {alpha}$', fontsize=26)

        temps_time = temps_list(alpha, which='tmem')
        plotdata = np.zeros((len(time_Lvals), len(temps_time)))
        for j, temp in enumerate(temps_time):
            Tmax = int(1.4e4)

            with open(f'../data/tmem_raw/tmem_alpha{alpha}_tempind{j}.p', 'rb') as h:
                fail_times = pickle.load(h)

            for i, L in enumerate(time_Lvals):
                frac_nan = np.mean(np.isnan(fail_times[i]))
                print(frac_nan)
                if frac_nan >= 0.1:
                    plotdata[i, j] = np.nan
                else:
                    capped_times = np.where(np.isnan(fail_times[i]), Tmax, fail_times[i])
                    plotdata[i, j] = np.mean(capped_times)

        colors2 = ['#9ECAE1', '#4292C6', '#1769AA', '#003F7D']
        marker_faces = ['#B5D8E8', '#63A3CD', '#4A91C0', '#28649A']

        for i, dataseries in enumerate(plotdata):
            ax[1, aind].semilogy(temps_time, dataseries, 'o-', 
                label=fr'$L = {time_Lvals[i]}$', color=colors2[i], markerfacecolor=marker_faces[i], 
                markeredgecolor=colors2[i], markeredgewidth=1.5, markersize=8, linewidth=2.5)

        ax[1, aind].set_xlabel('Temperature', fontsize=24)
    ax[1, 0].set_ylabel(r'$t_\mathrm{mem}$', fontsize=26)
    ax[0, 0].set_ylabel(r'$P_\mathrm{err}$', fontsize=26)
    for i in range(3):
        ax[0, i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    for i in range(3):
        ax[1, i].set_yticks([10, 10**2, 10**3, 10**4])
        ax[1, i].set_yticklabels(['', r'$10^2$', '', r'$10^4$'])
    fig.tight_layout()
    plt.show()

def wilson_plot_noncontractible():
    Lvals = np.array([16, 24, 32, 40])
    alphavals = [1.8, 2.5, 4.0] # do 4.0 separately
    fig2, ax2 = plt.subplots(1, 3, figsize=(12,4), sharey=True)
    # xticks = np.arange(0.35, 0.701, 0.05)
    # xticklabels = ['', r'$0.4$', '', r'$0.5$', '', r'$0.6$', '', r'$0.7$']

    for aind, alpha in enumerate(alphavals):
        for Lind, L in enumerate(Lvals):
            color = np.array(plt.cm.tab10.colors[Lind])
            marker_face = 0.85*color + 0.15*np.ones(3)

            with open(f'../data/wilson_noncontractible_raw/wilson_noncontractible_L{L}_alpha{alpha}.p', 'rb') as h:
                data = pickle.load(h)
            des, temps, w = data

            color = np.array(plt.cm.tab10.colors[Lind])
            marker_face = 0.85*color + 0.15*np.ones(3)

            ax2[aind].plot(temps[:], w[:], 'o-', markevery=2, label=fr'$L = {L}$', 
                color=color, markerfacecolor=marker_face, markeredgecolor=color, markeredgewidth=1.5, markersize=8, linewidth=2.5)
            # ax2[aind].plot(temps[:], -np.log(w[:])/L**2, 'o-', label=fr'$L = {L}$')
        ax2[aind].set_xlabel('Temperature', fontsize=24)
    
    ax2[0].set_ylabel(r'$\langle W_y \rangle$', fontsize=26)
    # ax2[1, 0].set_ylabel(r'$-\log \langle W \rangle/L$', fontsize=26)
    ax2[0].legend(fontsize=16, loc='lower left')
    ax2[0].set_title(r'$\alpha = 1.8$', fontsize=26)
    ax2[1].set_title(r'$\alpha = 2.5$', fontsize=26)
    ax2[2].set_title(r'$\alpha = 4.0$', fontsize=26)
    fig2.tight_layout()
    plt.show()

def specific_heat_plots():
    alphavals = [1.8, 2.5, 4.0]
    Lvals = [16, 24, 32, 40]
    fig, ax = plt.subplots(1, len(alphavals), figsize=(12, 4), sharey=False)
    for j, alpha in enumerate(alphavals):
        for Lind, L in enumerate(Lvals):
            with open(f'../data/specific_heat_plot/specific_heat_alpha{alpha}_L{L}.p', 'rb') as h:
                temps, Cv = pickle.load(h)
            tmin, tmax = 4,24
            color = np.array(plt.cm.tab10.colors[Lind])
            marker_face = 0.85*color + 0.15*np.ones(3)
            show_markers = list(range(0, 13, 2)) + list(range(13,17,1)) + list(range(17,30,2))
            show_markers = list(range(0,30))
            if alpha == 2.5 or alpha == 1.8:
                ax[j].plot(temps[tmin:tmax], Cv[tmin:tmax], 'o-', markevery=1, 
                    label=fr'$L = {L}$', color=color, markerfacecolor=marker_face, markeredgecolor=color, 
                    markeredgewidth=1.5, markersize=8, linewidth=2.5)
            else:
                ax[j].plot(temps, Cv, 'o-', markevery=2, 
                    label=fr'$L = {L}$', color=color, markerfacecolor=marker_face, markeredgecolor=color, 
                    markeredgewidth=1.5, markersize=8, linewidth=2.5)

        ax[j].set_title(rf'$\alpha={alpha}$', fontsize=26)
        ax[j].set_xlabel(f'Temperature', fontsize=26)
    ax[1].legend(fontsize=16)
    ax[0].set_ylabel(r'$C/L^2$', fontsize=26)
    plt.tight_layout()
    plt.show()

# do RG on finite temp config (hardcode some parameters)
def rg_decoder_plot():
    L, alpha = 128, 2.5
    temps = temps_list(alpha, which='pfail')
    tmin, tmax = 11, 20 # range of temperatures to include in plot (stay near Tc)

    des = 'description, temps (30,), data (30, 5) [anyon density after rg steps 0-4]'
    with open(f'../data/rg_decoder/rg_L{L}_alpha{alpha}_20ktrials.p', 'rb') as h:
        des, temps, data = pickle.load(h)
    n_steps = data.shape[-1] - 1
    print("Loaded data")
    
    fig, ax = plt.subplots(1,2, figsize=(9.5, 4))
    for k in range(0, n_steps+1):
        color = np.array(plt.cm.tab10.colors[k])
        marker_face = 0.9*color + 0.1*np.ones(3)

        ax[0].semilogy(temps[tmin:tmax], data[tmin:tmax, k], 'o-', label=fr'$k = {k}$', 
            color=color, markerfacecolor=marker_face, markeredgecolor=color, 
                    markeredgewidth=1.5, markersize=8, linewidth=2.5)
    ax[0].legend()
    ax[0].set_xlabel(r'Temperature $(T)$', fontsize=26)
    ax[0].set_ylabel(r'$n_{k}$', fontsize=30)
    ax[0].legend(fontsize=16)

    tind_to_plot = [12, 13, 14]
    colors = plt.cm.tab10.colors
    scatter_xy = []
    for i, tind in enumerate(tind_to_plot):
        T = temps[tind]
        for j in range(1, data.shape[1] - 1):
            ax[1].scatter(data[tind, j], data[tind, j+1], color=colors[i], 
                label=fr'$T = {np.round(T,2)}$' if j==1 else None, zorder=2, s=90)
            scatter_xy.append([np.log(data[tind, j]), np.log(data[tind, j+1])])
    scatter_xy = np.array(scatter_xy)
    m, b = np.polyfit(scatter_xy[:, 0], scatter_xy[:, 1], 1)
    lims = np.logspace(-5, -2, 100)
    ax[1].plot(lims, np.exp(b)*lims**m, label=fr'$n_{{k+1}} \sim n_{{k}}^{{{m:.1f}}}$', 
        linestyle='dotted', c='black', zorder=1, linewidth=2.5)
    ax[1].plot(lims, lims, '--', label=r'$n_{k+1}= n_{k}$', c='red', linewidth=3)
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$n_{k}$', fontsize=30)
    ax[1].set_ylabel(r'$n_{k+1}$', fontsize=30)
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    wilson_plot_cold()

