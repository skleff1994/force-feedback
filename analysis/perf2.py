from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from croco_mpc_utils.ocp_core_data import load_data
import analysis_utils


import sys
import numpy as np
import matplotlib.pyplot as plt
from analysis_utils import linear_interpolation



PREFIX_1 = '/home/skleff/Desktop/soft_contact_sim_exp/with_torque_control/'
PREFIX_2 = '/home/skleff/Desktop/soft_contact_sim_exp/no_torque_control/'

data_1 = load_data(PREFIX_1+'errors_TORQUE_CONTROL=True_full.npz')
data_2 = load_data(PREFIX_2+'errors_TORQUE_CONTROL=False_full.npz')

# Compute perf ratios
SEEDS           = data_1['SEEDS']
TILT_ANGLES_DEG = data_1['TILT_ANGLES_DEG']
N_SEEDS = len(data_1['SEEDS'])
N_EXP   = len(data_1['TILT_ANGLES_DEG'])


torque_ratio_classical = np.zeros((N_SEEDS, N_EXP))
torque_ratio_lpf       = np.zeros((N_SEEDS, N_EXP))
torque_ratio_soft      = np.zeros((N_SEEDS, N_EXP))

force_ratio_classical = np.zeros((N_SEEDS, N_EXP))
force_ratio_lpf       = np.zeros((N_SEEDS, N_EXP))
force_ratio_soft      = np.zeros((N_SEEDS, N_EXP))

for n_seed in range(N_SEEDS):
    
    for n_exp in range(N_EXP):
        
        torque_ratio_classical[n_seed, n_exp] = data_1['torque_errors'][0][n_seed, n_exp] / data_2['torque_errors'][0][n_seed, n_exp]
        torque_ratio_lpf[n_seed, n_exp]       = data_1['torque_errors'][1][n_seed, n_exp] / data_2['torque_errors'][1][n_seed, n_exp]
        torque_ratio_soft[n_seed, n_exp]      = data_1['torque_errors'][2][n_seed, n_exp] / data_2['torque_errors'][2][n_seed, n_exp]
        
        force_ratio_classical[n_seed, n_exp] = data_1['force_errors_AVG'][0][n_seed, n_exp] / data_2['force_errors_AVG'][0][n_seed, n_exp]
        force_ratio_lpf[n_seed, n_exp]       = data_1['force_errors_AVG'][1][n_seed, n_exp] / data_2['force_errors_AVG'][1][n_seed, n_exp]
        force_ratio_soft[n_seed, n_exp]      = data_1['force_errors_AVG'][2][n_seed, n_exp] / data_2['force_errors_AVG'][2][n_seed, n_exp]

torque_ratio_classical_AVG = np.sum(torque_ratio_classical, axis=0) / N_SEEDS
torque_ratio_lpf_AVG       = np.sum(torque_ratio_lpf, axis=0) / N_SEEDS
torque_ratio_soft_AVG      = np.sum(torque_ratio_soft, axis=0) / N_SEEDS

force_ratio_classical_AVG = np.sum(force_ratio_classical, axis=0) / N_SEEDS
force_ratio_lpf_AVG       = np.sum(force_ratio_lpf, axis=0) / N_SEEDS
force_ratio_soft_AVG      = np.sum(force_ratio_soft, axis=0) / N_SEEDS

fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Torque errors

colors            = ['b', 'r', 'g']
labels            = ['Classical MPC', 'Torque-feedback MPC', 'Force-feedback MPC']
markers           = ['o', 's', 'D']
torque_ratio_AVG  = [torque_ratio_classical_AVG, torque_ratio_lpf_AVG, torque_ratio_soft_AVG]
force_ratio_AVG   = [force_ratio_classical_AVG, force_ratio_lpf_AVG, force_ratio_soft_AVG]

torque_ratios    = [torque_ratio_classical, torque_ratio_lpf, torque_ratio_soft]
force_ratios     = [force_ratio_classical, force_ratio_lpf, force_ratio_soft]

for i in range(3):
    # Plot average lines
    if(n_seed == N_SEEDS-1 and n_exp == N_EXP-1):
        lab = labels[i]
    else:
        lab = None
    ax0.plot(TILT_ANGLES_DEG, force_ratio_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    # ax0.plot(TILT_ANGLES_DEG, force_ratio_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)

    # For each experiment plot perf as marker 
    for n_exp in range(N_EXP): 
        ax0.plot(float(TILT_ANGLES_DEG[n_exp]), force_ratio_AVG[i][n_exp], 
                                                marker=markers[i], markerfacecolor=colors[i], 
                                                markersize=18, markeredgecolor='k')
        # Plot each seed exp with  transparent marker
        for n_seed in range(N_SEEDS):
            ax0.plot(TILT_ANGLES_DEG[n_exp], force_ratios[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            # ax0.plot(TILT_ANGLES_DEG[n_exp], torque_ratios[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            
# Set axis and stuff
ax0.set_ylabel('$|| \tau^{mea} - \tau^{des} ||$  (N)', fontsize=26)
ax0.yaxis.set_major_locator(plt.MaxNLocator(2))
ax0.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax0.grid(True) 
ax0.tick_params(axis = 'y', labelsize=22)
ax0.set_xlabel('Angle (deg)', fontsize=26)
ax0.tick_params(axis = 'x', labelsize = 22)
plt.show()