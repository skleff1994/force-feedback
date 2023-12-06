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
        
        torque_ratio_classical[n_seed, n_exp] = data_2['torque_errors'][0][n_seed, n_exp] - data_1['torque_errors'][0][n_seed, n_exp]
        torque_ratio_lpf[n_seed, n_exp]       = data_2['torque_errors'][1][n_seed, n_exp] - data_1['torque_errors'][1][n_seed, n_exp]
        torque_ratio_soft[n_seed, n_exp]      = data_2['torque_errors'][2][n_seed, n_exp] - data_1['torque_errors'][2][n_seed, n_exp]
        
        force_ratio_classical[n_seed, n_exp] = data_2['force_errors_AVG'][0][n_seed, n_exp] - data_1['force_errors_AVG'][0][n_seed, n_exp]
        force_ratio_lpf[n_seed, n_exp]       = data_2['force_errors_AVG'][1][n_seed, n_exp] - data_1['force_errors_AVG'][1][n_seed, n_exp]
        force_ratio_soft[n_seed, n_exp]      = data_2['force_errors_AVG'][2][n_seed, n_exp] - data_1['force_errors_AVG'][2][n_seed, n_exp]

torque_ratio_classical_AVG = np.sum(torque_ratio_classical, axis=0) / N_SEEDS
torque_ratio_lpf_AVG       = np.sum(torque_ratio_lpf, axis=0) / N_SEEDS
torque_ratio_soft_AVG      = np.sum(torque_ratio_soft, axis=0) / N_SEEDS

force_ratio_classical_AVG = np.sum(force_ratio_classical, axis=0) / N_SEEDS
force_ratio_lpf_AVG       = np.sum(force_ratio_lpf, axis=0) / N_SEEDS
force_ratio_soft_AVG      = np.sum(force_ratio_soft, axis=0) / N_SEEDS

# Force & position aboslute perf 1 (with torque control)
print("--------")
print("--------")
print("[WITH TORQUE CONTROL] Position MAE (classical) = ", np.mean(data_1['position_errors'][0]), u"\u00B1", np.std(data_1['position_errors'][0]))
print("[WITH TORQUE CONTROL] Position MAE (LPF)       = ", np.mean(data_1['position_errors'][1]), u"\u00B1", np.std(data_1['position_errors'][0]))
print("[WITH TORQUE CONTROL] Position MAE (soft)      = ", np.mean(data_1['position_errors'][2]), u"\u00B1", np.std(data_1['position_errors'][0]))
print("----")
print("[WITH TORQUE CONTROL] Force MAE (classical) = ", np.mean(data_1['force_errors_AVG'][0]), u"\u00B1", np.std(data_1['force_errors_AVG'][0]))
print("[WITH TORQUE CONTROL] Force MAE (LPF)       = ", np.mean(data_1['force_errors_AVG'][1]), u"\u00B1", np.std(data_1['force_errors_AVG'][1]))
print("[WITH TORQUE CONTROL] Force MAE (soft)      = ", np.mean(data_1['force_errors_AVG'][2]), u"\u00B1", np.std(data_1['force_errors_AVG'][2]))
print("----")
print("[WITH TORQUE CONTROL] Force Max. (classical) = ", np.mean(data_1['force_errors_MAX'][0]), u"\u00B1", np.std(data_1['force_errors_MAX'][0]))
print("[WITH TORQUE CONTROL] Force Max. (LPF)       = ", np.mean(data_1['force_errors_MAX'][1]), u"\u00B1", np.std(data_1['force_errors_MAX'][1]))
print("[WITH TORQUE CONTROL] Force Max. (soft)      = ", np.mean(data_1['force_errors_MAX'][2]), u"\u00B1", np.std(data_1['force_errors_MAX'][2]))
print("----")
print("[WITH TORQUE CONTROL] Time not-in-contact (classical) = ", np.mean(data_1['cycles_not_in_contact'][0]), u"\u00B1", np.std(data_1['cycles_not_in_contact'][0]))
print("[WITH TORQUE CONTROL] Time not-in-contact (LPF)       = ", np.mean(data_1['cycles_not_in_contact'][1]), u"\u00B1", np.std(data_1['cycles_not_in_contact'][1]))
print("[WITH TORQUE CONTROL] Time not-in-contact (soft)      = ", np.mean(data_1['cycles_not_in_contact'][2]), u"\u00B1", np.std(data_1['cycles_not_in_contact'][2]))
print("--------")
print("--------")

# Force & position aboslute perf 2 (with torque control)
print("--------")
print("--------")
print("[NO TORQUE CONTROL] Position MAE (classical) = ", np.mean(data_2['position_errors'][0]), u"\u00B1", np.std(data_2['position_errors'][0]))
print("[NO TORQUE CONTROL] Position MAE (LPF)       = ", np.mean(data_2['position_errors'][1]), u"\u00B1", np.std(data_2['position_errors'][0]))
print("[NO TORQUE CONTROL] Position MAE (soft)      = ", np.mean(data_2['position_errors'][2]), u"\u00B1", np.std(data_2['position_errors'][0]))
print("----")
print("[NO TORQUE CONTROL] Force MAE (classical) = ", np.mean(data_2['force_errors_AVG'][0]), u"\u00B1", np.std(data_2['force_errors_AVG'][0]))
print("[NO TORQUE CONTROL] Force MAE (LPF)       = ", np.mean(data_2['force_errors_AVG'][1]), u"\u00B1", np.std(data_2['force_errors_AVG'][1]))
print("[NO TORQUE CONTROL] Force MAE (soft)      = ", np.mean(data_2['force_errors_AVG'][2]), u"\u00B1", np.std(data_2['force_errors_AVG'][2]))
print("----")
print("[NO TORQUE CONTROL] Force Max. (classical) = ", np.mean(data_2['force_errors_MAX'][0]), u"\u00B1", np.std(data_2['force_errors_MAX'][0]))
print("[NO TORQUE CONTROL] Force Max. (LPF)       = ", np.mean(data_2['force_errors_MAX'][1]), u"\u00B1", np.std(data_2['force_errors_MAX'][1]))
print("[NO TORQUE CONTROL] Force Max. (soft)      = ", np.mean(data_2['force_errors_MAX'][2]), u"\u00B1", np.std(data_2['force_errors_MAX'][2]))
print("----")
print("[NO TORQUE CONTROL] Time not-in-contact (classical) = ", np.mean(data_2['cycles_not_in_contact'][0]), u"\u00B1", np.std(data_2['cycles_not_in_contact'][0]))
print("[NO TORQUE CONTROL] Time not-in-contact (LPF)       = ", np.mean(data_2['cycles_not_in_contact'][1]), u"\u00B1", np.std(data_2['cycles_not_in_contact'][1]))
print("[NO TORQUE CONTROL] Time not-in-contact (soft)      = ", np.mean(data_2['cycles_not_in_contact'][2]), u"\u00B1", np.std(data_2['cycles_not_in_contact'][2]))
print("--------")
print("--------")
# # Force & position aboslute perf 2 (without torque control)
# print("Classical torque perf change = ", np.mean(torque_ratio_classical_AVG))
# print("LPF torque perf change = ",  np.mean(torque_ratio_lpf_AVG))
# print("Soft torque perf change = ",  np.mean(torque_ratio_soft_AVG))
# print("----")

# # aboslute perf
# print("Classical force perf change = ", np.mean(force_ratio_classical_AVG))
# print("LPF force perf change = ",  np.mean(force_ratio_lpf_AVG))
# print("Soft force perf change = ",  np.mean(force_ratio_soft_AVG))
# print("----")

# print("Classical torque perf change = ", np.mean(torque_ratio_classical_AVG))
# print("LPF torque perf change = ",  np.mean(torque_ratio_lpf_AVG))
# print("Soft torque perf change = ",  np.mean(torque_ratio_soft_AVG))
# print("----")

# print("Classical perf ratio = ", np.mean(force_ratio_classical_AVG)/np.mean(torque_ratio_classical_AVG))
# print("LPF perf ratio = ",  np.mean(force_ratio_lpf_AVG)/np.mean(torque_ratio_lpf_AVG))
# print("Soft perf ratio = ",  np.mean(force_ratio_soft_AVG)/np.mean(torque_ratio_soft_AVG))

# fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Torque errors

# colors            = ['b', 'r', 'g']
# labels            = ['Classical MPC', 'Torque-feedback MPC', 'Force-feedback MPC']
# markers           = ['o', 's', 'D']
# torque_ratio_AVG  = [torque_ratio_classical_AVG, torque_ratio_lpf_AVG, torque_ratio_soft_AVG]
# force_ratio_AVG   = [force_ratio_classical_AVG, force_ratio_lpf_AVG, force_ratio_soft_AVG]

# torque_ratios    = [torque_ratio_classical, torque_ratio_lpf, torque_ratio_soft]
# force_ratios     = [force_ratio_classical, force_ratio_lpf, force_ratio_soft]

# for i in range(3):
#     # Plot average lines
#     if(n_seed == N_SEEDS-1 and n_exp == N_EXP-1):
#         lab = labels[i]
#     else:
#         lab = None
#     ax0.plot(TILT_ANGLES_DEG, force_ratio_AVG[i]/torque_ratio_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
#     # ax0.plot(TILT_ANGLES_DEG, force_ratio_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)

#     # For each experiment plot perf as marker 
#     for n_exp in range(N_EXP): 
#         ax0.plot(float(TILT_ANGLES_DEG[n_exp]), force_ratio_AVG[i][n_exp]/torque_ratio_AVG[i][n_exp], 
#                                                 marker=markers[i], markerfacecolor=colors[i], 
#                                                 markersize=18, markeredgecolor='k')
#         # Plot each seed exp with  transparent marker
#         for n_seed in range(N_SEEDS):
#             ax0.plot(TILT_ANGLES_DEG[n_exp], force_ratios[i][n_seed,n_exp]/torque_ratios[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
#             # ax0.plot(TILT_ANGLES_DEG[n_exp], torque_ratios[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            
# # Set axis and stuff
# ax0.set_ylabel('Performance ratio', fontsize=26)
# ax0.yaxis.set_major_locator(plt.MaxNLocator(2))
# ax0.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
# ax0.grid(True) 
# ax0.tick_params(axis = 'y', labelsize=22)
# ax0.set_xlabel('Angle (deg)', fontsize=26)
# ax0.tick_params(axis = 'x', labelsize = 22)
# plt.show()