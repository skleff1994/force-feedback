
# which_plots         = ['f'] 
# PLOT_PREDICTIONS    = True 
# pred_plot_sampling  = 100 
# SAVE                = False
# SAVE_DIR            = None 
# SAVE_NAME           = None
# SHOW                = True
# AUTOSCALE           = False
# args = [which_plots, PLOT_PREDICTIONS, pred_plot_sampling, SAVE, SAVE_DIR, SAVE_NAME, SHOW, AUTOSCALE]

# sd1.plot_mpc_results(d1, *args)


from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from croco_mpc_utils.ocp_core_data import load_data
import analysis_utils


import sys
import numpy as np
import matplotlib.pyplot as plt



PREFIX = '/home/skleff/Desktop/soft_contact_sim_exp/with_torque_control/'
prefix_lpf       = PREFIX+'iiwa_LPF_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=1.0_Fs2.0'
prefix_soft      = PREFIX+'iiwa_aug_soft_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=1.0_Fs2.0'
prefix_classical = PREFIX+'iiwa_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=1.0_Fs2.0'

CUTOFF = 3. # in seconds

# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [6, 4, 2, 0, -2, -4, -6] 
TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([angle*np.pi/180, 0., 0.])
N_EXP = len(TILT_RPY)
SEEDS = [19, 71, 89, 83, 41, 73, 17, 47, 29, 7]
N_SEEDS = len(SEEDS)
  
position_error_AVG_NORM_classical = np.zeros((N_SEEDS, N_EXP))
force_error_AVG_classical         = np.zeros((N_SEEDS, N_EXP))
force_error_MAX_classical         = np.zeros((N_SEEDS, N_EXP))
cycles_not_in_contact_classical   = np.zeros((N_SEEDS, N_EXP))

position_error_AVG_NORM_lpf = np.zeros((N_SEEDS, N_EXP))
force_error_AVG_lpf         = np.zeros((N_SEEDS, N_EXP))
force_error_MAX_lpf         = np.zeros((N_SEEDS, N_EXP))
cycles_not_in_contact_lpf   = np.zeros((N_SEEDS, N_EXP))

position_error_AVG_NORM_soft = np.zeros((N_SEEDS, N_EXP))
force_error_AVG_soft         = np.zeros((N_SEEDS, N_EXP))
force_error_MAX_soft         = np.zeros((N_SEEDS, N_EXP))
cycles_not_in_contact_soft   = np.zeros((N_SEEDS, N_EXP))


# Compute errors 
FILTER  = 20

for n_seed in range(N_SEEDS):

    logger.debug("Seed "+str(n_seed+1) + "/" + str(N_SEEDS))

    for n_exp in range(N_EXP):

        logger.debug("Experiment n°"+str(n_exp+1)+"/"+str(N_EXP))
        # Extract data classical
        sd   = load_data(prefix_classical+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED='+str(SEEDS[n_seed])+'.npz')
        # sd   = load_data(prefix_classical+'.npz')
        data = sd#.extract_data(frame_of_interest='contact')
        # Smooth if necessary
        if(FILTER > 0):
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
        # Compute absolute tracking errors |mea - ref|
        N_START_SIMU = int(CUTOFF*data['simu_freq'])
        N_START_PLAN = int(CUTOFF*data['plan_freq'])
        Np = data['N_plan'] - N_START_PLAN
        Ns = data['N_simu'] - N_START_SIMU
        position_error = 0.
        for i in range( Ns ):
            position_error += np.linalg.norm( data['lin_pos_ee_mea'][i+N_START_SIMU,:2] - data['lin_pos_ee_ref'][int(i*Np/Ns)+N_START_PLAN,:2])
        # Average absolute error 
        position_error_AVG_NORM_classical[n_seed, n_exp] = position_error / Ns 
        print("Ns = ", Ns)
        # Force tracking
        force_reference = data['frameForceRef'][2] 
        force_error = np.zeros(Ns)
        for i in range( Ns ):
            force_error[i] = np.linalg.norm( data['f_ee_mea'][i+N_START_SIMU,2] - force_reference)
        # Maximum (peak) absolute error along x,y,z
        force_error_MAX_classical[n_seed, n_exp] = np.max(force_error)
        # Average absolute error 
        force_error_AVG_classical[n_seed, n_exp] = np.sum(force_error, axis=0) / Ns
        # Is in contact
        bool_contact = np.isclose(data['f_ee_mea'][N_START_SIMU:,2], np.zeros(data['f_ee_mea'][N_START_SIMU:,2].shape), rtol=1e-3)
        cycles_not_in_contact_classical[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
        logger.warning("Classical MPC avg position error  = "+str(position_error_AVG_NORM_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC avg force error     = "+str(force_error_AVG_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC max force           = "+str(force_error_MAX_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC not-in-contact rate = "+str(cycles_not_in_contact_classical[n_seed, n_exp] ))


        # Extract LPF
        sd   = load_data(prefix_lpf+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED='+str(SEEDS[n_seed])+'.npz')
        # sd   = load_data(prefix_lpf+'.npz')
        data = sd#.extract_data(frame_of_interest='contact')
        # Smooth if necessary
        if(FILTER > 0):
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
        # Compute absolute tracking errors |mea - ref|
        N_START_SIMU = int(CUTOFF*data['simu_freq'])
        N_START_PLAN = int(CUTOFF*data['plan_freq'])
        Np = data['N_plan'] - N_START_PLAN
        Ns = data['N_simu'] - N_START_SIMU
        position_error = 0.
        for i in range( Ns ):
            position_error += np.linalg.norm( data['lin_pos_ee_mea'][i+N_START_SIMU,:2] - data['lin_pos_ee_ref'][int(i*Np/Ns)+N_START_PLAN,:2])
        # Average absolute error 
        position_error_AVG_NORM_lpf[n_seed, n_exp] = position_error / Ns 
        print("Ns = ", Ns)
        # Force tracking
        force_reference = data['frameForceRef'][2] 
        force_error = np.zeros(Ns)
        for i in range( Ns ):
            force_error[i] = np.abs( data['f_ee_mea'][i+N_START_SIMU,2] - force_reference)
        # Maximum (peak) absolute error along x,y,z
        force_error_MAX_lpf[n_seed, n_exp]   = np.max(force_error)
        # Average absolute error 
        force_error_AVG_lpf[n_seed, n_exp] = np.sum(force_error, axis=0) / Ns
        # Is in contact
        bool_contact = np.isclose(data['f_ee_mea'][N_START_SIMU:,2], np.zeros(data['f_ee_mea'][N_START_SIMU:,2].shape), rtol=1e-3)
        cycles_not_in_contact_lpf[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
        logger.warning("LPF MPC avg position error  = "+str(position_error_AVG_NORM_lpf[n_seed, n_exp] ))
        logger.warning("LPF MPC avg force error     = "+str(force_error_AVG_lpf[n_seed, n_exp] ))
        logger.warning("LPF MPC max force           = "+str(force_error_MAX_lpf[n_seed, n_exp] ))
        logger.warning("LPF MPC not-in-contact rate = "+str(cycles_not_in_contact_lpf[n_seed, n_exp] ))     
        
        # slfdibfd
        # Extract soft 
        sd   = load_data(prefix_soft+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED='+str(SEEDS[n_seed])+'.npz')
        # sd   = load_data(prefix_soft+'.npz')
        data = sd#.extract_data(frame_of_interest='contact')
        # Smooth if necessary
        if(FILTER > 0):
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
        # Compute absolute tracking errors |mea - ref|
        N_START_SIMU = int(CUTOFF*data['simu_freq'])
        N_START_PLAN = int(CUTOFF*data['plan_freq'])
        Np = data['N_plan'] - N_START_PLAN
        Ns = data['N_simu'] - N_START_SIMU
        position_error = 0.
        for i in range( Ns ):
            position_error += np.linalg.norm( data['lin_pos_ee_mea'][i+N_START_SIMU,:2] - data['lin_pos_ee_ref'][int(i*Np/Ns)+N_START_PLAN,:2])
        # Average absolute error 
        position_error_AVG_NORM_soft[n_seed, n_exp] = position_error / Ns
        # Force tracking
        force_reference = data['frameForceRef'][2] 
        force_error = np.zeros( Ns )
        for i in range( Ns ):
            force_error[i] = np.linalg.norm( data['f_ee_mea'][i+N_START_SIMU] - force_reference)
        # Maximum (peak) absolute error along x,y,z
        force_error_MAX_soft[n_seed, n_exp]   = np.max(force_error)
        # Average absolute error 
        force_error_AVG_soft[n_seed, n_exp] = np.sum(force_error, axis=0) / Ns
        # Is in contact
        bool_contact = np.isclose(data['f_ee_mea'][N_START_SIMU:], np.zeros(data['f_ee_mea'][N_START_SIMU:].shape), rtol=1e-3)
        cycles_not_in_contact_soft[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
        # print(cycles_not_in_contact_soft)
        logger.warning("Soft MPC avg position error  = "+str(position_error_AVG_NORM_soft[n_seed, n_exp] ))
        logger.warning("Soft MPC avg force error     = "+str(force_error_AVG_soft[n_seed, n_exp] ))
        logger.warning("Soft MPC max force           = "+str(force_error_MAX_soft[n_seed, n_exp] ))
        logger.warning("Soft MPC not-in-contact rate = "+str(cycles_not_in_contact_soft[n_seed, n_exp] ))    
        
        # #  Plot position reference and errors
        # fig, ax = plt.subplots(2, 1, figsize=(19.2,10.8)) 
        # tspan = np.linspace(0, data['T_tot'], position_reference.shape[0])
        # # ax[0].plot(tspan, position_reference[:,0], label='ref_x')
        # ax[0].plot(tspan, position_error[:,1], label='error_x')
        # # ax[1].plot(tspan, position_reference[:,1], label='ref_y')
        # ax[1].plot(tspan, position_error[:,1], label='error_y')
        # plt.show()


# Plot 
fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Err position 
fig2, ax2 = plt.subplots(2, 1, figsize=(19.2,10.8)) # Err force + max force 
fig3, ax3 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Timings


# Average perfs over seeds
position_error_AVG_NORM_classical_AVG = np.sum(position_error_AVG_NORM_classical, axis=0) / N_SEEDS
position_error_AVG_NORM_lpf_AVG       = np.sum(position_error_AVG_NORM_lpf, axis=0) / N_SEEDS
position_error_AVG_NORM_soft_AVG      = np.sum(position_error_AVG_NORM_soft, axis=0) / N_SEEDS

force_error_AVG_classical_AVG = np.sum(force_error_AVG_classical, axis=0) / N_SEEDS
force_error_AVG_lpf_AVG       = np.sum(force_error_AVG_lpf, axis=0) / N_SEEDS
force_error_AVG_soft_AVG      = np.sum(force_error_AVG_soft, axis=0) / N_SEEDS

force_error_MAX_classical_AVG = np.sum(force_error_MAX_classical, axis=0) / N_SEEDS
force_error_MAX_lpf_AVG       = np.sum(force_error_MAX_lpf, axis=0) / N_SEEDS
force_error_MAX_soft_AVG      = np.sum(force_error_MAX_soft, axis=0) / N_SEEDS

cycles_not_in_contact_classical_AVG = np.sum(cycles_not_in_contact_classical, axis=0) / N_SEEDS
cycles_not_in_contact_lpf_AVG       = np.sum(cycles_not_in_contact_lpf, axis=0) / N_SEEDS
cycles_not_in_contact_soft_AVG      = np.sum(cycles_not_in_contact_soft, axis=0) / N_SEEDS

colors                    = ['b', 'r', 'g']
labels                    = ['Classical MPC', 'Torque-feedback MPC', 'Force-feedback MPC']
markers                   = ['o', 's', 'D']
position_errors_AVG         = [position_error_AVG_NORM_classical_AVG, position_error_AVG_NORM_lpf_AVG, position_error_AVG_NORM_soft_AVG]
force_errors_AVG_AVG        = [force_error_AVG_classical_AVG, force_error_AVG_lpf_AVG, force_error_AVG_soft_AVG]
force_errors_MAX_AVG        = [force_error_MAX_classical_AVG, force_error_MAX_lpf_AVG, force_error_MAX_soft_AVG]
cycles_not_in_contact_AVG   = [cycles_not_in_contact_classical_AVG, cycles_not_in_contact_lpf_AVG, cycles_not_in_contact_soft_AVG]

position_errors        = [position_error_AVG_NORM_classical, position_error_AVG_NORM_lpf, position_error_AVG_NORM_soft]
force_errors_AVG       = [force_error_AVG_classical, force_error_AVG_lpf, force_error_AVG_soft]
force_errors_MAX       = [force_error_MAX_classical, force_error_MAX_lpf, force_error_MAX_soft]
cycles_not_in_contact  = [cycles_not_in_contact_classical, cycles_not_in_contact_lpf, cycles_not_in_contact_soft]


for i in range(3):
    # Plot average lines
    if(n_seed == N_SEEDS-1 and n_exp == N_EXP-1):
        lab = labels[i]
    else:
        lab = None
    ax1.plot(TILT_ANGLES_DEG, position_errors_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    ax2[0].plot(TILT_ANGLES_DEG, force_errors_AVG_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    ax2[1].plot(TILT_ANGLES_DEG, force_errors_MAX_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    ax3.plot(TILT_ANGLES_DEG, cycles_not_in_contact_AVG[i], color=colors[i], linestyle='-', linewidth=4, label=lab)

    # For each experiment plot perf as marker 
    for n_exp in range(N_EXP): 
        ax1.plot(float(TILT_ANGLES_DEG[n_exp]), position_errors_AVG[i][n_exp], 
                                                marker=markers[i], markerfacecolor=colors[i], 
                                                markersize=18, markeredgecolor='k')
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), force_errors_AVG_AVG[i][n_exp], 
                                                        marker=markers[i], markerfacecolor=colors[i],
                                                        markersize=18, markeredgecolor='k')
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), force_errors_MAX_AVG[i][n_exp], 
                                                        marker=markers[i], markerfacecolor=colors[i],
                                                        markersize=18, markeredgecolor='k')
        ax3.plot(float(TILT_ANGLES_DEG[n_exp]), cycles_not_in_contact_AVG[i][n_exp], 
                                                        marker=markers[i], markerfacecolor=colors[i],
                                                        markersize=18, markeredgecolor='k')
        # Plot each seed exp with  transparent marker
        for n_seed in range(N_SEEDS):
            ax1.plot(TILT_ANGLES_DEG[n_exp], position_errors[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            ax2[0].plot(TILT_ANGLES_DEG[n_exp], force_errors_AVG[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            ax2[1].plot(TILT_ANGLES_DEG[n_exp], force_errors_MAX[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            ax3.plot(TILT_ANGLES_DEG[n_exp], cycles_not_in_contact[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)

# Set axis and stuff
ax1.set_ylabel('$|| \Delta P^{EE}_{xy} ||$  (m)', fontsize=26)
ax1.yaxis.set_major_locator(plt.MaxNLocator(2))
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax1.grid(True) 
ax1.tick_params(axis = 'y', labelsize=22)
ax1.set_xlabel('Angle (deg)', fontsize=26)
ax1.tick_params(axis = 'x', labelsize = 22)


ax2[0].set_ylabel('$\Delta \lambda_{z}$  (m)', fontsize=26)
ax2[0].yaxis.set_major_locator(plt.MaxNLocator(2))
ax2[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax2[0].grid(True) 
ax2[0].tick_params(axis = 'y', labelsize=22)
ax2[0].xaxis.set_ticklabels([])
ax2[0].tick_params(bottom=False)

ax2[1].set_ylabel('$\lambda^{max}_{z}$  (m)', fontsize=26)
ax2[1].yaxis.set_major_locator(plt.MaxNLocator(2))
ax2[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax2[1].grid(True) 
ax2[1].tick_params(axis = 'y', labelsize=22)
ax2[1].set_xlabel('Angle (deg)', fontsize=26)
ax2[1].tick_params(axis = 'x', labelsize = 22)

ax3.set_ylabel('Simulation cycles (%)', fontsize=30)
ax3.yaxis.set_major_locator(plt.MaxNLocator(10))
ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax3.set_yscale('symlog')
ax3.grid(True) 
ax3.tick_params(axis = 'y', labelsize=24)
ax3.set_xlabel('Angle (deg)', fontsize=30)
ax3.tick_params(axis = 'x', labelsize = 24)

# Legend error
handles1, labels1 = ax1.get_legend_handles_labels()
fig1.legend(handles1, labels1, loc='upper right', prop={'size': 26})
# Legend error norm 
handles2, labels2 = ax2[0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='upper right', prop={'size': 26})
# Legend contacts
handles3, labels3 = ax3.get_legend_handles_labels()
fig3.legend(handles3, labels3, loc='upper right', prop={'size': 26})
# Save, show , clean
fig1.savefig(PREFIX+'pos_err_test_'+str(CUTOFF)+'_new.png')
fig2.savefig(PREFIX+'force_err_test_'+str(CUTOFF)+'_new.png')
fig3.savefig(PREFIX+'contact_timings_test_'+str(CUTOFF)+'_new.png')
plt.show()
plt.close('all')

