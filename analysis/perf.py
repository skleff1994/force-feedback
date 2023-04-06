
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


from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from core_mpc.data import load_data
from core_mpc import analysis_utils


import sys
import numpy as np
import matplotlib.pyplot as plt


PREFIX = '/home/skleff/force-feedback/data/soft_contact_article/'
prefix_lpf       = PREFIX+'iiwa_LPF_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=1.0_Fc=1.0_Fs5.0'
prefix_soft      = PREFIX+'iiwa_aug_soft_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=1.0_Fc=1.0_Fs5.0'
prefix_classical = PREFIX+'iiwa_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=1.0_Fc=1.0_Fs5.0'


TILT_ANGLES_DEG = [15, 10, 5, 0, -5, -10, -15] 
TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([angle*np.pi/180, 0., 0.])
N_EXP = len(TILT_RPY)
SEEDS = [1] #, 2, 3, 4, 5]
N_SEEDS = len(SEEDS)



# def main(FILTER=1, PLOT=False):
  
position_error_AVG = np.zeros((2,N_EXP))
position_error_AVG_NORM = np.zeros(N_EXP)
force_error_AVG = np.zeros(N_EXP)
force_error_MAX = np.zeros(N_EXP)

lin_err_ee_xy_avg_LPF = np.zeros((2,N_EXP))
lin_err_ee_xy_avg_LPF_NORM = np.zeros(N_EXP)
f_ee_err_avg_z_LPF = np.zeros(N_EXP)
f_ee_err_max_z_LPF = np.zeros(N_EXP)

# Compute errors 
FILTER  = 1
# N_START = []

for n_exp in range(N_EXP):


    # Extract data LPF
    sd   = load_data(prefix_soft+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED=1.npz')
    data = sd.extract_data(frame_of_interest='contact')
    # Compute absolute tracking errors |mea - ref|
    Np = data['N_plan'] ; Ns = data['N_simu']
    N_START = int(data['T_CONTACT']*data['simu_freq'])
    # Duplicate last element
    lin_pos_ee_ref = np.zeros((data['lin_pos_ee_ref'].shape[0]+1, data['lin_pos_ee_ref'].shape[1]))
    lin_pos_ee_ref[:data['lin_pos_ee_ref'].shape[0], :] = data['lin_pos_ee_ref']
    lin_pos_ee_ref[-1,:] = data['lin_pos_ee_ref'][-1,:]
    position_reference = analysis_utils.linear_interpolation(lin_pos_ee_ref, int(Ns/Np))
    position_error = np.zeros( (position_reference.shape[0], 2) )
    for i in range( position_reference.shape[0] ):
        position_error[i,:] = np.abs( data['lin_pos_ee_mea'][i,:2] - position_reference[i,:2])
    # Average absolute error 
    position_error_AVG[:,n_exp] = np.sum(position_error, axis=0) / Ns
    position_error_AVG_NORM[n_exp] = np.linalg.norm(position_error_AVG[:,n_exp])
    print("avg of xy position error = ", position_error_AVG[:,n_exp])
    print("norm of xy position error = ", position_error_AVG_NORM[n_exp])
    # Force tracking
    Np = data['N_plan'] ; Ns = data['N_simu']
    force_reference = data['frameForceRef'][2] 
    force_error = np.zeros(data['f_ee_mea'].shape[0])
    for i in range( Ns ):
        # force_error[i] = np.abs( data['f_ee_mea'][i,2] - force_reference)
        force_error[i] = np.abs( data['f_ee_mea'][i] - force_reference)
    # Maximum (peak) absolute error along x,y,z
    force_error_MAX[n_exp]   = np.max(force_error[N_START:])
    print("max error in z force = ", force_error_MAX[n_exp] )
    # Average absolute error 
    force_error_AVG[n_exp] = np.sum(force_error[N_START:], axis=0) / Ns

    #  Plot position reference and errors
    fig, ax = plt.subplots(2, 1, figsize=(19.2,10.8)) 
    tspan = np.linspace(0, data['T_tot'], position_reference.shape[0])
    # ax[0].plot(tspan, position_reference[:,0], label='ref_x')
    ax[0].plot(tspan, position_error[:,1], label='error_x')
    # ax[1].plot(tspan, position_reference[:,1], label='ref_y')
    ax[1].plot(tspan, position_error[:,1], label='error_y')
    plt.show()
    
    # # Smooth if necessary
    # if(FILTER > 0):
    #     data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'].copy(), FILTER)
    #     data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'].copy(), FILTER)
    #     data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
    #     data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'].copy(), FILTER)
    #     data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'].copy(), FILTER)
    #     data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'].copy(), FILTER)
    #     data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
    
    
# Plot errors

fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Avg position err x,y,z
fig2, ax2 = plt.subplots(2, 1, figsize=(19.2,10.8)) # Avg force err z + max force z
xyz = ['x','y','z']
color_LPF = 'r'
color = 'b'

ax1.plot(TILT_ANGLES_DEG, position_error_AVG_NORM, color=color, linestyle='-', linewidth=3)
# ax1.plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg_LPF_NORM, color=color_LPF, linestyle='-', linewidth=3)

# ax1[2].plot(TILT_ANGLES_DEG, position_error_AVG[2, :], color=color, linestyle='-', linewidth=3)
# ax1[2].plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg_LPF[2, :], color=color_LPF, linestyle='-', linewidth=3)

ax2[0].plot(TILT_ANGLES_DEG, force_error_AVG, color=color, linestyle='-', linewidth=3, label='LPF MPC')
# ax2[0].plot(TILT_ANGLES_DEG, f_ee_err_avg_z_LPF, color=color_LPF, linestyle='-', linewidth=3, label='Force feedback MPC')

ax2[1].plot(TILT_ANGLES_DEG, force_error_MAX, color=color, linestyle='-', linewidth=3)
# ax2[1].plot(TILT_ANGLES_DEG, f_ee_err_max_z_LPF, color=color_LPF, linestyle='-', linewidth=3)


# For each experiment plot perf as marker
for n_exp in range(N_EXP): 

    # NORM
    ax1.plot(float(TILT_ANGLES_DEG[n_exp]), position_error_AVG_NORM[n_exp], 
                                                marker='o', markerfacecolor=color, 
                                                markersize=16, markeredgecolor='k')
    # ax1.plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xy_avg_LPF_NORM[n_exp], 
    #                                             marker='s', markerfacecolor=color_LPF, 
    #                                             markersize=16, markeredgecolor='k')
    
    ax1.set_ylabel('$|| \Delta P^{EE}_{xy} ||$  (m)', fontsize=26)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    ax1.grid(True) 
    ax1.tick_params(axis = 'y', labelsize=22)
    ax1.set_xlabel('Angle (deg)', fontsize=26)
    ax1.tick_params(axis = 'x', labelsize = 22)


    # Avg force err z
    ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), force_error_AVG[n_exp], 
                                                    marker='o', markerfacecolor=color,
                                                    markersize=16, markeredgecolor='k')
    # ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_avg_z_LPF[n_exp],
    #                                                 marker='s', markerfacecolor=color_LPF,
    #                                                 markersize=16, markeredgecolor='k')
    ax2[0].set_ylabel('$\Delta \lambda_{z}$  (m)', fontsize=26)
    ax2[0].yaxis.set_major_locator(plt.MaxNLocator(2))
    ax2[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    ax2[0].grid(True) 
    ax2[0].tick_params(axis = 'y', labelsize=22)
    ax2[0].xaxis.set_ticklabels([])
    ax2[0].tick_params(bottom=False)


    # Max force z
    ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), force_error_MAX[n_exp], 
                                                    marker='o', markerfacecolor=color,
                                                    markersize=16, markeredgecolor='k')
    # ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_max_z_LPF[n_exp], 
    #                                                 marker='s', markerfacecolor=color_LPF,
    #                                                 markersize=16, markeredgecolor='k')
    ax2[1].set_ylabel('$\lambda^{max}_{z}$  (m)', fontsize=26)
    ax2[1].yaxis.set_major_locator(plt.MaxNLocator(2))
    ax2[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    ax2[1].grid(True) 
    ax2[1].tick_params(axis = 'y', labelsize=22)
    ax2[1].set_xlabel('Angle (deg)', fontsize=26)
    ax2[1].tick_params(axis = 'x', labelsize = 22)

# Legend error
handles1, labels1 = ax1.get_legend_handles_labels()
fig1.legend(handles1, labels1, loc='upper right', prop={'size': 26})
# Legend error norm 
handles2, labels2 = ax2[0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='upper right', prop={'size': 26})
# titles
# fig1.suptitle('End-effector position')
# fig2.suptitle('Normal force')
# Save, show , clean
fig1.savefig('/home/skleff/force-feedback/data/soft_contact_article/pos_err.png')
fig2.savefig('/home/skleff/force-feedback/data/soft_contact_article/force_err.png')
plt.show()
plt.close('all')

