import sys
from utils import data_utils, analysis_utils
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [-20, -10, -5, 5, 10, 20] 
TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([0., angle*np.pi/180, 0.])
N_EXP = len(TILT_RPY)

PREFIX = '/home/skleff/force-feedback/data/'
prefix_lpf = PREFIX+'iiwa_LPF_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.25_Fc=0.5_Fs1.0'
# prefix =  PREFIX+'iiwa_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=0.5_Fs2.0'
prefix =  PREFIX+'iiwa_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=0.5_Fs1.0'

def main(FILTER=1, PLOT=False):
  
    lin_err_ee_xyz_avg = np.zeros((3,N_EXP))
    f_ee_err_avg_z = np.zeros(N_EXP)
    f_ee_err_max_z = np.zeros(N_EXP)

    lin_err_ee_xyz_avg_LPF = np.zeros((3,N_EXP))
    f_ee_err_avg_z_LPF = np.zeros(N_EXP)
    f_ee_err_max_z_LPF = np.zeros(N_EXP)

    for n_exp in range(N_EXP):


        # Extract data
        print("Extracting data...")
        data = data_utils.extract_plot_data_from_npz(prefix+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'.npz', LPF=False)    
        # Compute absolute tracking errors |mea - ref|
        # EE tracking
        Np = data['N_plan'] ; Ns = data['N_simu']
        lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
        lin_err_ee_xyz = np.zeros(data['lin_pos_ee_mea'].shape)
        for i in range( lin_pos_ee_ref.shape[0] ):
            lin_err_ee_xyz[i,:] = np.abs( data['lin_pos_ee_mea'][i,:] - lin_pos_ee_ref[i,:])
        # Cumulative absolute error
        lin_err_ee_xyz_sum = np.sum(lin_err_ee_xyz, axis=0)
        # Average absolute error 
        lin_err_ee_xyz_avg[:,n_exp] = lin_err_ee_xyz_sum / Ns
        # Force tracking
        Np = data['N_plan'] ; Ns = data['N_simu']
        f_ee_ref_z = -20
        f_ee_err_z = np.zeros(data['f_ee_mea'].shape[0])
        for i in range( Ns ):
            f_ee_err_z[i] = np.abs( data['f_ee_mea'][i,2] - f_ee_ref_z)
        # Maximum (peak) absolute error along x,y,z
        f_ee_err_max_z[n_exp]   = np.max(f_ee_err_z)
        # Cumulative absolute error
        f_ee_err_sum_z = np.sum(f_ee_err_z, axis=0)
        # Average absolute error 
        f_ee_err_avg_z[n_exp] = f_ee_err_sum_z / Ns
        # Smooth if necessary
        if(FILTER > 0):
            data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'].copy(), FILTER)
            data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'].copy(), FILTER)
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'].copy(), FILTER)
            data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'].copy(), FILTER)
            data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
        
        
        
        
        # Extract data and compute errors
        print("Extracting data LPF...")
        data = data_utils.extract_plot_data_from_npz(prefix_lpf+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'.npz', LPF=True)    
        Np = data['N_plan'] ; Ns = data['N_simu']
        lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
        lin_err_ee_xyz = np.zeros(data['lin_pos_ee_mea'].shape)
        for i in range( lin_pos_ee_ref.shape[0] ):
            lin_err_ee_xyz[i,:] = np.abs( data['lin_pos_ee_mea'][i,:] - lin_pos_ee_ref[i,:])
        # Cumulative absolute error
        lin_err_ee_xyz_sum = np.sum(lin_err_ee_xyz, axis=0)
        # Average absolute error 
        lin_err_ee_xyz_avg_LPF[:,n_exp] = lin_err_ee_xyz_sum / Ns
        # Force tracking
        Np = data['N_plan'] ; Ns = data['N_simu']
        f_ee_ref_z = -20
        f_ee_err_z = np.zeros(data['f_ee_mea'].shape[0])
        for i in range( Ns ):
            f_ee_err_z[i] = np.abs( data['f_ee_mea'][i,2] - f_ee_ref_z)
        # Maximum (peak) absolute error along x,y,z
        f_ee_err_max_z[n_exp]   = np.max(f_ee_err_z)
        # Cumulative absolute error
        f_ee_err_sum_z = np.sum(f_ee_err_z, axis=0)
        # Average absolute error 
        f_ee_err_avg_z_LPF[n_exp] = f_ee_err_sum_z / Ns
        # Smooth if necessary
        if(FILTER > 0):
            data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'].copy(), FILTER)
            data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'].copy(), FILTER)
            data['tau_mea'] = analysis_utils.moving_average_filter(data['tau_mea'].copy(), FILTER)
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'].copy(), FILTER)
            data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'].copy(), FILTER)
            data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
        print("----------------------------------")    





    # Plot
    fig1, ax1 = plt.subplots(3, 1, figsize=(19.2,10.8)) # Avg position err x,y,z
    fig2, ax2 = plt.subplots(2, 1, figsize=(19.2,10.8)) # Avg force err z + max force z
    xyz = ['x','y','z']
    color_LPF = 'r'
    color = 'b'

    ax1[0].plot(TILT_ANGLES_DEG, lin_err_ee_xyz_avg[0, :], color=color, linestyle='-', linewidth=3, label='Classical MPC')
    ax1[0].plot(TILT_ANGLES_DEG, lin_err_ee_xyz_avg_LPF[0, :], color=color_LPF, linestyle='-', linewidth=3, label='Force feedback MPC')
    
    ax1[1].plot(TILT_ANGLES_DEG, lin_err_ee_xyz_avg[1, :], color=color, linestyle='-', linewidth=3)
    ax1[1].plot(TILT_ANGLES_DEG, lin_err_ee_xyz_avg_LPF[1, :], color=color_LPF, linestyle='-', linewidth=3)
   
    ax1[2].plot(TILT_ANGLES_DEG, lin_err_ee_xyz_avg[2, :], color=color, linestyle='-', linewidth=3)
    ax1[2].plot(TILT_ANGLES_DEG, lin_err_ee_xyz_avg_LPF[2, :], color=color_LPF, linestyle='-', linewidth=3)
    
    ax2[0].plot(TILT_ANGLES_DEG, f_ee_err_avg_z, color=color, linestyle='-', linewidth=3, label='Classical MPC')
    ax2[0].plot(TILT_ANGLES_DEG, f_ee_err_avg_z_LPF, color=color_LPF, linestyle='-', linewidth=3, label='Force feedback MPC')
    
    ax2[1].plot(TILT_ANGLES_DEG, f_ee_err_max_z, color=color, linestyle='-', linewidth=3)
    ax2[1].plot(TILT_ANGLES_DEG, f_ee_err_max_z_LPF, color=color_LPF, linestyle='-', linewidth=3)


    # For each experiment plot perf as marker
    for n_exp in range(N_EXP): 
        # Avg position err x,y,z
        for i in range(3):
            ax1[i].plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xyz_avg[i, n_exp], 
                                                        marker='o', markerfacecolor=color, 
                                                        markersize=16, markeredgecolor='k')
            ax1[i].plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xyz_avg_LPF[i, n_exp], 
                                                        marker='s', markerfacecolor=color_LPF, 
                                                        markersize=16, markeredgecolor='k')
            ax1[i].grid(True) 
            ax1[i].set_ylabel('$\Delta P^{EE}_%s$  (m)'%xyz[i], fontsize=16)
            ax1[i].yaxis.set_major_locator(plt.MaxNLocator(2))
            ax1[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
            ax1[i].grid(True)
        ax1[-1].set_xlabel('Angle (deg)', fontsize=16)

        # Avg force err
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_avg_z[n_exp], 
                                                     marker='o', markerfacecolor=color,
                                                     markersize=16, markeredgecolor='k')
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_avg_z_LPF[n_exp],
                                                     marker='s', markerfacecolor=color_LPF,
                                                     markersize=16, markeredgecolor='k')
        ax2[0].set_ylabel('$\Delta \lambda_{z}$  (m)', fontsize=16)
        ax2[0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax2[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax2[0].grid(True)
        # Max force err
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_max_z[n_exp], 
                                                     marker='o', markerfacecolor=color,
                                                     markersize=16, markeredgecolor='k')
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_max_z_LPF[n_exp], 
                                                     marker='s', markerfacecolor=color_LPF,
                                                     markersize=16, markeredgecolor='k')
        ax2[1].set_ylabel('$\lambda^{max}_{z}$  (m)', fontsize=16)
        ax2[1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax2[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax2[1].grid(True)
        ax2[-1].set_xlabel('Angle (deg)', fontsize=16)
  
    # Legend error
    handles1, labels1 = ax1[0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc='upper right', prop={'size': 16})
    # Legend error norm 
    handles2, labels2 = ax2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper right', prop={'size': 16})
    # titles
    fig1.suptitle('Average position tracking error')
    fig2.suptitle('Average and maximum force error')
    # Save, show , clean
    fig1.savefig('/home/skleff/force-feedback/data//pos_err.png')
    fig2.savefig('/home/skleff/force-feedback/data//force_err.png')
    plt.show()
    plt.close('all')



if __name__=='__main__':
    if len(sys.argv) <= 1:
        print("Usage: python analyze_TILT.py [arg1: FILTER (int)]")
        sys.exit(0)
    sys.exit(main(int(sys.argv[1])))


