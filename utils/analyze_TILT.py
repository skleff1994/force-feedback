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

def main(npz_path=None, FILTER=1, PLOT=False):
  
  
  for n_exp in range(N_EXP):
    
    lin_err_ee_xyz_avg = np.zeros((3,N_EXP))
    f_ee_err_avg_z = np.zeros(N_EXP)
    f_ee_err_max_z = np.zeros(N_EXP)

    # load plot data
    if npz_path is None:
        logger.error("Please specify a DATASET to analyze !")
    else:
        # Extract data
        LPF = 'LPF' in npz_path
        print(" Extracting data (LPF = "+str(LPF)+")")
        data = data_utils.extract_plot_data_from_npz(npz_path+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp]), LPF=LPF)    

        # Compute absolute tracking errors |mea - ref|
        # EE tracking
        Np = data['N_plan'] ; Ns = data['N_simu']
        lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
        lin_err_ee_xyz = np.zeros(data['lin_pos_ee_mea'].shape)
        for i in range( lin_pos_ee_ref.shape[0] ):
            lin_err_ee_xyz[i,:] = np.abs( data['lin_pos_ee_mea'][i,:] - lin_pos_ee_ref[i,:])
        # Maximum (peak) absolute error along x,y,z
        # lin_err_ee_max_x   = np.max(lin_err_ee_xyz[:,0])
        # lin_err_ee_max_y   = np.max(lin_err_ee_xyz[:,1])
        # lin_err_ee_max_z   = np.max(lin_err_ee_xyz[:,2])
        # Cumulative absolute error
        lin_err_ee_xyz_sum = np.sum(lin_err_ee_xyz, axis=0)
        # Average absolute error 
        lin_err_ee_xyz_avg[:,n_exp] = lin_err_ee_xyz_sum / Ns
        # # Logs
        print("\n")
        print("EE tracking errors : \n")
        # print(" Peak abs. EE error along x   : "+str(lin_err_ee_max_x))
        # print(" Peak abs. EE error along y   : "+str(lin_err_ee_max_y))
        # print(" Peak abs. EE error along z   : "+str(lin_err_ee_max_z))
        # print(" Cumulative abs. EE xyz error : "+str(lin_err_ee_xyz_sum))
        print(" Average abs. EE xyz error    : "+str(lin_err_ee_xyz_avg))
        print("\n")
        print("----------------------------------")
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
        # Logs
        print("\n")
        print("FORCE tracking errors : \n")
        print(" Peak abs. FORCE error along z : "+str(f_ee_err_max_z))
        # print(" Cumulative abs. FORCE z error : "+str(f_ee_err_sum_z))
        print(" Average abs. FORCE z error    : "+str(f_ee_err_avg_z))
        print("\n")

        # Smooth if necessary
        if(FILTER > 0):
            data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'].copy(), FILTER)
            data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'].copy(), FILTER)
            if(LPF):
                data['tau_mea'] = analysis_utils.moving_average_filter(data['tau_mea'].copy(), FILTER)
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'].copy(), FILTER)
            data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'].copy(), FILTER)
            data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 


    # Plot
    fig1, ax1 = plt.subplots(3, 1, figsize=(19.2,10.8)) # Avg position err x,y,z
    fig2, ax2 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Avg force err z + max force z
    # For each experiment plot perf 
    for n_exp in range(N_EXP): 
        # Color for the current freq
        coef_col = float(n_exp+1) / float(len(data)) 
        col_exp_avg = [coef_col, coef_col/3., 1-coef_col, 1.]
        # Transparency gradient for expes
        coef_exp = float(n_exp+1) / (2*float(N_EXP))
        col_exp = [coef_col-coef_col*coef_exp, 0.25 - coef_exp/2, 1-coef_col, 2.*coef_exp]
        # Avg position err x,y,z
        for i in range(3):
            ax1.plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xyz_avg[i, n_exp], marker='o', markerfacecolor=col_exp, 
                                                        markersize=10, markeredgecolor='k')
        # Avg force err
        ax2.plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_avg_z[n_exp], marker='o', markerfacecolor=col_exp,
                                                        markersize=10, markeredgecolor='k')
        # Max force err
        ax1.plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_max_z[n_exp], marker='s', markerfacecolor=col_exp_avg,
                    markersize=14, markeredgecolor='k', label=str(TILT_ANGLES_DEG[n_exp]))
        ax1.set(xlabel='Angle (degrees)', ylabel='Avg. Err. $|p_{z} - pref_{z}|$ (m)')
        # # Err norm
        # ax2.plot(float(TILT_ANGLES_DEG[n_exp]), pz_err_res_avg[k], marker='s', markerfacecolor=col_exp_avg,
        #             markersize=14, markeredgecolor='k', label=str(TILT_ANGLES_DEG[n_exp])+' Hz')
        # ax2.set(xlabel='Frequency (Hz)', ylabel='Residual Error $|p_{z} - pref_{z}|$ (m)')

    # AVG max err
    # ax1.plot(1000, pz_err_max_avg[0], marker='s', markerfacecolor=[0., 1., 0., 1.], 
    #                                     markersize=14, markeredgecolor='k', label='BASELINE (1000) Hz')
    # ax1.set(xlabel='Frequency (Hz)', ylabel='$AVG max|p_{z} - pref_{z}|$ (m)')
    # # Err norm
    # ax2.plot(1000, pz_err_res_avg[0], marker='s', markerfacecolor=[0., 1., 0., 1.], 
    #                                     markersize=14, markeredgecolor='k', label='BASELINE (1000) Hz')
    # ax2.set(xlabel='Frequency (Hz)', ylabel='$AVG Steady-State Error |p_{z} - pref_{z}|$')
    # Grids
    ax2.grid() 
    ax1.grid() 
    # Legend error
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc='upper right', prop={'size': 16})
    # Legend error norm 
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper right', prop={'size': 16})
    # titles
    fig1.suptitle('Average position tracking error')
    fig2.suptitle('Average and maximum force error')
    # Save, show , clean
    # fig1.savefig('/home/skleff/impedance_mpc/data/'+DATASET_NAME+'/peak_err.png')
    # fig2.savefig('/home/skleff/impedance_mpc/data/'+DATASET_NAME+'/resi_err.png')
    plt.show()
    plt.close('all')



if __name__=='__main__':
    if len(sys.argv) <= 2:
        print("Usage: python analyze_TILT.py [arg1: npz_path (str)] [arg2: FILTER (int)]")
        sys.exit(0)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))


