import time
import numpy as np
import os
from utils import data_utils, plot_utils

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


    
# Save data (dict) into compressed npz
def smooth_out(plot_data, LPF=False, filter_size=1):
    '''
    Saves data to a compressed npz file (binary)
    '''
    # Extract noised measurements
    q_mea   = plot_data['q_mea']
    v_mea   = plot_data['v_mea']
    if(LPF):
        tau_mea = plot_data['tau_mea']
    lin_pos = plot_data['lin_pos_ee_mea']
    ang_pos = plot_data['ang_pos_ee_mea']
    lin_vel = plot_data['lin_vel_ee_mea']
    ang_vel = plot_data['ang_vel_ee_mea']
    f_mea   = plot_data['f_ee_mea'] 
    # Filter them with moving average
    for i in range( plot_data['N_simu']+1 ):
        n_sum = min(i, filter_size)
        # Sum up over window
        for k in range(n_sum):
            q_mea[i,:] += q_mea[i-k-1, :]
            v_mea[i,:] += v_mea[i-k-1, :]
            if(i == plot_data['N_simu']):
                break
            if(LPF):
                tau_mea[i,:] += tau_mea[i-k-1, :]
            lin_pos[i,:] += lin_pos[i-k-1, :]
            ang_pos[i,:] += ang_pos[i-k-1, :]
            lin_vel[i,:] += lin_vel[i-k-1, :]
            ang_vel[i,:] += ang_vel[i-k-1, :]
            f_mea[i,:]   += f_mea[i-k-1, :]
        # Divide by number of samples
        q_mea   = q_mea / (n_sum + 1)
        v_mea   = v_mea / (n_sum + 1)
        if(LPF):
            tau_mea = tau_mea / (n_sum + 1)
        lin_pos = lin_pos / (n_sum + 1)
        ang_pos = ang_pos / (n_sum + 1)
        lin_vel = lin_vel / (n_sum + 1)
        ang_vel = ang_vel / (n_sum + 1)
    # Overwrite noised with smoothed
    plot_data['q_mea']          = q_mea  
    plot_data['v_mea']          = v_mea  
    if(LPF):
        plot_data['tau_mea']        = tau_mea
    plot_data['lin_pos_ee_mea'] = lin_pos
    plot_data['ang_pos_ee_mea'] = ang_pos
    plot_data['lin_vel_ee_mea'] = lin_vel
    plot_data['ang_vel_ee_mea'] = ang_vel
    plot_data['f_ee_mea']       = f_mea   
    
# import sys# def main(DATASET_NAME=None, N_=1):
#     pass

# if __name__=='__main__':
#     if len(sys.argv) <= 1:
#         print("Usage: python plot_end_effector_errors < arg1: DATASET_NAME, arg2: N_EXP >")
#         sys.exit(0)
#     sys.exit(main(sys.argv[1]))


