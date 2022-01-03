import time
import numpy as np
import sys
from os import listdir
from os.path import isdir, join
from utils import data_utils, plot_utils

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(npz_path=None, filter=1):
  
  # load plot data
  if npz_path is None:
    logger.error("Please specify a DATASET to analyze !")
  else:
    LPF = '_LPF_' in npz_path
    logger.info(" LPF = "+str(LPF))
    data = data_utils.extract_plot_data_from_npz(npz_path, LPF=LPF)    

    # # plot_utils.plot_mpc_endeff_linear(data)
    # # Filter out noise 
    # q_mea   = data['q_mea'].copy() 
    # v_mea   = data['v_mea'].copy() 
    # if(LPF):
    #     tau_mea = data['tau_mea'].copy() 
    # lin_pos = data['lin_pos_ee_mea'].copy() 
    # ang_pos = data['ang_pos_ee_mea'].copy() 
    # lin_vel = data['lin_vel_ee_mea'].copy() 
    # ang_vel = data['ang_vel_ee_mea'].copy() 
    # f_mea   = data['f_ee_mea'].copy() 
    # # Filter them with moving average
    # for i in range( lin_pos.shape[0] ):
    #     n_sum = min(i, filter)
    #     # Sum up over window
    #     for k in range(n_sum):
    #         q_mea[i,:] += data['q_mea'][i-k-1, :]
    #         v_mea[i,:] += data['v_mea'][i-k-1, :]
    #         if(i == data['N_simu']):
    #             break
    #         if(LPF):
    #             tau_mea[i,:] += data['tau_mea'][i-k-1, :]
    #         lin_pos[i,:] += data['lin_pos_ee_mea'][i-k-1, :]
    #         ang_pos[i,:] += data['ang_pos_ee_mea'][i-k-1, :]
    #         lin_vel[i,:] += data['lin_vel_ee_mea'][i-k-1, :]
    #         ang_vel[i,:] += data['ang_vel_ee_mea'][i-k-1, :]
    #         f_mea[i,:]   += data['f_ee_mea'][i-k-1, :]

    #     #  Divide by number of samples
    #     q_mea[i,:] = q_mea[i,:] / (n_sum + 1)
    #     v_mea[i,:] = v_mea[i,:] / (n_sum + 1)
    #     if(LPF):
    #         tau_mea[i,:] = tau_mea[i,:] / (n_sum + 1)
    #     lin_pos[i,:] = lin_pos[i,:] / (n_sum + 1)
    #     ang_pos[i,:] = ang_pos[i,:] / (n_sum + 1)
    #     lin_vel[i,:] = lin_vel[i,:] / (n_sum + 1)
    #     ang_vel[i,:] = ang_vel[i,:] / (n_sum + 1)
    # # Overwrite noised with smoothed
    # data['q_mea']          = q_mea  
    # data['v_mea']          = v_mea  
    # if(LPF):
    #     data['tau_mea']    = tau_mea
    # data['lin_pos_ee_mea']      = lin_pos
    # data['ang_pos_ee_mea'] = ang_pos
    # data['lin_vel_ee_mea'] = lin_vel
    # data['ang_vel_ee_mea'] = ang_vel
    # data['f_ee_mea']       = f_mea   
    if(LPF):
        plot_utils.plot_mpc_results_LPF(data, which_plots='all', PLOT_PREDICTIONS=False)
    else:
        plot_utils.plot_mpc_results(data, which_plots='all', PLOT_PREDICTIONS=False)



if __name__=='__main__':
    if len(sys.argv) <= 2:
        print("Usage: python plot_end_effector_errors < arg1: DATASET_NAME, arg2: N_EXP >")
        sys.exit(0)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))


