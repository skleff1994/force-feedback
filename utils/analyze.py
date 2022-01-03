import sys
from utils import data_utils, plot_utils, analysis_utils

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(npz_path=None, FILTER=1):
  
  # load plot data
  if npz_path is None:
    logger.error("Please specify a DATASET to analyze !")
  else:
    # Extract data
    LPF = 'LPF' in npz_path
    logger.info(" Extracting data (LPF = "+str(LPF)+")")
    data = data_utils.extract_plot_data_from_npz(npz_path, LPF=LPF)    
    
    # Smooth if necessary
    if(FILTER > 0):
        data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'], FILTER)
        data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'], FILTER)
        if(LPF):
            tau_mea = analysis_utils.moving_average_filter(data['tau_mea'], FILTER)
        data['tau_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'], FILTER)
        data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'], FILTER)
        data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'], FILTER)
        data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'], FILTER)
        data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'], FILTER) 
    
    # Plot
    WHICH_PLOTS = ['f', 'ee_lin']
    if(LPF):
        plot_utils.plot_mpc_results_LPF(data, which_plots=WHICH_PLOTS, PLOT_PREDICTIONS=False)
    else:
        plot_utils.plot_mpc_results(data, which_plots=WHICH_PLOTS, PLOT_PREDICTIONS=False)



if __name__=='__main__':
    if len(sys.argv) <= 2:
        print("Usage: python plot_end_effector_errors < arg1: DATASET_NAME, arg2: N_EXP >")
        sys.exit(0)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))


