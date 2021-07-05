# from demos.run_mpc_experiments import N_EXP
from utils import utils
import numpy as np 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import fnmatch
import sys
import matplotlib.pyplot as plt


def main(DATASET_NAME=None, N_EXP=10):
  
  if DATASET_NAME is None:
    print("Please specify a DATASET to analyze !")
  
  else:
    # Process data files
    data_path = '/home/skleff/force-feedback/data/'+DATASET_NAME
      # List of strings freqs
    freqs = [freq for freq in listdir(data_path) if isdir(join(data_path, freq)) if any(listdir(join(data_path, freq)))] 
      # Sort and add BASELINE (1000Hz)
    if('BASELINE' in freqs):
      freqs.remove('BASELINE')
      freqs.sort(key=int)
      freqs.insert(0, 'BASELINE')
    # To store data for each MPC freq
    data = {}
    # Load data 
    for freq in freqs:
      data[freq] = {}
      if freq == 'BASELINE':
        TRACKING = True
        freq_value = 1000
      else:
        TRACKING = False
        freq_value = int(freq)
      for n_exp in range(N_EXP):
        # Extract plot data
        file_name = '/tracking='+str(TRACKING)+'_'+str(freq_value)+'Hz__exp_'+str(n_exp)+'.npz'
        data[freq][str(n_exp)]= utils.load_data(data_path + '/' + freq + file_name)

    # Process data for performance analysis along z axis
    pz_err_max = np.zeros((len(freqs), N_EXP))
    pz_err_max_avg = np.zeros(len(freqs))
    pz_err_res = np.zeros((len(freqs), N_EXP))
    pz_err_res_avg = np.zeros(len(freqs))
      # Compute errors (peak and residual)
    for k, MPC_frequency in enumerate(freqs):
      for n_exp in range(N_EXP):
        # Get data
        d = data[MPC_frequency][str(n_exp)]
        # Record error peak (max deviation from ref) along z axis
        pz_abs_err = np.abs(d['p_mea_no_noise'][:,2] - d['p_ref'][2])
        pz_err_max[k, n_exp] = np.max(pz_abs_err)
        pz_err_max_avg[k] += pz_err_max[k, n_exp]
        # Calculate steady-state error (avg error over last points) along z 
        length = int(d['N_simu']/2)
        pz_err_res[k, n_exp] = np.sum(pz_abs_err[-length:])/length
        pz_err_res_avg[k] += pz_err_res[k, n_exp]
      pz_err_max_avg[k] = pz_err_max_avg[k]/N_EXP
      pz_err_res_avg[k] = pz_err_res_avg[k]/N_EXP

    # Plots
    fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Max err in z (averaged over N_EXP) , vs MPC frequency
    fig2, ax2 = plt.subplots(1, 1, figsize=(19.2,10.8)) # plot avg SS ERROR in z vs frequencies DOTS connected 
    # For each experiment plot errors 
    for k in range(1, len(freqs)): 
      # PLOT DATA
      if(freqs[k] != 'BASELINE'):
        # Color for the current freq
        coef = np.tanh(float(k) / float(len(data)) )
        col = [coef, coef/3., 1-coef, 1.]
        # For each exp plot max err , steady-state err
        for n_exp in range(N_EXP):
          # max err
          ax1.plot(float(freqs[k]), pz_err_max[k, n_exp], marker='o', color=[coef, coef/3., 1-coef, .3]) 
          # SS err
          ax2.plot(float(freqs[k]), pz_err_res[k, n_exp], marker='o', color=[coef, coef/3., 1-coef, .3])
        # AVG max err
        ax1.plot(float(freqs[k]), pz_err_max_avg[k], marker='o', markersize=12, color=col, label=str(freqs[k])+' Hz')
        ax1.set(xlabel='Frequency (kHz)', ylabel='$AVG max|p_{z} - pref_{z}|$ (m)')
        # Err norm
        ax2.plot(float(freqs[k]), pz_err_res_avg[k], marker='o', markersize=12, color=col, label=str(freqs[k])+' Hz')
        ax2.set(xlabel='Frequency (kHz)', ylabel='$AVG Steady-State Error |p_{z} - pref_{z}|$')

  # PLOT BASELINE
  for n_exp in range(N_EXP):
    # max err
    ax1.plot(1000., pz_err_max[0, n_exp], marker='o', color=[0., 1., 0., .5],) 
    # SS err
    ax2.plot(1000, pz_err_res[0, n_exp], marker='o', color=[0., 1., 0., .5],) 
  # AVG max err
  ax1.plot(1000, pz_err_max_avg[0], marker='o', markersize=12, color=[0., 1., 0., 1.], label='BASELINE (1000) Hz')
  ax1.set(xlabel='Frequency (kHz)', ylabel='$AVG max|p_{z} - pref_{z}|$ (m)')
  # Err norm
  ax2.plot(1000, pz_err_res_avg[0], marker='o', markersize=12, color=[0., 1., 0., 1.], label='BASELINE (1000) Hz')
  ax2.set(xlabel='Frequency (kHz)', ylabel='$AVG Steady-State Error |p_{z} - pref_{z}|$')
  # Grids
  ax2.grid() 
  ax1.grid() 
  # Legend error
  handles1, labels1 = ax1.get_legend_handles_labels()
  fig1.legend(handles1, labels1, loc='upper right', prop={'size': 16})
  # Legend error norm 
  handles2, labels2 = ax2.get_legend_handles_labels()
  fig2.legend(handles2, labels2, loc='upper right', prop={'size': 16})
  # titles
  fig1.suptitle('Average peak error for EE task')
  fig2.suptitle('Average steady-state error for EE task')
  # Save, show , clean
  fig1.savefig('/home/skleff/force-feedback/data/'+DATASET_NAME+'/peak_err.png')
  fig2.savefig('/home/skleff/force-feedback/data/'+DATASET_NAME+'/resi_err.png')
  plt.show()
  plt.close('all')

if __name__=='__main__':
    if len(sys.argv) <= 1:
        print("Usage: python plot_end_effector_errors < arg1: DATASET_NAME, arg2: N_EXP >")
        sys.exit(0)
    sys.exit(main(sys.argv[1]))
