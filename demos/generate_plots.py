from utils import utils
import numpy as np 
import matplotlib.pyplot as plt


freqs = [250, 500, 1000, 2000, 5000, 10000, 20000]
N_EXP = 10 
data_path = '/home/skleff/force-feedback/data/DATASET3'
data ={}
pz_max = np.zeros((len(freqs), N_EXP))
pz_max_avg = np.zeros(len(freqs))
pz_residual = np.zeros((len(freqs), N_EXP))


# Extract plot data for each sim  
for k_freq, MPC_frequency in enumerate(freqs):
    
    for n_exp in range(N_EXP):
      
        # Extract plot data
        file_name = '/no_tracking__'+str(MPC_frequency)+'Hz__exp_'+str(n_exp)+'.yml'
        data_name = str(MPC_frequency)+'_'+str(n_exp)
        d = utils.extract_plot_data_from_yaml(data_path + '/' + str(MPC_frequency) + file_name)
        data[data_name] = d

        # Save plot figs
        save_dir = '/home/skleff/force-feedback/data/DATASET3/'+str(MPC_frequency)
        figs = utils.plot_results_from_plot_data(d, PLOT_PREDICTIONS=True, 
                                                    pred_plot_sampling=int(MPC_frequency/10),
                                                    SAVE=True,
                                                    SAVE_DIR=save_dir,
                                                    SAVE_NAME=str(MPC_frequency)+'Hz__exp_'+str(n_exp),
                                                    SHOW=False)


# Co
# Baseline with tracking
data_tracking = utils.extract_plot_data_from_yaml(data_path + '/DATASET2/with_tracking_1.0kHz.yml')
p_err_tracking = np.zeros(3)
p_err_norm_tracking = 0.
for i in range(3):
  p_err_tracking[i] = np.sum( np.abs( data_tracking['p_mea_no_noise'][:,i] - data_tracking['p_ref'][i] ) ) / data_tracking['N_simu']
# Compute average abs error norm 
p_err_nor_tracking = np.sum(p_err_tracking)

# Plots
fig1, ax1 = plt.subplots(3, 1)
fig2, ax2 = plt.subplots(1, 1)
# Plot errors 
  # x,y,z
for i in range(3):
    ax1[i].plot(np.array(data_names, dtype=float),  p_err[:,i], 'k-', alpha=0.5)
  # norm
ax2.plot(np.array(data_names, dtype=float),  p_err_norm, 'k-', alpha=0.5)
# Plot datapoints + legen
for k,data_name in enumerate(data_names):
    # Color for different frequencies
    coef = np.tanh(float(k) / float(len(data)) )
    col = [coef, coef/3., 1-coef, 1.]
    # Err in x
    ax1[0].plot(float(data_name), p_err[k,0], marker='o', color=col, label=data_name+' kHz')
    ax1[0].set(xlabel='Frequency (kHz)', ylabel='$|p_{x} - pref_{x}|$ (m)')
    ax1[0].grid() 
    # Err in y
    ax1[1].plot(float(data_name), p_err[k,1], marker='o', color=col)
    ax1[1].set(xlabel='Frequency (kHz)', ylabel='$|p_{y} - pref_{z}|$ (m)')
    ax1[1].grid() 
    # Err in z
    ax1[2].plot(float(data_name), p_err[k,2], marker='o', color=col)
    ax1[2].set(xlabel='Frequency (kHz)', ylabel='$|p_{y} - pref_{z}|$ (m)')
    ax1[2].grid() 
    # Err norm
    ax2.plot(float(data_name), p_err_norm[k], marker='o', color=col, label=data_name+' kHz')
    ax2.set(xlabel='Frequency (kHz)', ylabel='$||p - pref||$')
    ax2.grid() 
# BASELINE tracking
for i in range(3):
  ax1[i].plot(1., p_err_tracking[i], marker='o', markersize=12, color=[0., 1., 0., 1.], label='1.0 kHz (WITH TORQUE TRACKING)')
ax2.plot(1., p_err_norm_tracking, marker='o', markersize=12, color=[0., 1., 0., 1.], label='1.0 kHz (WITH TORQUE TRACKING)')

# Legend error
handles1, labels1 = ax1[0].get_legend_handles_labels()
fig1.legend(handles1, labels1, loc='upper right', prop={'size': 16})
# Legend error norm 
handles2, labels2 = ax2.get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='upper right', prop={'size': 16})
# titles
fig1.suptitle('Average absolute error (x,y,z) for EE task')
fig2.suptitle('Average absolute error norm for EE task')

plt.show()

# # Extract plotting data from yaml
# print("Extracting data (0.1 kHz)...")
# data_01 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_0.1kHz.yml')
# print("Extracting data (0.25 kHz)...")
# data_025 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_0.25kHz.yml')
# print("Extracting data (0.5 kHz)...")
# data_05 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_0.5kHz.yml')
# print("Extracting data (1 kHz)...")
# data_1 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_1kHz.yml')
# print("Extracting data (2 kHz)...")
# data_2 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_2kHz.yml')
# print("Extracting data (5 kHz)...")
# data_5 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_5kHz.yml')
# print("Extracting data (10 kHz)...")
# data_10 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_10kHz.yml')
# print("Extracting data (20 kHz)...")
# data_20 = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_20kHz.yml')

# if __name__=='__main__':
#     if len(sys.argv) <= 1:
#         print("Usage: python compare_data <0.1 0.25 0.5 1 2 5 10 20>")
#         sys.exit(0)
#     sys.exit(main(sys.argv[1]))
