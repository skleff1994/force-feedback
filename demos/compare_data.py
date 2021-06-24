from utils import utils
import numpy as np 
import matplotlib.pyplot as plt

# import sys

# def main(freqs=None):
#     if freqs is None:
#         print("No data to compare !")
#     else:
#         if freqs[0] in {}

data_names = ['0.1', '0.25'] #, '0.5', '1', '2', '5', '10', '20']
data = {}
p_err = np.zeros((len(data_names), 3))
p_err_sum = np.zeros(len(data_names))

for k,data_name in enumerate(data_names):
    # Extract data
    print("Extracting data ("+data_name+")...")
    d = utils.extract_plot_data_from_yaml('/home/skleff/force-feedback/data/no_tracking_filter_state_'+data_name+'kHz.yml')
    data[data_name] = d
    # Compute average absolute error on EE task 
    for i in range(3):
        p_err[k,i] = np.sum( np.abs( d['p_mea_no_noise'][:,i] - d['p_ref'][i] ) ) / d['N_simu']
    p_err_sum[k] = np.sum(p_err[k,:])

# Total error over x,y,z
print("err = ", p_err)
print("err_sum = ", p_err_sum)

# Plot errors
fig1, ax1 = plt.subplots(3, 1)
fig2, ax2 = plt.subplots(1, 1)
# Plot lines
for i in range(3):
    ax1[i].plot(np.array(data_names, dtype=float),  p_err[:,i], 'k-', alpha=0.5)
ax2.plot(np.array(data_names, dtype=float),  p_err_sum, 'k-', alpha=0.5)
# Plot datapoints + legen
for k,data_name in enumerate(data_names):
    # Err sum
    coef = np.tanh(float(k) / float(len(data)) )
    print(coef)
    col = [coef, coef/3., 1-coef, 1.]
    ax1[0].plot(float(data_name), p_err[k,0], marker='o', color=col, label=data_name+' kHz')
    ax1[0].set(xlabel='Frequency (kHz)', ylabel='$|p_{x} - pref_{x}|$ (m)')
    ax1[0].grid() 

    ax1[1].plot(float(data_name), p_err[k,1], marker='o', color=col)
    ax1[1].set(xlabel='Frequency (kHz)', ylabel='$|p_{y} - pref_{z}|$ (m)')
    ax1[1].grid() 

    ax1[2].plot(float(data_name), p_err[k,2], marker='o', color=col)
    ax1[2].set(xlabel='Frequency (kHz)', ylabel='$|p_{y} - pref_{z}|$ (m)')
    ax1[2].grid() 
   
    # Total err
    ax2.plot(float(data_name), p_err_sum[k], marker='o', color=col, label=data_name+' kHz')
    ax2.set(xlabel='Frequency (kHz)', ylabel='$||p - pref||$')
    ax2.grid() 

# Legend err
handles1, labels1 = ax1[0].get_legend_handles_labels()
fig1.legend(handles1, labels1, loc='upper right', prop={'size': 16})
# Legend err sum
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
