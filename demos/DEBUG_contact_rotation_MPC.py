from utils import data_utils, path_utils, plot_utils

# Load MPC sim data and extract plot data
path_GOOD = '/home/skleff/force-feedback/data/iiwa_contact_rotation_MPC_bullet__BIAS=False_NOISE=False_DELAY=False_Fp=0.5_Fc=0.5_Fs1.0_GOOD.npz'
path_BAD  = '/home/skleff/force-feedback/data/iiwa_contact_rotation_MPC_bullet__BIAS=False_NOISE=False_DELAY=False_Fp=0.5_Fc=0.5_Fs1.0_BAD.npz'
sim_data_GOOD = data_utils.load_data(path_GOOD)
sim_data_BAD  = data_utils.load_data(path_BAD)
plot_data_GOOD = data_utils.extract_plot_data_from_sim_data(sim_data_GOOD) 
plot_data_BAD  = data_utils.extract_plot_data_from_sim_data(sim_data_BAD)

# Get time where things mess up?



