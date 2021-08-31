import time
import numpy as np
import os 

# Save data (dict) into compressed npz
def save_data(sim_data, save_name=None, save_dir=None):
    '''
    Saves data to a compressed npz file (binary)
    '''
    print('Compressing & saving data...')
    if(save_name is None):
        save_name = 'sim_data_NO_NAME'+str(time.time())
    if(save_dir is None):
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
    save_path = save_dir+'/'+save_name+'.npz'
    np.savez_compressed(save_path, data=sim_data)
    print("Saved data to "+str(save_path)+" !")

# Loads dict from compressed npz
def load_data(npz_file):
    '''
    Loads a npz archive of sim_data into a dict
    '''
    print('Loading data...')
    d = np.load(npz_file, allow_pickle=True)
    return d['data'][()]


def init_sim_data(config, robot, y0):
    '''
    Initialize simulation data from config file
    '''
    sim_data = {}
    # MPC & simulation parameters
    sim_data['T_tot'] = config['T_tot']                             # Total duration of simulation (s)
    sim_data['simu_freq'] = config['simu_freq']
    sim_data['ctrl_freq'] = config['ctrl_freq']
    sim_data['plan_freq'] = config['plan_freq']  
    sim_data['N_plan'] = int(sim_data['T_tot']*sim_data['plan_freq']) # Total number of planning steps in the simulation
    sim_data['N_ctrl'] = int(sim_data['T_tot']*sim_data['ctrl_freq']) # Total number of control steps in the simulation 
    sim_data['N_simu'] = int(sim_data['T_tot']*sim_data['simu_freq']) # Total number of simulation steps 
    sim_data['T_h'] = config['N_h']*config['dt']                    # Duration of the MPC horizon (s)
    sim_data['N_h'] = config['N_h']                                 # Number of nodes in MPC horizon
    sim_data['dt_ctrl'] = float(1./sim_data['ctrl_freq'])             # Duration of 1 control cycle (s)
    sim_data['dt_plan'] = float(1./sim_data['plan_freq'])             # Duration of 1 planning cycle (s)
    sim_data['dt_simu'] = float(1./sim_data['simu_freq'])             # Duration of 1 simulation cycle (s)
    # Misc params
    sim_data['nq'] = robot.model.nq
    sim_data['nv'] = robot.model.nv
    sim_data['nu'] = robot.model.nq
    sim_data['ny'] = sim_data['nq'] + sim_data['nv'] + sim_data['nu']
    sim_data['id_endeff'] = robot.model.getFrameId('contact')
    sim_data['p_ref'] = robot.data.oMf[sim_data['id_endeff']].translation
    # Predictions
    sim_data['Y_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, sim_data['ny'])) # Predicted states  ( ddp.xs : {y* = (q*, v*, tau*)} )
    sim_data['W_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nu']))   # Predicted torques ( ddp.us : {w*} )
    sim_data['Y_ref_CTRL'] = np.zeros((sim_data['N_ctrl']+1, sim_data['ny']))              # Reference state at motor drivers freq ( y* interpolated at CTRL freq )
    sim_data['W_ref_CTRL'] = np.zeros((sim_data['N_ctrl'], sim_data['nu']))              # Reference input at motor drivers freq ( w* interpolated at CTRL freq )
    sim_data['Y_ref_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))              # Reference state at actuation freq ( y* interpolated at SIMU freq )
    sim_data['W_ref_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['nu']))              # Reference input at actuation freq ( w* interpolated at SIMU freq )
    sim_data['Tau_des'] = np.zeros((sim_data['N_simu']+1, sim_data['nu']))                 # Desired control at actuation freq sent to actuator ( tau0* + EPS(tau1* - tau0*) )
    # Measurements
    sim_data['Y_mea_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))            # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq )
    sim_data['Y_mea_no_noise_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))   # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq ) without noise
    # # Derivatives  
    sim_data['dY_ref_CTRL'] = np.zeros((sim_data['N_ctrl'], sim_data['ny']))             # Estimated (FD) derivative of ref. state at CTRL frequency
    sim_data['dY_ref_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['ny']))             # Estimated (FD) derivative of ref. state at SIMU frequency
    sim_data['dY_mea_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['ny']))             # Estimated (FD) derivative of mea. state at SIMU frequency
    # Initialize measured state 
    sim_data['Y_ref_CTRL'][0, :] = y0
    sim_data['Y_ref_SIMU'][0, :] = y0
    sim_data['Y_mea_SIMU'][0, :] = y0
    sim_data['Y_mea_no_noise_SIMU'][0, :] = y0
    sim_data['Tau_des'][0, :] = y0[-sim_data['nu']:]
    sim_data['dY_ref_CTRL'][0,:] = np.zeros(sim_data['ny'])
    sim_data['dY_ref_SIMU'][0,:] = np.zeros(sim_data['ny'])
    sim_data['dY_mea_SIMU'][0,:] = np.zeros(sim_data['ny'])
    # Low-level simulation parameters (actuation model)
    # Scaling of desired torque
    alpha = np.random.uniform(low=config['alpha_min'], high=config['alpha_max'], size=(sim_data['nq'],))
    beta = np.random.uniform(low=config['beta_min'], high=config['beta_max'], size=(sim_data['nq'],))
    sim_data['alpha'] = alpha
    sim_data['beta'] = beta
    # White noise on desired torque and measured state
    sim_data['var_q'] = np.asarray(config['var_q'])
    sim_data['var_v'] = np.asarray(config['var_v'])
    sim_data['var_u'] = 0.001*(2*np.asarray(config['u_lim'])) #u_np.asarray(config['var_u']) 0.5% of range on the joint
    # White noise on desired torque and measured state
    sim_data['gain_P'] = config['Kp']*np.eye(sim_data['nq'])
    sim_data['gain_I'] = config['Ki']*np.eye(sim_data['nq'])
    sim_data['gain_D'] = config['Kd']*np.eye(sim_data['nq'])
    # Delays
    sim_data['delay_OCP_cycle'] = int(config['delay_OCP_ms'] * 1e-3 * sim_data['plan_freq']) # in planning cycles
    sim_data['delay_sim_cycle'] = int(config['delay_sim_cycle'])                           # in simu cycles
    
    return sim_data

# Extract MPC simu-specific plotting data from sim data
def extract_plot_data_from_sim_data(sim_data):
    '''
    Extract plot data from simu data
    '''
    print('Extracting plotting data from simulation data...')
    plot_data = {}
    nq = sim_data['nq']
    nv = sim_data['nv']
    if('nx' in sim_data.keys()):
        nx = sim_data['nx']
        plot_data['nx'] = nx
    if('ny' in sim_data.keys()):
        ny = sim_data['ny']  
        plot_data['ny'] = ny     
    plot_data['nq'] = nq
    plot_data['nv'] = nv
    nu = nq
    # State data 
    if('X_pred' in sim_data.keys()):
        # MPC predictions
        plot_data['q_pred'] = sim_data['X_pred'][:,:,:nq]
        plot_data['v_pred'] = sim_data['X_pred'][:,:,nq:nq+nv]
        # Measurements
        plot_data['q_mea'] = sim_data['X_mea'][:,:nq]
        plot_data['v_mea'] = sim_data['X_mea'][:,nq:nq+nv]
        plot_data['q_mea_no_noise'] = sim_data['X_mea_no_noise'][:,:nq]
        plot_data['v_mea_no_noise'] = sim_data['X_mea_no_noise'][:,nq:nq+nv]
        # Desired states (1st prediction of MPC horizon)
        plot_data['q_des'] = np.vstack([plot_data['q_mea'][0], sim_data['X_pred'][:,1,:nq]])
        plot_data['v_des'] = np.vstack([plot_data['v_mea'][0], sim_data['X_pred'][:,1,nq:nq+nv]])
    if('Y_pred' in sim_data.keys()):
        # Predictions at PLAN freq
        plot_data['q_pred'] = sim_data['Y_pred'][:,:,:nq]
        plot_data['v_pred'] = sim_data['Y_pred'][:,:,nq:nq+nv]
        plot_data['tau_pred'] = sim_data['Y_pred'][:,:,-nu:]
    
        # Measurements at SIMU freq
        plot_data['q_mea'] = sim_data['Y_mea_SIMU'][:,:nq]
        plot_data['v_mea'] = sim_data['Y_mea_SIMU'][:,nq:nq+nv]
        plot_data['tau_mea'] = sim_data['Y_mea_SIMU'][:,-nu:]
    
        plot_data['q_mea_no_noise'] = sim_data['Y_mea_no_noise_SIMU'][:,:nq]
        plot_data['v_mea_no_noise'] = sim_data['Y_mea_no_noise_SIMU'][:,nq:nq+nv]
        plot_data['tau_mea_no_noise'] = sim_data['Y_mea_no_noise_SIMU'][:,-nu:]
    
        # References at PLAN freq
        plot_data['q_des_PLAN'] = np.vstack([plot_data['q_mea'][0], plot_data['q_pred'][:,1]])
        plot_data['q_des_CTRL'] = sim_data['Y_ref_CTRL'][:,:nq] 
        plot_data['q_des_SIMU'] = sim_data['Y_ref_SIMU'][:,:nq]

        plot_data['v_des_PLAN'] = np.vstack([plot_data['v_mea'][0], plot_data['v_pred'][:,1]])
        plot_data['v_des_CTRL'] = sim_data['Y_ref_CTRL'][:,nq:nq+nv] 
        plot_data['v_des_SIMU'] = sim_data['Y_ref_SIMU'][:,nq:nq+nv]

        plot_data['tau_des_PLAN'] = np.vstack([plot_data['tau_mea'][0], plot_data['tau_pred'][:,1]])
        plot_data['tau_des_CTRL'] = sim_data['Y_ref_CTRL'][:,-nu:]
        plot_data['tau_des_SIMU'] = sim_data['Y_ref_SIMU'][:,-nu:]

        plot_data['tau_des'] = sim_data['Tau_des']

    # end-eff position
    plot_data['p_mea'] = sim_data['P_mea_SIMU']
    plot_data['p_mea_no_noise'] = sim_data['P_mea_no_noise_SIMU']
    plot_data['p_pred'] = sim_data['P_pred']
    plot_data['p_des'] = sim_data['P_des_PLAN'] 
    # control
    if('U_pred' in sim_data.keys()):
        plot_data['u_pred'] = sim_data['U_pred']
        plot_data['u_des'] = sim_data['U_pred'][:,0,:]
        plot_data['u_mea'] = sim_data['U_mea']
    if('W_pred' in sim_data.keys()):
        plot_data['w_pred'] = sim_data['W_pred']
        plot_data['w_des_PLAN'] = sim_data['W_pred'][:,0,:]
        plot_data['w_des_CTRL'] = sim_data['W_ref_CTRL']
        plot_data['w_des_SIMU'] = sim_data['W_ref_SIMU']
    # acc error
    if('A_err' in sim_data.keys()):
        plot_data['a_err'] = sim_data['A_err']
    # Misc. params
    plot_data['T_tot'] = sim_data['T_tot']
    plot_data['N_simu'] = sim_data['N_simu']
    plot_data['N_ctrl'] = sim_data['N_ctrl']
    plot_data['N_plan'] = sim_data['N_plan']
    plot_data['dt_plan'] = sim_data['dt_plan']
    plot_data['dt_ctrl'] = sim_data['dt_ctrl']
    plot_data['dt_simu'] = sim_data['dt_simu']
    plot_data['T_h'] = sim_data['T_h']
    plot_data['N_h'] = sim_data['N_h']
    plot_data['p_ref'] = sim_data['p_ref']
    plot_data['alpha'] = sim_data['alpha']
    plot_data['beta'] = sim_data['beta']
    # Get SVD & diagonal of Ricatti + record in sim data
    if('K' in sim_data.keys()):
      plot_data['K_svd'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nq))
      plot_data['Kp_diag'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nq))
      plot_data['Kv_diag'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nv))
      for i in range(sim_data['N_plan']):
        for j in range(sim_data['N_h']):
          plot_data['Kp_diag'][i, j, :] = sim_data['K'][i, j, :, :nq].diagonal()
          plot_data['Kv_diag'][i, j, :] = sim_data['K'][i, j, :, nq:nx].diagonal()
          _, sv, _ = np.linalg.svd(sim_data['K'][i, j, :, :])
          plot_data['K_svd'][i, j, :] = np.sort(sv)[::-1]
    # Get diagonal and eigenvals of Vxx + record in sim data
    if('Vxx' in sim_data.keys()):
      plot_data['Vxx_diag'] = np.zeros((sim_data['N_plan'],sim_data['N_h']+1, nx))
      plot_data['Vxx_eig'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, nx))
      for i in range(sim_data['N_plan']):
        for j in range(sim_data['N_h']+1):
          plot_data['Vxx_diag'][i, j, :] = sim_data['Vxx'][i, j, :, :].diagonal()
          plot_data['Vxx_eig'][i, j, :] = np.sort(np.linalg.eigvals(sim_data['Vxx'][i, j, :, :]))[::-1]
    # Get diagonal and eigenvals of Quu + record in sim data
    if('Quu' in sim_data.keys()):
      plot_data['Quu_diag'] = np.zeros((sim_data['N_plan'],sim_data['N_h'], nu))
      plot_data['Quu_eig'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nu))
      for i in range(sim_data['N_plan']):
        for j in range(sim_data['N_h']):
          plot_data['Quu_diag'][i, j, :] = sim_data['Quu'][i, j, :, :].diagonal()
          plot_data['Quu_eig'][i, j, :] = np.sort(np.linalg.eigvals(sim_data['Quu'][i, j, :, :]))[::-1]
    if('J_rank' in sim_data.keys()):
        plot_data['J_rank'] = sim_data['J_rank']
    if('xreg' in sim_data.keys()):
        plot_data['xreg'] = sim_data['xreg']
    if('ureg' in sim_data.keys()):
        plot_data['ureg'] = sim_data['ureg']
    # Cost weighs 
    if('ee_weight' in sim_data.keys() and
       'x_reg_weight' in sim_data.keys() and
       'u_reg_weight' in sim_data.keys()):
       plot_data['ee_weight'] = sim_data['ee_weight']
       plot_data['x_reg_weight'] = sim_data['x_reg_weight']
       plot_data['u_reg_weight'] = sim_data['u_reg_weight']

    return plot_data

# Same giving npz path OR dict as argument
def extract_plot_data(input_data):
    '''
    Extract plot data from npz archive or sim_data
    '''
    if(type(input_data)==str):
        sim_data = load_data(input_data)
    elif(type(input_data)==dict):
        sim_data = input_data
    else:
        TypeError("Input data must be a Python dict or a path to .npz archive")
    return extract_plot_data_from_sim_data(sim_data)