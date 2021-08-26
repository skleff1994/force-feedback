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

# Extract MPC simu-specific plotting data from sim data
def extract_plot_data_from_sim_data(sim_data):
    '''
    Extract plot data from simu data
    '''
    print('Extracting plotting data from simulation data...')
    plot_data = {}
    nq = sim_data['nq']
    nv = sim_data['nv']
    nx = sim_data['nx']
    plot_data['nq'] = nq
    plot_data['nv'] = nv
    plot_data['nx'] = nx
    nu = nq
    # state predictions
    plot_data['q_pred'] = sim_data['X_pred'][:,:,:nq]
    plot_data['v_pred'] = sim_data['X_pred'][:,:,nq:nq+nv]
    plot_data['tau_pred'] = sim_data['X_pred'][:,:,nq+nv:]
    # measured state
    plot_data['q_mea'] = sim_data['X_mea'][:,:nq]
    plot_data['v_mea'] = sim_data['X_mea'][:,nq:nq+nv]
    plot_data['tau_mea'] = sim_data['X_mea'][:,nq+nv:]
    plot_data['q_mea_no_noise'] = sim_data['X_mea_no_noise'][:,:nq]
    plot_data['v_mea_no_noise'] = sim_data['X_mea_no_noise'][:,nq:nq+nv]
    plot_data['tau_mea_no_noise'] = sim_data['X_mea_no_noise'][:,nq+nv:]
    # desired state (append 1st state at start)
    plot_data['q_des'] = np.vstack([sim_data['X_mea'][0,:nq], sim_data['X_pred'][:,1,:nq]])
    plot_data['v_des'] = np.vstack([sim_data['X_mea'][0,nq:nq+nv], sim_data['X_pred'][:,1,nq:nq+nv]])
    plot_data['tau_des'] = np.vstack([sim_data['X_mea'][0,nq+nv:], sim_data['X_pred'][:,1,nq+nv:]])
    if('X_ref' in sim_data.keys()):
        plot_data['q_ref'] = sim_data['X_ref'][:,:nq]
    # end-eff position
    plot_data['p_mea'] = sim_data['P_mea']
    plot_data['p_mea_no_noise'] = sim_data['P_mea_no_noise']
    plot_data['p_pred'] = sim_data['P_pred']
    plot_data['p_des'] = sim_data['P_des'] 
    # control
    plot_data['u_pred'] = sim_data['U_pred']
    plot_data['u_des'] = sim_data['U_pred'][:,0,:]
    plot_data['u_mea'] = sim_data['U_mea']
    # acc error
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