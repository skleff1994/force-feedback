import time

from matplotlib.pyplot import plot
import numpy as np
import os 
from utils import pin_utils



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



# Initialize simulation data for MPC simulation
def init_sim_data(config, robot, y0):
    '''
    Initialize simulation data from config file
    '''
    # TO BE IMPLEMENTED (like in /impedance_mpc repo)
    pass 

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
    plot_data['nx'] = nx   
    plot_data['nq'] = nq
    plot_data['nv'] = nv
    nu = nq
    # State data 
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

    # EE data
      # Measurements
    plot_data['p_mea'] = sim_data['P_mea_SIMU']
    plot_data['p_mea_no_noise'] = sim_data['P_mea_no_noise_SIMU']
      # MPC predictions
    plot_data['p_pred'] = sim_data['P_pred']
    plot_data['p_des'] = sim_data['P_des_PLAN'] 

    # Control data 
      # Predictions
    plot_data['u_pred'] = sim_data['U_pred']
      # Desired
    plot_data['u_des'] = sim_data['U_pred'][:,0,:]
      # Measured
    plot_data['u_mea'] = sim_data['U_mea']

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




# Initialize MPC simulation with torque feedback based on Low-Pass-Filter (LPF) Actuation Model
def init_sim_data_LPF(config, robot, y0):
    '''
    Initialize simulation data from config file (for torque feedback MPC based on LPF)
    '''
    sim_data = {}
    # MPC & simulation parameters
    sim_data['T_tot'] = config['T_tot']                               # Total duration of simulation (s)
    sim_data['simu_freq'] = config['simu_freq']                       # Simulation frequency
    sim_data['ctrl_freq'] = config['ctrl_freq']                       # Control frequency (reference sent to motors)
    sim_data['plan_freq'] = config['plan_freq']                       # Planning frequency (OCP solution update rate)
    sim_data['N_plan'] = int(sim_data['T_tot']*sim_data['plan_freq']) # Total number of planning steps in the simulation
    sim_data['N_ctrl'] = int(sim_data['T_tot']*sim_data['ctrl_freq']) # Total number of control steps in the simulation 
    sim_data['N_simu'] = int(sim_data['T_tot']*sim_data['simu_freq']) # Total number of simulation steps 
    sim_data['T_h'] = config['N_h']*config['dt']                      # Duration of the MPC horizon (s)
    sim_data['N_h'] = config['N_h']                                   # Number of nodes in MPC horizon
    sim_data['dt_ctrl'] = float(1./sim_data['ctrl_freq'])             # Duration of 1 control cycle (s)
    sim_data['dt_plan'] = float(1./sim_data['plan_freq'])             # Duration of 1 planning cycle (s)
    sim_data['dt_simu'] = float(1./sim_data['simu_freq'])             # Duration of 1 simulation cycle (s)
    # Misc params
    sim_data['pin_model'] = robot.model
    sim_data['nq'] = sim_data['pin_model'].nq
    sim_data['nv'] = sim_data['pin_model'].nv
    sim_data['nu'] = sim_data['pin_model'].nq
    sim_data['nx'] = sim_data['nq'] + sim_data['nv']
    sim_data['ny'] = sim_data['nx'] + sim_data['nu']
    sim_data['id_endeff'] = sim_data['pin_model'].getFrameId('contact')
    # Target translation
    if(config['p_des']=='None'):
      robot.framesForwardKinematics(y0[:sim_data['nq']])
      robot.computeJointJacobians(y0[:sim_data['nq']])
      sim_data['p_ee_ref'] = robot.data.oMf[sim_data['id_endeff']].translation.copy()
    else:
      sim_data['p_ee_ref'] = config['p_des']
    sim_data['v_ee_ref'] = config['v_des']
    # Predictions
    sim_data['Y_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, sim_data['ny'])) # Predicted states  ( ddp.xs : {y* = (q*, v*, tau*)} )
    sim_data['W_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nu']))   # Predicted torques ( ddp.us : {w*} )
    sim_data['Y_des_PLAN'] = np.zeros((sim_data['N_plan']+1, sim_data['ny']))            # Predicted states at planner frequency  ( y* interpolated at PLAN freq )
    sim_data['W_des_PLAN'] = np.zeros((sim_data['N_plan'], sim_data['nu']))              # Predicted torques at planner frequency ( w* interpolated at PLAN freq )
    sim_data['Y_des_CTRL'] = np.zeros((sim_data['N_ctrl']+1, sim_data['ny']))            # Reference state at motor drivers freq ( y* interpolated at CTRL freq )
    sim_data['W_des_CTRL'] = np.zeros((sim_data['N_ctrl'], sim_data['nu']))              # Reference input at motor drivers freq ( w* interpolated at CTRL freq )
    sim_data['Y_des_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))            # Reference state at actuation freq ( y* interpolated at SIMU freq )
    sim_data['W_des_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['nu']))              # Reference input at actuation freq ( w* interpolated at SIMU freq )
    # Measurements
    sim_data['Y_mea_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))            # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq )
    sim_data['Y_mea_no_noise_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))   # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq ) without noise
    sim_data['Y_mea_SIMU'][0, :] = y0
    sim_data['Y_mea_no_noise_SIMU'][0, :] = y0
    # # Derivatives  
    # sim_data['dY_pred_CTRL'] = np.zeros((sim_data['N_ctrl'], sim_data['ny']))          # Estimated (FD) derivative of ref. state at CTRL frequency
    # sim_data['dY_pred_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['ny']))          # Estimated (FD) derivative of ref. state at SIMU frequency
    # sim_data['dY_mea_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['ny']))           # Estimated (FD) derivative of mea. state at SIMU frequency
    # Initialize measured state 
    # sim_data['Y_pred_PLAN'][0, :] = y0
    # sim_data['Y_pred_CTRL'][0, :] = y0
    # sim_data['Y_pred_SIMU'][0, :] = y0
    # sim_data['Tau_des'][0, :] = y0[-sim_data['nu']:]
    # sim_data['dY_pred_CTRL'][0,:] = np.zeros(sim_data['ny'])
    # sim_data['dY_pred_SIMU'][0,:] = np.zeros(sim_data['ny'])
    # sim_data['dY_mea_SIMU'][0,:] = np.zeros(sim_data['ny'])
    # Low-level simulation parameters (actuation model)
    # Scaling of desired torque
    alpha = np.random.uniform(low=config['alpha_min'], high=config['alpha_max'], size=(sim_data['nq'],))
    beta = np.random.uniform(low=config['beta_min'], high=config['beta_max'], size=(sim_data['nq'],))
    sim_data['alpha'] = alpha
    sim_data['beta'] = beta
    # White noise on desired torque and measured state
    sim_data['var_q'] = np.asarray(config['var_q'])
    sim_data['var_v'] = np.asarray(config['var_v'])
    sim_data['var_u'] = 0.5*np.asarray(config['var_u']) #0.5% of range on the joint
    # White noise on desired torque and measured state
    sim_data['gain_P'] = config['Kp']*np.eye(sim_data['nq'])
    sim_data['gain_I'] = config['Ki']*np.eye(sim_data['nq'])
    sim_data['gain_D'] = config['Kd']*np.eye(sim_data['nq'])
    # Delays
    sim_data['delay_OCP_cycle'] = int(config['delay_OCP_ms'] * 1e-3 * sim_data['plan_freq']) # in planning cycles
    sim_data['delay_sim_cycle'] = int(config['delay_sim_cycle'])                             # in simu cycles
    # Other stuff
    sim_data['RECORD_SOLVER_DATA'] = config['RECORD_SOLVER_DATA']
    if(sim_data['RECORD_SOLVER_DATA']):
      sim_data['K'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nq'], sim_data['ny']))     # Ricatti gains (K_0)
      sim_data['Vxx'] = np.zeros((sim_data['N_plan'], config['N_h']+1, sim_data['ny'], sim_data['ny'])) # Hessian of the Value Function  
      sim_data['Quu'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nu'], sim_data['nu']))   # Hessian of the Value Function 
      sim_data['xreg'] = np.zeros(sim_data['N_plan'])                                                   # State reg in solver (diag of Vxx)
      sim_data['ureg'] = np.zeros(sim_data['N_plan'])                                                   # Control reg in solver (diag of Quu)
      sim_data['J_rank'] = np.zeros(sim_data['N_plan'])                                                 # Rank of Jacobian
    return sim_data

# Extract MPC simu-specific plotting data from sim data (LPF)
def extract_plot_data_from_sim_data_LPF(sim_data):
    '''
    Extract plot data from simu data (for torque feedback MPC based on LPF)
    '''
    print('Extracting plotting data from simulation data...')
    plot_data = {}
    # Robot model & params
    plot_data['pin_model'] = sim_data['pin_model']
    nq = plot_data['pin_model'].nq; plot_data['nq'] = nq
    nv = plot_data['pin_model'].nv; plot_data['nv'] = nv
    nx = nq+nv; plot_data['nx'] = nx
    ny = sim_data['ny']; plot_data['ny'] = ny
    nu = nq
    # MPC params
    plot_data['T_tot'] = sim_data['T_tot']
    plot_data['N_simu'] = sim_data['N_simu']; plot_data['N_ctrl'] = sim_data['N_ctrl']; plot_data['N_plan'] = sim_data['N_plan']
    plot_data['dt_plan'] = sim_data['dt_plan']; plot_data['dt_ctrl'] = sim_data['dt_ctrl']; plot_data['dt_simu'] = sim_data['dt_simu']
    plot_data['T_h'] = sim_data['T_h']; plot_data['N_h'] = sim_data['N_h']
    plot_data['alpha'] = sim_data['alpha']; plot_data['beta'] = sim_data['beta']
    plot_data['p_ee_ref'] = sim_data['p_ee_ref']
    plot_data['v_ee_ref'] = sim_data['v_ee_ref']
    # Control predictions
    plot_data['w_pred'] = sim_data['W_pred']
      # Extract 1st prediction
    plot_data['w_des_PLAN'] = sim_data['W_des_PLAN']
    plot_data['w_des_CTRL'] = sim_data['W_des_CTRL']
    plot_data['w_des_SIMU'] = sim_data['W_des_SIMU']
    # State predictions (at PLAN freq)
    plot_data['q_pred'] = sim_data['Y_pred'][:,:,:nq]
    plot_data['v_pred'] = sim_data['Y_pred'][:,:,nq:nq+nv]
    plot_data['tau_pred'] = sim_data['Y_pred'][:,:,-nu:]
      # Extract 1st prediction + shift 1 planning cycle
    plot_data['q_des_PLAN'] = sim_data['Y_des_PLAN'][:,:nq] #np.zeros((plot_data['N_plan']+1,nq))
    plot_data['v_des_PLAN'] = sim_data['Y_des_PLAN'][:,nq:nq+nv] #np.zeros((plot_data['N_plan']+1,nv))
    plot_data['tau_des_PLAN'] = sim_data['Y_des_PLAN'][:,-nu:] #np.zeros((plot_data['N_plan']+1,nq))
    plot_data['q_des_CTRL'] = sim_data['Y_des_CTRL'][:,:nq] #np.zeros((plot_data['N_plan']+1,nq))
    plot_data['v_des_CTRL'] = sim_data['Y_des_CTRL'][:,nq:nq+nv] #np.zeros((plot_data['N_plan']+1,nv))
    plot_data['tau_des_CTRL'] = sim_data['Y_des_CTRL'][:,-nu:] #np.zeros((plot_data['N_plan']+1,nq))
    plot_data['q_des_SIMU'] = sim_data['Y_des_SIMU'][:,:nq] #np.zeros((plot_data['N_plan']+1,nq))
    plot_data['v_des_SIMU'] = sim_data['Y_des_SIMU'][:,nq:nq+nv] #np.zeros((plot_data['N_plan']+1,nv))
    plot_data['tau_des_SIMU'] = sim_data['Y_des_SIMU'][:,-nu:] #np.zeros((plot_data['N_plan']+1,nq))
    # State measurements (at SIMU freq)
    plot_data['q_mea'] = sim_data['Y_mea_SIMU'][:,:nq]
    plot_data['v_mea'] = sim_data['Y_mea_SIMU'][:,nq:nq+nv]
    plot_data['tau_mea'] = sim_data['Y_mea_SIMU'][:,-nu:]
    plot_data['q_mea_no_noise'] = sim_data['Y_mea_no_noise_SIMU'][:,:nq]
    plot_data['v_mea_no_noise'] = sim_data['Y_mea_no_noise_SIMU'][:,nq:nq+nv]
    plot_data['tau_mea_no_noise'] = sim_data['Y_mea_no_noise_SIMU'][:,-nu:]
    # Extract gravity torques
    plot_data['grav'] = np.zeros((sim_data['N_simu']+1, plot_data['nq']))
    for i in range(plot_data['N_simu']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav_(plot_data['q_mea'][i,:], plot_data['pin_model'])
    # EE predictions (at PLAN freq)
    plot_data['p_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
    plot_data['v_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
    for node_id in range(sim_data['N_h']+1):
        plot_data['p_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['v_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
    # EE measurements (at SIMU freq)
    plot_data['p_ee_mea'] = pin_utils.get_p_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_mea'] = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['p_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    # EE des
    plot_data['p_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['p_ee_des_CTRL'] = pin_utils.get_p_(plot_data['q_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_des_CTRL'] = pin_utils.get_v_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['p_ee_des_SIMU'] = pin_utils.get_p_(plot_data['q_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_des_SIMU'] = pin_utils.get_v_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])

    # Solver data (optional)
    if(sim_data['RECORD_SOLVER_DATA']):
      # Get SVD & diagonal of Ricatti + record in sim data
      plot_data['K_svd'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nq))
      plot_data['Kp_diag'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nq))
      plot_data['Kv_diag'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nv))
      plot_data['Ktau_diag'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nu))
      for i in range(sim_data['N_plan']):
        for j in range(sim_data['N_h']):
          plot_data['Kp_diag'][i, j, :] = sim_data['K'][i, j, :, :nq].diagonal()
          plot_data['Kv_diag'][i, j, :] = sim_data['K'][i, j, :, nq:nq+nv].diagonal()
          plot_data['Ktau_diag'][i, j, :] = sim_data['K'][i, j, :, -nu:].diagonal()
          _, sv, _ = np.linalg.svd(sim_data['K'][i, j, :, :])
          plot_data['K_svd'][i, j, :] = np.sort(sv)[::-1]
      # Get diagonal and eigenvals of Vxx + record in sim data
      plot_data['Vxx_diag'] = np.zeros((sim_data['N_plan'],sim_data['N_h']+1, ny))
      plot_data['Vxx_eig'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, ny))
      for i in range(sim_data['N_plan']):
        for j in range(sim_data['N_h']+1):
          plot_data['Vxx_diag'][i, j, :] = sim_data['Vxx'][i, j, :, :].diagonal()
          plot_data['Vxx_eig'][i, j, :] = np.sort(np.linalg.eigvals(sim_data['Vxx'][i, j, :, :]))[::-1]
      # Get diagonal and eigenvals of Quu + record in sim data
      plot_data['Quu_diag'] = np.zeros((sim_data['N_plan'],sim_data['N_h'], nu))
      plot_data['Quu_eig'] = np.zeros((sim_data['N_plan'], sim_data['N_h'], nu))
      for i in range(sim_data['N_plan']):
        for j in range(sim_data['N_h']):
          plot_data['Quu_diag'][i, j, :] = sim_data['Quu'][i, j, :, :].diagonal()
          plot_data['Quu_eig'][i, j, :] = np.sort(np.linalg.eigvals(sim_data['Quu'][i, j, :, :]))[::-1]
      # Get Jacobian
      plot_data['J_rank'] = sim_data['J_rank']
      # Get solve regs
      plot_data['xreg'] = sim_data['xreg']
      plot_data['ureg'] = sim_data['ureg']

    return plot_data





# Extract DDP data (classic or LPF)
def extract_ddp_data(ddp):
    '''
    Record relevant data from ddp solver in order to plot 
    '''
    # Store data
    ddp_data = {}
    # OCP params
    ddp_data['T'] = ddp.problem.T
    ddp_data['dt'] = ddp.problem.runningModels[0].dt
    ddp_data['nq'] = ddp.problem.runningModels[0].state.nq
    ddp_data['nv'] = ddp.problem.runningModels[0].state.nv
    ddp_data['nu'] = ddp.problem.runningModels[0].differential.actuation.nu
    ddp_data['nx'] = ddp.problem.runningModels[0].state.nx
    # Pin model
    ddp_data['pin_model'] = ddp.problem.runningModels[0].differential.pinocchio
    ddp_data['frame_id'] = ddp_data['pin_model'].getFrameId('contact')
    # Solution trajectories
    ddp_data['xs'] = ddp.xs
    ddp_data['us'] = ddp.us
    # Extract force at EE frame
    ddp_data['fs'] = [ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(ddp.problem.T)]
    # Extract refs for active costs 
    ddp_data['active_costs'] = ddp.problem.runningModels[0].differential.costs.active.tolist()
    if('stateReg' in ddp_data['active_costs']):
        ddp_data['stateReg_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['stateReg'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['stateReg_ref'].append(ddp.problem.terminalModel.differential.costs.costs['stateReg'].cost.residual.reference)
    if('stateLim' in ddp_data['active_costs']):
        ddp_data['stateLim_ub'] = [ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.ub for i in range(ddp.problem.T)]
        ddp_data['stateLim_lb'] = [ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.lb for i in range(ddp.problem.T)]
        ddp_data['stateLim_ub'].append(ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.ub)
        ddp_data['stateLim_lb'].append(ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.lb)
    if('ctrlLim' in ddp_data['active_costs']):
        ddp_data['ctrlLim_ub'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub for i in range(ddp.problem.T)]
        ddp_data['ctrlLim_lb'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb for i in range(ddp.problem.T)]
        ddp_data['ctrlLim_ub'].append(ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub)
        ddp_data['ctrlLim_lb'].append(ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb)
    if('placement' in ddp_data['active_costs']):
        ddp_data['placement_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.vector for i in range(ddp.problem.T)]
        ddp_data['placement_ref'].append(ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.vector)
    if('translation' in ddp_data['active_costs']):
        ddp_data['translation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['translation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
    if('velocity' in ddp_data['active_costs']):
        ddp_data['velocity_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['velocity'].cost.residual.reference.vector for i in range(ddp.problem.T)]
        ddp_data['velocity_ref'].append(ddp.problem.terminalModel.differential.costs.costs['velocity'].cost.residual.reference.vector)
        ddp_data['frame_id'] = ddp.problem.runningModels[0].differential.costs.costs['velocity'].cost.residual.id
    if(hasattr(ddp.problem.runningModels[0].differential, 'contacts')):
        ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts["contact"].contact.reference.translation for i in range(ddp.problem.T)]
        ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.reference.translation)
        ddp_data['contact_rotation'] = [ddp.problem.runningModels[i].differential.contacts.contacts["contact"].contact.reference.rotation for i in range(ddp.problem.T)]
        ddp_data['contact_rotation'].append(ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.reference.rotation)
    if('force' in ddp_data['active_costs']): 
        ddp_data['force_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['force'].cost.residual.reference.vector for i in range(ddp.problem.T)]
    return ddp_data

# Extract DDP data (classic or LPF)
def extract_ddp_data_LPF(ddp):
    '''
    Record relevant data from ddp solver in order to plot 
    '''
    return extract_ddp_data(ddp)
