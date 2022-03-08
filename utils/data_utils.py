import time
import numpy as np
import os
from utils import pin_utils
import pinocchio as pin

from utils.misc_utils import CustomLogger
logger = CustomLogger(__name__, log_level_name='DEBUG', USE_LONG_FORMAT=False).logger


# Save data (dict) into compressed npz
def save_data(sim_data, save_name=None, save_dir=None):
    '''
    Saves data to a compressed npz file (binary)
    '''
    logger.info('Compressing & saving data...')
    if(save_name is None):
        save_name = 'sim_data_NO_NAME'+str(time.time())
    if(save_dir is None):
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
    save_path = save_dir+'/'+save_name+'.npz'
    np.savez_compressed(save_path, data=sim_data)
    logger.info("Saved data to "+str(save_path)+" !")


# Loads dict from compressed npz
def load_data(npz_file):
    '''
    Loads a npz archive of sim_data into a dict
    '''
    logger.info('Loading data...')
    d = np.load(npz_file, allow_pickle=True)
    return d['data'][()]








#### Classical MPC
# Initialize simulation data for MPC simulation
def init_sim_data(config, robot, x0, frame_of_interest='contact'):
    '''
    Initialize simulation data from config file
    '''
    sim_data = {}
    # Costs
    sim_data['WHICH_COSTS'] = config['WHICH_COSTS']
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
    sim_data['armature'] = config['armature']
    sim_data['nq'] = sim_data['pin_model'].nq
    sim_data['nv'] = sim_data['pin_model'].nv
    sim_data['nu'] = sim_data['pin_model'].nq
    sim_data['nx'] = sim_data['nq'] + sim_data['nv']
    sim_data['id_endeff'] = sim_data['pin_model'].getFrameId(frame_of_interest) # hard-coded contact frame here !!!
    # Cost references 
    sim_data['ctrl_ref'] = np.zeros((sim_data['N_plan'], sim_data['nu']))
    sim_data['state_ref'] = np.zeros((sim_data['N_plan'], sim_data['nx']))
    sim_data['lin_pos_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['lin_vel_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['ang_pos_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['ang_vel_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['f_ee_ref'] = np.zeros((sim_data['N_plan'], 6))
    # Predictions
    sim_data['state_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, sim_data['nx'])) # Predicted states  ( ddp.xs : {x* = (q*, v*)} )
    sim_data['ctrl_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nu']))   # Predicted torques ( ddp.us : {u*} )
    sim_data['force_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], 6))                # Predicted EE contact forces
    sim_data['state_des_PLAN'] = np.zeros((sim_data['N_plan']+1, sim_data['nx']))            # Predicted states at planner frequency  ( x* interpolated at PLAN freq )
    sim_data['ctrl_des_PLAN'] = np.zeros((sim_data['N_plan'], sim_data['nu']))              # Predicted torques at planner frequency ( u* interpolated at PLAN freq )
    sim_data['force_des_PLAN'] = np.zeros((sim_data['N_plan'], 6))                           # Predicted EE contact forces planner frequency  
    sim_data['state_des_CTRL'] = np.zeros((sim_data['N_ctrl']+1, sim_data['nx']))            # Reference state at motor drivers freq ( x* interpolated at CTRL freq )
    sim_data['ctrl_des_CTRL'] = np.zeros((sim_data['N_ctrl'], sim_data['nu']))              # Reference input at motor drivers freq ( u* interpolated at CTRL freq )
    sim_data['force_des_CTRL'] = np.zeros((sim_data['N_ctrl'], 6))                           # Reference EE contact force at motor drivers freq
    sim_data['state_des_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['nx']))            # Reference state at actuation freq ( x* interpolated at SIMU freq )
    sim_data['ctrl_des_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['nu']))              # Reference input at actuation freq ( u* interpolated at SIMU freq )
    sim_data['force_des_SIMU'] = np.zeros((sim_data['N_simu'], 6))                           # Reference EE contact force at actuation freq
    # Measurements
    sim_data['state_mea_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['nx']))            # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq )
    sim_data['state_mea_no_noise_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['nx']))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq ) without noise
    sim_data['force_mea_SIMU'] = np.zeros((sim_data['N_simu'], 6)) 
    sim_data['state_mea_SIMU'][0, :] = x0
    sim_data['state_mea_no_noise_SIMU'][0, :] = x0
    # Scaling of desired torque
    alpha = np.random.uniform(low=config['alpha_min'], high=config['alpha_max'], size=(sim_data['nq'],))
    beta = np.random.uniform(low=config['beta_min'], high=config['beta_max'], size=(sim_data['nq'],))
    sim_data['alpha'] = alpha
    sim_data['beta'] = beta
    # White noise on desired torque and measured state
    sim_data['var_q'] = np.asarray(config['var_q'])
    sim_data['var_v'] = np.asarray(config['var_v'])
    sim_data['var_u'] = 0.5*np.asarray(config['var_u']) 
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
      sim_data['K'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nq'], sim_data['nx']))     # Ricatti gains (K_0)
      sim_data['Vxx'] = np.zeros((sim_data['N_plan'], config['N_h']+1, sim_data['nx'], sim_data['nx'])) # Hessian of the Value Function  
      sim_data['Quu'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nu'], sim_data['nu']))   # Hessian of the Value Function 
      sim_data['xreg'] = np.zeros(sim_data['N_plan'])                                                   # State reg in solver (diag of Vxx)
      sim_data['ureg'] = np.zeros(sim_data['N_plan'])                                                   # Control reg in solver (diag of Quu)
      sim_data['J_rank'] = np.zeros(sim_data['N_plan'])                                                 # Rank of Jacobian
    logger.info("Initialized MPC simulation data.")

    if(config['INIT_LOG']):
      print('')
      print('                       *************************')
      print('                       ** Simulation is ready **') 
      print('                       *************************')        
      print("-------------------------------------------------------------------")
      print('- Total simulation duration            : T_tot           = '+str(sim_data['T_tot'])+' s')
      print('- Simulation frequency                 : f_simu          = '+str(float(sim_data['simu_freq']/1000.))+' kHz')
      print('- Control frequency                    : f_ctrl          = '+str(float(sim_data['ctrl_freq']/1000.))+' kHz')
      print('- Replanning frequency                 : f_plan          = '+str(float(sim_data['plan_freq']/1000.))+' kHz')
      print('- Total # of simulation steps          : N_simu          = '+str(sim_data['N_simu']))
      print('- Total # of control steps             : N_ctrl          = '+str(sim_data['N_ctrl']))
      print('- Total # of planning steps            : N_plan          = '+str(sim_data['N_plan']))
      print('- Duration of MPC horizon              : T_ocp           = '+str(sim_data['T_h'])+' s')
      print('- OCP integration step                 : dt              = '+str(config['dt'])+' s')
      print('- Simulate delay in low-level torque?  : DELAY_SIM       = '+str(config['DELAY_SIM'])+' ('+str(sim_data['delay_sim_cycle'])+' cycles)')
      print('- Simulate delay in OCP solution?      : DELAY_OCP       = '+str(config['DELAY_OCP'])+' ('+str(config['delay_OCP_ms'])+' ms)')
      print('- Affine scaling of ref. ctrl torque?  : SCALE_TORQUES   = '+str(config['SCALE_TORQUES']))
      if(config['SCALE_TORQUES']):
        print('    a='+str(sim_data['alpha'])+'\n')
        print('    b='+str(sim_data['beta'])+')')
      print('- Noise on torques?                    : NOISE_TORQUES   = '+str(config['NOISE_TORQUES']))
      print('- Filter torques?                      : FILTER_TORQUES  = '+str(config['FILTER_TORQUES']))
      print('- Noise on state?                      : NOISE_STATE     = '+str(config['NOISE_STATE']))
      print('- Filter state?                        : FILTER_STATE    = '+str(config['FILTER_STATE']))
      print("-------------------------------------------------------------------")
      print('')
      time.sleep(config['init_log_display_time'])

    return sim_data



# Record solver data
def record_solver_data(ddp, sim_data, nb_plan):
  '''
  Handy function to record solver related data during MPC simulation
  '''
  sim_data['K'][nb_plan, :, :, :] = np.array(ddp.K)         # Ricatti gains
  sim_data['Vxx'][nb_plan, :, :, :] = np.array(ddp.Vxx)     # Hessians of V.F. 
  sim_data['Quu'][nb_plan, :, :, :] = np.array(ddp.Quu)     # Hessians of Q 
  sim_data['xreg'][nb_plan] = ddp.x_reg                     # Reg solver on x
  sim_data['ureg'][nb_plan] = ddp.u_reg                     # Reg solver on u
  sim_data['J_rank'][nb_plan] = np.linalg.matrix_rank(ddp.problem.runningDatas[0].differential.pinocchio.J)



# Record cost references
def record_cost_references(ddp, sim_data, nb_plan):
  '''
  Handy function for MPC + clean plots
  Extract and record cost references of DAM into sim_data at i^th simulation step
  for the whole horizon (all nodes) 
  '''
  # Get nodes
  m = ddp.problem.runningModels[0]
  # Extract references and record
  if('ctrlReg' in sim_data['WHICH_COSTS']):
    sim_data['ctrl_ref'][nb_plan, :] = m.differential.costs.costs['ctrlReg'].cost.residual.reference
  if('ctrlRegGrav' in sim_data['WHICH_COSTS']):
    q = sim_data['state_pred'][nb_plan, 0, :sim_data['nq']]
    sim_data['ctrl_ref'][nb_plan, :] = pin_utils.get_u_grav(q, m.differential.pinocchio, sim_data['armature'])
  if('force' in sim_data['WHICH_COSTS']):
    sim_data['f_ee_ref'][nb_plan, :] = m.differential.costs.costs['force'].cost.residual.reference.vector
  if('stateReg' in sim_data['WHICH_COSTS']):
    sim_data['state_ref'][nb_plan, :] = m.differential.costs.costs['stateReg'].cost.residual.reference
  if('translation' in sim_data['WHICH_COSTS']):
    sim_data['lin_pos_ee_ref'][nb_plan, :] = m.differential.costs.costs['translation'].cost.residual.reference
  if('rotation' in sim_data['WHICH_COSTS']):
    sim_data['ang_pos_ee_ref'][nb_plan, :] = pin.utils.matrixToRpy(m.differential.costs.costs['rotation'].cost.residual.reference)
  if('velocity' in sim_data['WHICH_COSTS']):
    sim_data['lin_vel_ee_ref'][nb_plan, :] = m.differential.costs.costs['velocity'].cost.residual.reference.vector[:3]
    sim_data['ang_vel_ee_ref'][nb_plan, :] = m.differential.costs.costs['velocity'].cost.residual.reference.vector[3:]
  if('placement' in sim_data['WHICH_COSTS']):
    sim_data['lin_pos_ee_ref'][nb_plan, :] = m.differential.costs.costs['placement'].cost.residual.reference.translation
    sim_data['ang_pos_ee_ref'][nb_plan, :] = pin.utils.matrixToRpy(m.differential.costs.costs['placement'].cost.residual.reference.rotation)



# Extract MPC simu-specific plotting data from sim data
def extract_plot_data_from_sim_data(sim_data):
    '''
    Extract plot data from simu data
    '''
    logger.info('Extracting plot data from simulation data...')
    
    plot_data = {}
    # Get costs
    plot_data['WHICH_COSTS'] = sim_data['WHICH_COSTS']
    # Robot model & params
    plot_data['pin_model'] = sim_data['pin_model']
    nq = plot_data['pin_model'].nq; plot_data['nq'] = nq
    nv = plot_data['pin_model'].nv; plot_data['nv'] = nv
    nx = nq+nv; plot_data['nx'] = nx
    nu = nq
    # MPC params
    plot_data['T_tot'] = sim_data['T_tot']
    plot_data['N_simu'] = sim_data['N_simu']; plot_data['N_ctrl'] = sim_data['N_ctrl']; plot_data['N_plan'] = sim_data['N_plan']
    plot_data['dt_plan'] = sim_data['dt_plan']; plot_data['dt_ctrl'] = sim_data['dt_ctrl']; plot_data['dt_simu'] = sim_data['dt_simu']
    plot_data['T_h'] = sim_data['T_h']; plot_data['N_h'] = sim_data['N_h']
    plot_data['alpha'] = sim_data['alpha']; plot_data['beta'] = sim_data['beta']
    # Record cost references
    plot_data['ctrl_ref'] = sim_data['ctrl_ref']
    plot_data['state_ref'] = sim_data['state_ref']
    plot_data['lin_pos_ee_ref'] = sim_data['lin_pos_ee_ref']
    plot_data['lin_vel_ee_ref'] = sim_data['lin_vel_ee_ref']
    plot_data['ang_pos_ee_ref'] = sim_data['ang_pos_ee_ref']
    plot_data['ang_vel_ee_ref'] = sim_data['ang_vel_ee_ref']
    plot_data['f_ee_ref'] = sim_data['f_ee_ref']
    # Control predictions
    plot_data['u_pred'] = sim_data['ctrl_pred']
    plot_data['u_des_PLAN'] = sim_data['ctrl_des_PLAN']
    plot_data['u_des_CTRL'] = sim_data['ctrl_des_CTRL']
    plot_data['u_des_SIMU'] = sim_data['ctrl_des_SIMU']
    # State predictions (at PLAN freq)
    plot_data['q_pred'] = sim_data['state_pred'][:,:,:nq]
    plot_data['v_pred'] = sim_data['state_pred'][:,:,nq:nq+nv]
    plot_data['q_des_PLAN'] = sim_data['state_des_PLAN'][:,:nq]
    plot_data['v_des_PLAN'] = sim_data['state_des_PLAN'][:,nq:nq+nv] 
    plot_data['q_des_CTRL'] = sim_data['state_des_CTRL'][:,:nq] 
    plot_data['v_des_CTRL'] = sim_data['state_des_CTRL'][:,nq:nq+nv]
    plot_data['q_des_SIMU'] = sim_data['state_des_SIMU'][:,:nq]
    plot_data['v_des_SIMU'] = sim_data['state_des_SIMU'][:,nq:nq+nv]
    # State measurements (at SIMU freq)
    plot_data['q_mea'] = sim_data['state_mea_SIMU'][:,:nq]
    plot_data['v_mea'] = sim_data['state_mea_SIMU'][:,nq:nq+nv]
    plot_data['q_mea_no_noise'] = sim_data['state_mea_no_noise_SIMU'][:,:nq]
    plot_data['v_mea_no_noise'] = sim_data['state_mea_no_noise_SIMU'][:,nq:nq+nv]
    # Extract gravity torques
    plot_data['grav'] = np.zeros((sim_data['N_simu']+1, plot_data['nq']))
    for i in range(plot_data['N_simu']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'], sim_data['armature'])
    # EE predictions (at PLAN freq)
      # Linear position velocity of EE
    plot_data['lin_pos_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
    plot_data['lin_vel_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
      # Angular position velocity of EE
    plot_data['ang_pos_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3)) 
    plot_data['ang_vel_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3)) 
    for node_id in range(sim_data['N_h']+1):
        plot_data['lin_pos_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['lin_vel_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['ang_pos_ee_pred'][:, node_id, :] = pin_utils.get_rpy_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['ang_vel_ee_pred'][:, node_id, :] = pin_utils.get_w_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
    # EE measurements (at SIMU freq)
      # Linear
    plot_data['lin_pos_ee_mea'] = pin_utils.get_p_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_mea'] = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_pos_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
      # Angular
    plot_data['ang_pos_ee_mea'] = pin_utils.get_rpy_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_mea'] = pin_utils.get_w_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_pos_ee_mea_no_noise'] = pin_utils.get_rpy_(plot_data['q_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_mea_no_noise'] = pin_utils.get_w_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    # EE des
      # Linear
    plot_data['lin_pos_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_pos_ee_des_CTRL'] = pin_utils.get_p_(plot_data['q_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_CTRL'] = pin_utils.get_v_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_pos_ee_des_SIMU'] = pin_utils.get_p_(plot_data['q_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_SIMU'] = pin_utils.get_v_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
      # Angular
    plot_data['ang_pos_ee_des_PLAN'] = pin_utils.get_rpy_(plot_data['q_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_PLAN'] = pin_utils.get_w_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_pos_ee_des_CTRL'] = pin_utils.get_rpy_(plot_data['q_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_CTRL'] = pin_utils.get_w_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_pos_ee_des_SIMU'] = pin_utils.get_rpy_(plot_data['q_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_SIMU'] = pin_utils.get_w_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    # Extract EE force
    plot_data['f_ee_pred'] = sim_data['force_pred']
    plot_data['f_ee_mea'] = sim_data['force_mea_SIMU']
    plot_data['f_ee_des_PLAN'] = sim_data['force_des_PLAN']
    plot_data['f_ee_des_CTRL'] = sim_data['force_des_CTRL']
    plot_data['f_ee_des_SIMU'] = sim_data['force_des_SIMU']

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
      plot_data['Vxx_diag'] = np.zeros((sim_data['N_plan'],sim_data['N_h']+1, nx))
      plot_data['Vxx_eig'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, nx))
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






#### Low Pass Filter MPC
# Initialize MPC simulation with torque feedback based on Low-Pass-Filter (LPF) Actuation Model
def init_sim_data_LPF(config, robot, y0, frame_of_interest='contact'):
    '''
    Initialize simulation data from config file (for torque feedback MPC based on LPF)
    '''
    sim_data = {}
    # Get costs names
    sim_data['WHICH_COSTS'] = config['WHICH_COSTS']
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
    # # Misc params
    sim_data['pin_model'] = robot.model
    sim_data['armature'] = config['armature']
    sim_data['nq'] = sim_data['pin_model'].nq
    sim_data['nv'] = sim_data['pin_model'].nv
    sim_data['nu'] = sim_data['pin_model'].nq
    sim_data['nx'] = sim_data['nq'] + sim_data['nv']
    sim_data['ny'] = sim_data['nx'] + sim_data['nu']
    sim_data['id_endeff'] = sim_data['pin_model'].getFrameId(frame_of_interest)
    # Cost references 
    sim_data['ctrl_ref'] = np.zeros((sim_data['N_plan'], sim_data['nu']))
    sim_data['state_ref'] = np.zeros((sim_data['N_plan'], sim_data['nx']))
    sim_data['lin_pos_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['lin_vel_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['ang_pos_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['ang_vel_ee_ref'] = np.zeros((sim_data['N_plan'], 3))
    sim_data['f_ee_ref'] = np.zeros((sim_data['N_plan'], 6))
    # Predictions
    sim_data['state_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, sim_data['ny'])) # Predicted states  ( ddp.xs : {y* = (q*, v*, tau*)} )
    sim_data['ctrl_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], sim_data['nu']))   # Predicted torques ( ddp.us : {w*} )
    sim_data['force_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], 6))                # Predicted EE contact forces
    sim_data['state_des_PLAN'] = np.zeros((sim_data['N_plan']+1, sim_data['ny']))            # Predicted states at planner frequency  ( y* interpolated at PLAN freq )
    sim_data['ctrl_des_PLAN'] = np.zeros((sim_data['N_plan'], sim_data['nu']))              # Predicted torques at planner frequency ( w* interpolated at PLAN freq )
    sim_data['force_des_PLAN'] = np.zeros((sim_data['N_plan'], 6))                           # Predicted EE contact forces planner frequency  
    sim_data['state_des_CTRL'] = np.zeros((sim_data['N_ctrl']+1, sim_data['ny']))            # Reference state at motor drivers freq ( y* interpolated at CTRL freq )
    sim_data['ctrl_des_CTRL'] = np.zeros((sim_data['N_ctrl'], sim_data['nu']))              # Reference input at motor drivers freq ( w* interpolated at CTRL freq )
    sim_data['force_des_CTRL'] = np.zeros((sim_data['N_ctrl'], 6))                           # Reference EE contact force at motor drivers freq
    sim_data['state_des_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))            # Reference state at actuation freq ( y* interpolated at SIMU freq )
    sim_data['ctrl_des_SIMU'] = np.zeros((sim_data['N_simu'], sim_data['nu']))              # Reference input at actuation freq ( w* interpolated at SIMU freq )
    sim_data['force_des_SIMU'] = np.zeros((sim_data['N_simu'], 6))                           # Reference EE contact force at actuation freq
    # Measurements
    sim_data['state_mea_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))            # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq )
    sim_data['state_mea_no_noise_SIMU'] = np.zeros((sim_data['N_simu']+1, sim_data['ny']))   # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq ) without noise
    sim_data['force_mea_SIMU'] = np.zeros((sim_data['N_simu'], 6)) 
    sim_data['state_mea_SIMU'][0, :] = y0
    sim_data['state_mea_no_noise_SIMU'][0, :] = y0
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
    logger.info("Initialized MPC simulation data (LPF).")
    return sim_data



# Record cost references
def record_cost_references_LPF(ddp, sim_data, nb_plan):
  '''
  Handy function for MPC + clean plots
  Extract and record cost references of DAM into sim_data at i^th simulation step
  for the whole horizon (all nodes) 
  '''
  record_cost_references(ddp, sim_data, nb_plan)



# Extract MPC simu-specific plotting data from sim data (LPF)
def extract_plot_data_from_sim_data_LPF(sim_data):
    '''
    Extract plot data from simu data (for torque feedback MPC based on LPF)
    '''
    logger.info('Extracting plot data from MPC simulation data (LPF)...')
    plot_data = {}
    plot_data['WHICH_COSTS'] = sim_data['WHICH_COSTS']
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
    # Record cost references
    plot_data['ctrl_ref'] = sim_data['ctrl_ref']
    plot_data['state_ref'] = sim_data['state_ref']
    plot_data['lin_pos_ee_ref'] = sim_data['lin_pos_ee_ref']
    plot_data['lin_vel_ee_ref'] = sim_data['lin_vel_ee_ref']
    plot_data['ang_pos_ee_ref'] = sim_data['ang_pos_ee_ref']
    plot_data['ang_vel_ee_ref'] = sim_data['ang_vel_ee_ref']
    plot_data['f_ee_ref'] = sim_data['f_ee_ref']
    # Control predictions
    plot_data['w_pred'] = sim_data['ctrl_pred']
      # Extract 1st prediction
    plot_data['w_des_PLAN'] = sim_data['ctrl_des_PLAN']
    plot_data['w_des_CTRL'] = sim_data['ctrl_des_CTRL']
    plot_data['w_des_SIMU'] = sim_data['ctrl_des_SIMU']
    # State predictions (at PLAN freq)
    plot_data['q_pred'] = sim_data['state_pred'][:,:,:nq]
    plot_data['v_pred'] = sim_data['state_pred'][:,:,nq:nq+nv]
    plot_data['tau_pred'] = sim_data['state_pred'][:,:,-nu:]
      # Extract 1st prediction + shift 1 planning cycle
    plot_data['q_des_PLAN'] = sim_data['state_des_PLAN'][:,:nq] 
    plot_data['v_des_PLAN'] = sim_data['state_des_PLAN'][:,nq:nq+nv] 
    plot_data['tau_des_PLAN'] = sim_data['state_des_PLAN'][:,-nu:]
    plot_data['q_des_CTRL'] = sim_data['state_des_CTRL'][:,:nq] 
    plot_data['v_des_CTRL'] = sim_data['state_des_CTRL'][:,nq:nq+nv] 
    plot_data['tau_des_CTRL'] = sim_data['state_des_CTRL'][:,-nu:]
    plot_data['q_des_SIMU'] = sim_data['state_des_SIMU'][:,:nq] 
    plot_data['v_des_SIMU'] = sim_data['state_des_SIMU'][:,nq:nq+nv]
    plot_data['tau_des_SIMU'] = sim_data['state_des_SIMU'][:,-nu:] 
    # State measurements (at SIMU freq)
    plot_data['q_mea'] = sim_data['state_mea_SIMU'][:,:nq]
    plot_data['v_mea'] = sim_data['state_mea_SIMU'][:,nq:nq+nv]
    plot_data['tau_mea'] = sim_data['state_mea_SIMU'][:,-nu:]
    plot_data['q_mea_no_noise'] = sim_data['state_mea_no_noise_SIMU'][:,:nq]
    plot_data['v_mea_no_noise'] = sim_data['state_mea_no_noise_SIMU'][:,nq:nq+nv]
    plot_data['tau_mea_no_noise'] = sim_data['state_mea_no_noise_SIMU'][:,-nu:]
    # Extract gravity torques
    plot_data['grav'] = np.zeros((sim_data['N_simu']+1, plot_data['nq']))
    for i in range(plot_data['N_simu']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'], sim_data['armature'])
    # EE predictions (at PLAN freq)
      # Linear position velocity of EE
    plot_data['lin_pos_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
    plot_data['lin_vel_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
      # Angular position velocity of EE
    plot_data['ang_pos_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3)) 
    plot_data['ang_vel_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3)) 
    for node_id in range(sim_data['N_h']+1):
        plot_data['lin_pos_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['lin_vel_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['ang_pos_ee_pred'][:, node_id, :] = pin_utils.get_rpy_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['ang_vel_ee_pred'][:, node_id, :] = pin_utils.get_w_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
    # EE measurements (at SIMU freq)
      # Linear
    plot_data['lin_pos_ee_mea'] = pin_utils.get_p_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_mea'] = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_pos_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
      # Angular
    plot_data['ang_pos_ee_mea'] = pin_utils.get_rpy_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_mea'] = pin_utils.get_w_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_pos_ee_mea_no_noise'] = pin_utils.get_rpy_(plot_data['q_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_mea_no_noise'] = pin_utils.get_w_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    # EE des
      # Linear
    plot_data['lin_pos_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_pos_ee_des_CTRL'] = pin_utils.get_p_(plot_data['q_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_CTRL'] = pin_utils.get_v_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_pos_ee_des_SIMU'] = pin_utils.get_p_(plot_data['q_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_SIMU'] = pin_utils.get_v_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
      # Angular
    plot_data['ang_pos_ee_des_PLAN'] = pin_utils.get_rpy_(plot_data['q_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_PLAN'] = pin_utils.get_w_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_pos_ee_des_CTRL'] = pin_utils.get_rpy_(plot_data['q_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_CTRL'] = pin_utils.get_w_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_pos_ee_des_SIMU'] = pin_utils.get_rpy_(plot_data['q_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_SIMU'] = pin_utils.get_w_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    # Extract EE force
    plot_data['f_ee_pred'] = sim_data['force_pred']
    plot_data['f_ee_mea'] = sim_data['force_mea_SIMU']
    plot_data['f_ee_des_PLAN'] = sim_data['force_des_PLAN']
    plot_data['f_ee_des_CTRL'] = sim_data['force_des_CTRL']
    plot_data['f_ee_des_SIMU'] = sim_data['force_des_SIMU']
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






#### Classical OCP
def extract_ddp_data(ddp, frame_of_interest='contact'): 
    '''
    Record relevant data from ddp solver in order to plot 
    frame_of_interest = name of frame for which ee plots will be generated
                        by default 'contact' as in KUKA urdf model (Tennis ball)
    '''
    logger.info("Extracting DDP data...")
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
    ddp_data['armature'] = ddp.problem.runningModels[0].differential.armature
    ddp_data['frame_id'] = ddp_data['pin_model'].getFrameId(frame_of_interest)
    # Solution trajectories
    ddp_data['xs'] = ddp.xs
    ddp_data['us'] = ddp.us
    ddp_data['CONTACT_TYPE'] = None
    # Extract force at EE frame and contact info 
    if(hasattr(ddp.problem.runningModels[0].differential, 'contacts')):
      # Get refs for contact model
      contactModelRef0 = ddp.problem.runningModels[0].differential.contacts.contacts["contact"].contact.reference
      # Case 6D contact (x,y,z,Ox,Oy,Oz)
      if(hasattr(contactModelRef0, 'rotation')):
        ddp_data['contact_rotation'] = [ddp.problem.runningModels[i].differential.contacts.contacts["contact"].contact.reference.rotation for i in range(ddp.problem.T)]
        ddp_data['contact_rotation'].append(ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.reference.rotation)
        ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts["contact"].contact.reference.translation for i in range(ddp.problem.T)]
        ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.reference.translation)
        ddp_data['CONTACT_TYPE'] = '6D'
      # Case 3D contact (x,y,z)
      elif(np.size(contactModelRef0)==3):
        # Get ref translation for 3D 
        ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts["contact"].contact.reference for i in range(ddp.problem.T)]
        ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.reference)
        ddp_data['CONTACT_TYPE'] = '3D'
      # Case 1D contact (z)
      elif(np.size(contactModelRef0)==1):
        ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts["contact"].contact.reference for i in range(ddp.problem.T)]
        ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.reference)
        ddp_data['CONTACT_TYPE'] = '1D'
      # Get contact force
      datas = [ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'] for i in range(ddp.problem.T)]
      # data.f = force exerted at parent joint expressed in WORLD frame (oMi)
      # express it in LOCAL contact frame using jMf 
      ee_forces = [data.jMf.actInv(data.f).vector for data in datas] 
      ddp_data['fs'] = [ee_forces[i] for i in range(ddp.problem.T)]
    # Extract refs for active costs 
    # TODO : active costs may change along horizon : how to deal with that when plotting? 
    ddp_data['active_costs'] = ddp.problem.runningModels[0].differential.costs.active.tolist()
    if('stateReg' in ddp_data['active_costs']):
        ddp_data['stateReg_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['stateReg'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['stateReg_ref'].append(ddp.problem.terminalModel.differential.costs.costs['stateReg'].cost.residual.reference)
    if('ctrlReg' in ddp_data['active_costs']):
        ddp_data['ctrlReg_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].cost.residual.reference for i in range(ddp.problem.T)]
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ddp_data['ctrlRegGrav_ref'] = [pin_utils.get_u_grav(ddp.xs[i][:ddp_data['nq']], ddp_data['pin_model'], ddp_data['armature']) for i in range(ddp.problem.T)]
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
        ddp_data['translation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.translation for i in range(ddp.problem.T)]
        ddp_data['translation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.translation)
        ddp_data['rotation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.rotation for i in range(ddp.problem.T)]
        ddp_data['rotation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.rotation)
    if('translation' in ddp_data['active_costs']):
        ddp_data['translation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['translation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
    if('velocity' in ddp_data['active_costs']):
        ddp_data['velocity_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['velocity'].cost.residual.reference.vector for i in range(ddp.problem.T)]
        ddp_data['velocity_ref'].append(ddp.problem.terminalModel.differential.costs.costs['velocity'].cost.residual.reference.vector)
        ddp_data['frame_id'] = ddp.problem.runningModels[0].differential.costs.costs['velocity'].cost.residual.id
    if('rotation' in ddp_data['active_costs']):
        ddp_data['rotation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['rotation'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['rotation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['rotation'].cost.residual.reference)
    if('force' in ddp_data['active_costs']): 
        ddp_data['force_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['force'].cost.residual.reference.vector for i in range(ddp.problem.T)]
    return ddp_data


#### Low Pass Filter OCP
def extract_ddp_data_LPF(ddp, frame_of_interest='contact'):
    '''
    Record relevant data from ddp solver in order to plot 
    '''
    logger.info("Extracting DDP data (LPF)...")
    ddp_data = extract_ddp_data(ddp, frame_of_interest=frame_of_interest)
    # Add terminal regularization references on filtered torques
    if('ctrlReg' in ddp_data['active_costs']):
        ddp_data['ctrlReg_ref'].append(ddp.problem.terminalModel.differential.costs.costs['ctrlReg'].cost.residual.reference)
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ddp_data['ctrlRegGrav_ref'].append(pin_utils.get_u_grav(ddp.xs[-1][:ddp_data['nq']], ddp_data['pin_model'], ddp_data['armature']))
    return ddp_data








# Extract directly plot data 
def extract_plot_data_from_npz(file, LPF=False):
  sim_data = load_data(file)
  if(not LPF):
    plot_data = extract_plot_data_from_sim_data(sim_data)
  else:
    plot_data = extract_plot_data_from_sim_data_LPF(sim_data)
  return plot_data











# # Extract MPC simu-specific plotting data from sim data (LPF)
# def extract_plot_data_from_sim_data_LPF(sim_data):
#     '''
#     Extract plot data from simu data (for torque feedback MPC based on LPF)
#     '''
#     # Extract like regular OCP
#     logger.info('Extracting plot data from MPC simulation data (LPF)...')
#     plot_data = extract_plot_data_from_sim_data(sim_data)
#     nu = plot_data['nu'] ; ny = plot_data['ny']
#     # OVerwrite control with unfitlered torques (control)
#     plot_data['w_pred'] = sim_data['ctrl_pred']
#     plot_data['w_des_PLAN'] = sim_data['ctrl_des_PLAN']
#     plot_data['w_des_CTRL'] = sim_data['ctrl_des_CTRL']
#     plot_data['w_des_SIMU'] = sim_data['ctrl_des_SIMU']
#     # Add filtered torques (state)
#     plot_data['tau_pred'] = sim_data['state_pred'][:,:,-nu:]
#     plot_data['tau_des_PLAN'] = sim_data['state_des_PLAN'][:,-nu:]
#     plot_data['tau_des_CTRL'] = sim_data['state_des_CTRL'][:,-nu:]
#     plot_data['tau_des_SIMU'] = sim_data['state_des_SIMU'][:,-nu:] 
#     plot_data['tau_mea'] = sim_data['state_mea_SIMU'][:,-nu:]
#     plot_data['tau_mea_no_noise'] = sim_data['state_mea_no_noise_SIMU'][:,-nu:]
#     # Solver data (change size o)
#     if(sim_data['RECORD_SOLVER_DATA']):
#       # Get diagonal and eigenvals of Vxx + record in sim data
#       plot_data['Vxx_diag'] = np.zeros((sim_data['N_plan'],sim_data['N_h']+1, ny))
#       plot_data['Vxx_eig'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, ny))
#       for i in range(sim_data['N_plan']):
#         for j in range(sim_data['N_h']+1):
#           plot_data['Vxx_diag'][i, j, :] = sim_data['Vxx'][i, j, :, :].diagonal()
#           plot_data['Vxx_eig'][i, j, :] = np.sort(np.linalg.eigvals(sim_data['Vxx'][i, j, :, :]))[::-1]
#     return plot_data