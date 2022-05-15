import time
import numpy as np
import os
from utils import pin_utils
import pinocchio as pin

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



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





class MPCDataHandlerAbstract:

  '''
  Helper class to manage data in MPC simulations
  '''

  def __init__(self, config, robot):

    self.__dict__ = config

    self.rmodel = robot.model
    self.rdata = robot.data

    self.nq = robot.model.nq
    self.nv = robot.model.nv
    self.nu = self.nq
    self.nx = self.nq + self.nv

  def check_attribute(self, attribute): 
    '''
    Check whether attribute exists and is well defined
    '''
    assert(type(attribute)==str), "Attribute to be checked must be a string"
    if(not hasattr(self, attribute)):
      logger.error("The MPC config parameter : "+str(attribute)+ " has not been defined ! Please correct the yaml config file.")

  def check_config(self):
    '''
    Check that config file is complete
    '''
    # general params
    self.check_attribute('simu_freq') #, int)
    self.check_attribute('ctrl_freq') #, int)
    self.check_attribute('plan_freq') #, int)
    self.check_attribute('T_tot')
    self.check_attribute('SAVE_DATA')
    self.check_attribute('RECORD_SOLVER_DATA')
    self.check_attribute('INIT_LOG')
    self.check_attribute('init_log_display_time')
    self.check_attribute('LOG')
    self.check_attribute('log_rate')
    self.check_attribute('WHICH_PLOTS')
    self.check_attribute('RICCATI')

    # actuation model stuff
    self.check_attribute('DELAY_SIM')
    self.check_attribute('DELAY_OCP')
    self.check_attribute('SCALE_TORQUES')
    self.check_attribute('NOISE_TORQUES')
    self.check_attribute('FILTER_TORQUES')
    self.check_attribute('TORQUE_TRACKING')
    self.check_attribute('NOISE_STATE')
    self.check_attribute('FILTER_STATE')
  
    # OCP stuff
    self.check_attribute('dt')
    self.check_attribute('WHICH_COSTS')



  def init_actuation_model(self):
    '''
    Initialize actuation model if necessary
    '''
    if(self.DELAY_OCP):
      self.check_attribute('delay_ocp_ms')
    if(self.DELAY_SIM):
      self.check_attribute('delay_sim_cycle')
    if(self.SCALE_TORQUES):
      self.check_attribute('alpha_min')
      self.check_attribute('alpha_max')
      self.check_attribute('beta_min')
      self.check_attribute('beta_max')
      alpha = np.random.uniform(low=self.alpha_min, high=self.alpha_max, size=(self.nq,))
      beta = np.random.uniform(low=self.beta_min, high=self.beta_max, size=(self.nq,))
      self.alpha = alpha
      self.beta  = beta
    if(self.NOISE_STATE):
      self.check_attribute('var_q')
      self.check_attribute('var_v')
      self.var_q = np.asarray(self.var_q)
      self.var_v = np.asarray(self.var_v)
    if(self.NOISE_TORQUES):
      self.check_attribute('var_u')
      self.var_u = 0.5*np.asarray(self.var_u) 
    if(self.TORQUE_TRACKING):
      self.check_attribute('Kp')
      self.check_attribute('Ki')
      self.check_attribute('Kd')
      self.gain_P = self.Kp*np.eye(self.nq)
      self.gain_I = self.Ki*np.eye(self.nq)
      self.gain_D = self.Kd*np.eye(self.nq)
    if(self.FILTER_STATE):
      self.check_attribute('x_avg_filter_length')
    if(self.FILTER_TORQUES):
      self.check_attribute('u_avg_filter_length')

  def init_solver_data(self):
    '''
    Allocate data for DDP solver stuff (useful to debug)
    '''
    self.K      = np.zeros((self.N_plan, self.N_h, self.nq, self.nx))     # Ricatti gains (K_0)
    self.Vxx    = np.zeros((self.N_plan, self.N_h+1, self.nx, self.nx)) # Hessian of the Value Function  
    self.Quu    = np.zeros((self.N_plan, self.N_h, self.nu, self.nu))   # Hessian of the Value Function 
    self.xreg   = np.zeros(self.N_plan)                                                   # State reg in solver (diag of Vxx)
    self.ureg   = np.zeros(self.N_plan)                                                   # Control reg in solver (diag of Quu)
    self.J_rank = np.zeros(self.N_plan)                                                 # Rank of Jacobian

  def init_cost_references(self):
    '''
    Allocate data for cost references to record
    '''
    if('ctrlReg' in self.WHICH_COSTS or 'ctrlRegGrav' in self.WHICH_COSTS):
      self.ctrl_ref       = np.zeros((self.N_plan, self.nu))
    if('stateReg' in self.WHICH_COSTS):
      self.state_ref      = np.zeros((self.N_plan, self.nx))
    if('translation' in self.WHICH_COSTS or 'placement' in self.WHICH_COSTS):
      self.lin_pos_ee_ref = np.zeros((self.N_plan, 3))
    if('velocity' in self.WHICH_COSTS):
      self.lin_vel_ee_ref = np.zeros((self.N_plan, 3))
      self.ang_vel_ee_ref = np.zeros((self.N_plan, 3))
    if('rotation' in self.WHICH_COSTS):
      self.ang_pos_ee_ref = np.zeros((self.N_plan, 3))
    if('force' in self.WHICH_COSTS):
      self.f_ee_ref       = np.zeros((self.N_plan, 6))

  def print_sim_params(self, sleep):
    '''
    Print out simulation parameters
    '''
    print('')
    print('                       *************************')
    print('                       ** Simulation is ready **') 
    print('                       *************************')        
    print("-------------------------------------------------------------------")
    print('- Total simulation duration            : T_tot           = '+str(self.T_tot)+' s')
    print('- Simulation frequency                 : f_simu          = '+str(float(self.simu_freq/1000.))+' kHz')
    print('- Control frequency                    : f_ctrl          = '+str(float(self.ctrl_freq/1000.))+' kHz')
    print('- Replanning frequency                 : f_plan          = '+str(float(self.plan_freq/1000.))+' kHz')
    print('- Total # of simulation steps          : N_simu          = '+str(self.N_simu))
    print('- Total # of control steps             : N_ctrl          = '+str(self.N_ctrl))
    print('- Total # of planning steps            : N_plan          = '+str(self.N_plan))
    print('- Duration of MPC horizon              : T_ocp           = '+str(self.T_h)+' s')
    print('- OCP integration step                 : dt              = '+str(self.dt)+' s')
    if(self.DELAY_SIM):
      print('- Simulate delay in low-level torque?  : DELAY_SIM       = '+str(self.DELAY_SIM)+' ('+str(self.delay_sim_cycle)+' cycles)')
    if(self.DELAY_OCP):
      print('- Simulate delay in OCP solution?      : DELAY_OCP       = '+str(self.DELAY_OCP)+' ('+str(self.delay_OCP_ms)+' ms)')
    print('- Affine scaling of ref. ctrl torque?  : SCALE_TORQUES   = '+str(self.SCALE_TORQUES))
    if(self.SCALE_TORQUES):
      print('    a='+str(self.alpha)+'\n')
      print('    b='+str(self.beta)+')')
    print('- Noise on torques?                    : NOISE_TORQUES   = '+str(self.NOISE_TORQUES))
    print('- Filter torques?                      : FILTER_TORQUES  = '+str(self.FILTER_TORQUES))
    print('- Noise on state?                      : NOISE_STATE     = '+str(self.NOISE_STATE))
    print('- Filter state?                        : FILTER_STATE    = '+str(self.FILTER_STATE))
    print("-------------------------------------------------------------------")
    print('')
    time.sleep(sleep)



  def record_solver_data(self, ddp, nb_plan):
    '''
    Handy function to record solver related data during MPC simulation
    '''
    if(self.RECORD_SOLVER_DATA):
      self.K[nb_plan, :, :, :]   = np.array(ddp.K)         # Ricatti gains
      self.Vxx[nb_plan, :, :, :] = np.array(ddp.Vxx)       # Hessians of V.F. 
      self.Quu[nb_plan, :, :, :] = np.array(ddp.Quu)       # Hessians of Q 
      self.xreg[nb_plan]         = ddp.x_reg               # Reg solver on x
      self.ureg[nb_plan]         = ddp.u_reg               # Reg solver on u
      self.J_rank[nb_plan]       = np.linalg.matrix_rank(ddp.problem.runningDatas[0].differential.pinocchio.J)

  def record_cost_references(self, ddp, nb_plan):
    '''
    Handy function for MPC + clean plots
    Extract and record cost references of DAM into sim_data at i^th simulation step
     # careful, ref is hard-coded only for the first node
    '''
    # Get nodes
    m = ddp.problem.runningModels[0]
    # Extract references and record
    if('ctrlReg' in self.WHICH_COSTS):
      self.ctrl_ref[nb_plan, :] = m.differential.costs.costs['ctrlReg'].cost.residual.reference
    if('ctrlRegGrav' in self.WHICH_COSTS):
      q = self.state_pred[nb_plan, 0, :self.nq]
      self.ctrl_ref[nb_plan, :] = pin_utils.get_u_grav(q, m.differential.pinocchio, self.armature)
    if('force' in self.WHICH_COSTS):
      self.f_ee_ref[nb_plan, :] = m.differential.costs.costs['force'].cost.residual.reference.vector
    if('stateReg' in self.WHICH_COSTS):
      self.state_ref[nb_plan, :] = m.differential.costs.costs['stateReg'].cost.residual.reference
    if('translation' in self.WHICH_COSTS):
      self.lin_pos_ee_ref[nb_plan, :] = m.differential.costs.costs['translation'].cost.residual.reference
    if('rotation' in self.WHICH_COSTS):
      self.ang_pos_ee_ref[nb_plan, :] = pin.utils.matrixToRpy(m.differential.costs.costs['rotation'].cost.residual.reference)
    if('velocity' in self.WHICH_COSTS):
      self.lin_vel_ee_ref[nb_plan, :] = m.differential.costs.costs['velocity'].cost.residual.reference.vector[:3]
      self.ang_vel_ee_ref[nb_plan, :] = m.differential.costs.costs['velocity'].cost.residual.reference.vector[3:]
    if('placement' in self.WHICH_COSTS):
      self.lin_pos_ee_ref[nb_plan, :] = m.differential.costs.costs['placement'].cost.residual.reference.translation
      self.ang_pos_ee_ref[nb_plan, :] = pin.utils.matrixToRpy(m.differential.costs.costs['placement'].cost.residual.reference.rotation)



# #### Low Pass Filter MPC
# # Initialize MPC simulation with torque feedback based on Low-Pass-Filter (LPF) Actuation Model
# def init_sim_data_LPF(config, robot, y0, ee_frame_name='contact'):
#     '''
#     Initialize simulation data from config file (for torque feedback MPC based on LPF)
#     '''
#     sim_data = {}
#     # Get costs names
#     self.WHICH_COSTS'] = config['WHICH_COSTS']
#     # MPC & simulation parameters
#     self.T_tot'] = config['T_tot']                               # Total duration of simulation (s)
#     self.simu_freq'] = config['simu_freq']                       # Simulation frequency
#     self.ctrl_freq'] = config['ctrl_freq']                       # Control frequency (reference sent to motors)
#     self.plan_freq'] = config['plan_freq']                       # Planning frequency (OCP solution update rate)
#     self.N_plan'] = int(self.T_tot']*self.plan_freq']) # Total number of planning steps in the simulation
#     self.N_ctrl'] = int(self.T_tot']*self.ctrl_freq']) # Total number of control steps in the simulation 
#     self.N_simu'] = int(self.T_tot']*self.simu_freq']) # Total number of simulation steps 
#     self.T_h'] = config['N_h']*config['dt']                      # Duration of the MPC horizon (s)
#     self.N_h'] = config['N_h']                                   # Number of nodes in MPC horizon
#     self.dt_ctrl'] = float(1./self.ctrl_freq'])             # Duration of 1 control cycle (s)
#     self.dt_plan'] = float(1./self.plan_freq'])             # Duration of 1 planning cycle (s)
#     self.dt_simu'] = float(1./self.simu_freq'])             # Duration of 1 simulation cycle (s)
#     # # Misc params
#     self.rmodel'] = robot.model
#     self.armature'] = config['armature']
#     self.nq'] = self.rmodel'].nq
#     self.nv'] = self.rmodel'].nv
#     self.nu'] = self.rmodel'].nq
#     self.nx'] = self.nq'] + self.nv']
#     self.ny'] = self.nx'] + self.nu']
#     self.id_endeff'] = self.rmodel'].getFrameId(ee_frame_name)
#     # Cost references 
#     self.ctrl_ref'] = np.zeros((self.N_plan'], self.nu']))
#     self.state_ref'] = np.zeros((self.N_plan'], self.nx']))
#     self.lin_pos_ee_ref'] = np.zeros((self.N_plan'], 3))
#     self.lin_vel_ee_ref'] = np.zeros((self.N_plan'], 3))
#     self.ang_pos_ee_ref'] = np.zeros((self.N_plan'], 3))
#     self.ang_vel_ee_ref'] = np.zeros((self.N_plan'], 3))
#     self.f_ee_ref'] = np.zeros((self.N_plan'], 6))
#     # Predictions
#     self.state_pred'] = np.zeros((self.N_plan'], config['N_h']+1, self.ny'])) # Predicted states  ( ddp.xs : {y* = (q*, v*, tau*)} )
#     self.ctrl_pred'] = np.zeros((self.N_plan'], config['N_h'], self.nu']))   # Predicted torques ( ddp.us : {w*} )
#     self.force_pred'] = np.zeros((self.N_plan'], config['N_h'], 6))                # Predicted EE contact forces
#     self.state_des_PLAN'] = np.zeros((self.N_plan']+1, self.ny']))            # Predicted states at planner frequency  ( y* interpolated at PLAN freq )
#     self.ctrl_des_PLAN'] = np.zeros((self.N_plan'], self.nu']))              # Predicted torques at planner frequency ( w* interpolated at PLAN freq )
#     self.force_des_PLAN'] = np.zeros((self.N_plan'], 6))                           # Predicted EE contact forces planner frequency  
#     self.state_des_CTRL'] = np.zeros((self.N_ctrl']+1, self.ny']))            # Reference state at motor drivers freq ( y* interpolated at CTRL freq )
#     self.ctrl_des_CTRL'] = np.zeros((self.N_ctrl'], self.nu']))              # Reference input at motor drivers freq ( w* interpolated at CTRL freq )
#     self.force_des_CTRL'] = np.zeros((self.N_ctrl'], 6))                           # Reference EE contact force at motor drivers freq
#     self.state_des_SIMU'] = np.zeros((self.N_simu']+1, self.ny']))            # Reference state at actuation freq ( y* interpolated at SIMU freq )
#     self.ctrl_des_SIMU'] = np.zeros((self.N_simu'], self.nu']))              # Reference input at actuation freq ( w* interpolated at SIMU freq )
#     self.force_des_SIMU'] = np.zeros((self.N_simu'], 6))                           # Reference EE contact force at actuation freq
#     # Measurements
#     self.state_mea_SIMU'] = np.zeros((self.N_simu']+1, self.ny']))            # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq )
#     self.state_mea_no_noise_SIMU'] = np.zeros((self.N_simu']+1, self.ny']))   # Measured states ( y^mea = (q, v, tau) from actuator & PyB at SIMU freq ) without noise
#     self.force_mea_SIMU'] = np.zeros((self.N_simu'], 6)) 
#     self.state_mea_SIMU'][0, :] = y0
#     self.state_mea_no_noise_SIMU'][0, :] = y0
#     # Scaling of desired torque
#     alpha = np.random.uniform(low=config['alpha_min'], high=config['alpha_max'], size=(self.nq'],))
#     beta = np.random.uniform(low=config['beta_min'], high=config['beta_max'], size=(self.nq'],))
#     self.alpha'] = alpha
#     self.beta'] = beta
#     # White noise on desired torque and measured state
#     self.var_q'] = np.asarray(config['var_q'])
#     self.var_v'] = np.asarray(config['var_v'])
#     self.var_u'] = 0.5*np.asarray(config['var_u']) #0.5% of range on the joint
#     # White noise on desired torque and measured state
#     self.gain_P'] = config['Kp']*np.eye(self.nq'])
#     self.gain_I'] = config['Ki']*np.eye(self.nq'])
#     self.gain_D'] = config['Kd']*np.eye(self.nq'])
#     # Delays
#     self.delay_OCP_cycle'] = int(config['delay_OCP_ms'] * 1e-3 * self.plan_freq']) # in planning cycles
#     self.delay_sim_cycle'] = int(config['delay_sim_cycle'])                             # in simu cycles
#     # Other stuff
#     self.RECORD_SOLVER_DATA'] = config['RECORD_SOLVER_DATA']
#     if(self.RECORD_SOLVER_DATA']):
#       self.K'] = np.zeros((self.N_plan'], config['N_h'], self.nq'], self.ny']))     # Ricatti gains (K_0)
#       self.Vxx'] = np.zeros((self.N_plan'], config['N_h']+1, self.ny'], self.ny'])) # Hessian of the Value Function  
#       self.Quu'] = np.zeros((self.N_plan'], config['N_h'], self.nu'], self.nu']))   # Hessian of the Value Function 
#       self.xreg'] = np.zeros(self.N_plan'])                                                   # State reg in solver (diag of Vxx)
#       self.ureg'] = np.zeros(self.N_plan'])                                                   # Control reg in solver (diag of Quu)
#       self.J_rank'] = np.zeros(self.N_plan'])                                                 # Rank of Jacobian
#     logger.info("Initialized MPC simulation data (LPF).")
#     return sim_data



# # Record cost references
# def record_cost_references_LPF(ddp, sim_data, nb_plan):
#   '''
#   Handy function for MPC + clean plots
#   Extract and record cost references of DAM into sim_data at i^th simulation step
#   for the whole horizon (all nodes) 
#   '''
#   record_cost_references(ddp, sim_data, nb_plan)



# # Extract MPC simu-specific plotting data from sim data (LPF)
# def extract_plot_data_from_sim_data_LPF(sim_data):
#     '''
#     Extract plot data from simu data (for torque feedback MPC based on LPF)
#     '''
#     logger.info('Extracting plot data from MPC simulation data (LPF)...')
#     plot_data = {}
#     plot_data['WHICH_COSTS'] = self.WHICH_COSTS']
#     # Robot model & params
#     plot_data['pin_model'] = self.rmodel']
#     nq = plot_data['pin_model'].nq; plot_data['nq'] = nq
#     nv = plot_data['pin_model'].nv; plot_data['nv'] = nv
#     nx = nq+nv; plot_data['nx'] = nx
#     ny = self.ny']; plot_data['ny'] = ny
#     nu = nq
#     # MPC params
#     plot_data['T_tot'] = self.T_tot']
#     plot_data['N_simu'] = self.N_simu']; plot_data['N_ctrl'] = self.N_ctrl']; plot_data['N_plan'] = self.N_plan']
#     plot_data['dt_plan'] = self.dt_plan']; plot_data['dt_ctrl'] = self.dt_ctrl']; plot_data['dt_simu'] = self.dt_simu']
#     plot_data['T_h'] = self.T_h']; plot_data['N_h'] = self.N_h']
#     plot_data['alpha'] = self.alpha']; plot_data['beta'] = self.beta']
#     # Record cost references
#     plot_data['ctrl_ref'] = self.ctrl_ref']
#     plot_data['state_ref'] = self.state_ref']
#     plot_data['lin_pos_ee_ref'] = self.lin_pos_ee_ref']
#     plot_data['lin_vel_ee_ref'] = self.lin_vel_ee_ref']
#     plot_data['ang_pos_ee_ref'] = self.ang_pos_ee_ref']
#     plot_data['ang_vel_ee_ref'] = self.ang_vel_ee_ref']
#     plot_data['f_ee_ref'] = self.f_ee_ref']
#     # Control predictions
#     plot_data['w_pred'] = self.ctrl_pred']
#       # Extract 1st prediction
#     plot_data['w_des_PLAN'] = self.ctrl_des_PLAN']
#     plot_data['w_des_CTRL'] = self.ctrl_des_CTRL']
#     plot_data['w_des_SIMU'] = self.ctrl_des_SIMU']
#     # State predictions (at PLAN freq)
#     plot_data['q_pred'] = self.state_pred'][:,:,:nq]
#     plot_data['v_pred'] = self.state_pred'][:,:,nq:nq+nv]
#     plot_data['tau_pred'] = self.state_pred'][:,:,-nu:]
#       # Extract 1st prediction + shift 1 planning cycle
#     plot_data['q_des_PLAN'] = self.state_des_PLAN'][:,:nq] 
#     plot_data['v_des_PLAN'] = self.state_des_PLAN'][:,nq:nq+nv] 
#     plot_data['tau_des_PLAN'] = self.state_des_PLAN'][:,-nu:]
#     plot_data['q_des_CTRL'] = self.state_des_CTRL'][:,:nq] 
#     plot_data['v_des_CTRL'] = self.state_des_CTRL'][:,nq:nq+nv] 
#     plot_data['tau_des_CTRL'] = self.state_des_CTRL'][:,-nu:]
#     plot_data['q_des_SIMU'] = self.state_des_SIMU'][:,:nq] 
#     plot_data['v_des_SIMU'] = self.state_des_SIMU'][:,nq:nq+nv]
#     plot_data['tau_des_SIMU'] = self.state_des_SIMU'][:,-nu:] 
#     # State measurements (at SIMU freq)
#     plot_data['q_mea'] = self.state_mea_SIMU'][:,:nq]
#     plot_data['v_mea'] = self.state_mea_SIMU'][:,nq:nq+nv]
#     plot_data['tau_mea'] = self.state_mea_SIMU'][:,-nu:]
#     plot_data['q_mea_no_noise'] = self.state_mea_no_noise_SIMU'][:,:nq]
#     plot_data['v_mea_no_noise'] = self.state_mea_no_noise_SIMU'][:,nq:nq+nv]
#     plot_data['tau_mea_no_noise'] = self.state_mea_no_noise_SIMU'][:,-nu:]
#     # Extract gravity torques
#     plot_data['grav'] = np.zeros((self.N_simu']+1, plot_data['nq']))
#     for i in range(plot_data['N_simu']+1):
#       plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'], self.armature'])
#     # EE predictions (at PLAN freq)
#       # Linear position velocity of EE
#     plot_data['lin_pos_ee_pred'] = np.zeros((self.N_plan'], self.N_h']+1, 3))
#     plot_data['lin_vel_ee_pred'] = np.zeros((self.N_plan'], self.N_h']+1, 3))
#       # Angular position velocity of EE
#     plot_data['ang_pos_ee_pred'] = np.zeros((self.N_plan'], self.N_h']+1, 3)) 
#     plot_data['ang_vel_ee_pred'] = np.zeros((self.N_plan'], self.N_h']+1, 3)) 
#     for node_id in range(self.N_h']+1):
#         plot_data['lin_pos_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff'])
#         plot_data['lin_vel_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff'])
#         plot_data['ang_pos_ee_pred'][:, node_id, :] = pin_utils.get_rpy_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff'])
#         plot_data['ang_vel_ee_pred'][:, node_id, :] = pin_utils.get_w_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff'])
#     # EE measurements (at SIMU freq)
#       # Linear
#     plot_data['lin_pos_ee_mea'] = pin_utils.get_p_(plot_data['q_mea'], self.rmodel'], self.id_endeff'])
#     plot_data['lin_vel_ee_mea'] = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel'], self.id_endeff'])
#     plot_data['lin_pos_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff'])
#     plot_data['lin_vel_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff'])
#       # Angular
#     plot_data['ang_pos_ee_mea'] = pin_utils.get_rpy_(plot_data['q_mea'], self.rmodel'], self.id_endeff'])
#     plot_data['ang_vel_ee_mea'] = pin_utils.get_w_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel'], self.id_endeff'])
#     plot_data['ang_pos_ee_mea_no_noise'] = pin_utils.get_rpy_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff'])
#     plot_data['ang_vel_ee_mea_no_noise'] = pin_utils.get_w_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff'])
#     # EE des
#       # Linear
#     plot_data['lin_pos_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], self.rmodel'], self.id_endeff'])
#     plot_data['lin_vel_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel'], self.id_endeff'])
#     plot_data['lin_pos_ee_des_CTRL'] = pin_utils.get_p_(plot_data['q_des_CTRL'], self.rmodel'], self.id_endeff'])
#     plot_data['lin_vel_ee_des_CTRL'] = pin_utils.get_v_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], self.rmodel'], self.id_endeff'])
#     plot_data['lin_pos_ee_des_SIMU'] = pin_utils.get_p_(plot_data['q_des_SIMU'], self.rmodel'], self.id_endeff'])
#     plot_data['lin_vel_ee_des_SIMU'] = pin_utils.get_v_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], self.rmodel'], self.id_endeff'])
#       # Angular
#     plot_data['ang_pos_ee_des_PLAN'] = pin_utils.get_rpy_(plot_data['q_des_PLAN'], self.rmodel'], self.id_endeff'])
#     plot_data['ang_vel_ee_des_PLAN'] = pin_utils.get_w_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel'], self.id_endeff'])
#     plot_data['ang_pos_ee_des_CTRL'] = pin_utils.get_rpy_(plot_data['q_des_CTRL'], self.rmodel'], self.id_endeff'])
#     plot_data['ang_vel_ee_des_CTRL'] = pin_utils.get_w_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], self.rmodel'], self.id_endeff'])
#     plot_data['ang_pos_ee_des_SIMU'] = pin_utils.get_rpy_(plot_data['q_des_SIMU'], self.rmodel'], self.id_endeff'])
#     plot_data['ang_vel_ee_des_SIMU'] = pin_utils.get_w_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], self.rmodel'], self.id_endeff'])
#     # Extract EE force
#     plot_data['f_ee_pred'] = self.force_pred']
#     plot_data['f_ee_mea'] = self.force_mea_SIMU']
#     plot_data['f_ee_des_PLAN'] = self.force_des_PLAN']
#     plot_data['f_ee_des_CTRL'] = self.force_des_CTRL']
#     plot_data['f_ee_des_SIMU'] = self.force_des_SIMU']
#     # Solver data (optional)
#     if(self.RECORD_SOLVER_DATA']):
#       # Get SVD & diagonal of Ricatti + record in sim data
#       plot_data['K_svd'] = np.zeros((self.N_plan'], self.N_h'], nq))
#       plot_data['Kp_diag'] = np.zeros((self.N_plan'], self.N_h'], nq))
#       plot_data['Kv_diag'] = np.zeros((self.N_plan'], self.N_h'], nv))
#       plot_data['Ktau_diag'] = np.zeros((self.N_plan'], self.N_h'], nu))
#       for i in range(self.N_plan']):
#         for j in range(self.N_h']):
#           plot_data['Kp_diag'][i, j, :] = self.K'][i, j, :, :nq].diagonal()
#           plot_data['Kv_diag'][i, j, :] = self.K'][i, j, :, nq:nq+nv].diagonal()
#           plot_data['Ktau_diag'][i, j, :] = self.K'][i, j, :, -nu:].diagonal()
#           _, sv, _ = np.linalg.svd(self.K'][i, j, :, :])
#           plot_data['K_svd'][i, j, :] = np.sort(sv)[::-1]
#       # Get diagonal and eigenvals of Vxx + record in sim data
#       plot_data['Vxx_diag'] = np.zeros((self.N_plan'],self.N_h']+1, ny))
#       plot_data['Vxx_eig'] = np.zeros((self.N_plan'], self.N_h']+1, ny))
#       for i in range(self.N_plan']):
#         for j in range(self.N_h']+1):
#           plot_data['Vxx_diag'][i, j, :] = self.Vxx'][i, j, :, :].diagonal()
#           plot_data['Vxx_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Vxx'][i, j, :, :]))[::-1]
#       # Get diagonal and eigenvals of Quu + record in sim data
#       plot_data['Quu_diag'] = np.zeros((self.N_plan'],self.N_h'], nu))
#       plot_data['Quu_eig'] = np.zeros((self.N_plan'], self.N_h'], nu))
#       for i in range(self.N_plan']):
#         for j in range(self.N_h']):
#           plot_data['Quu_diag'][i, j, :] = self.Quu'][i, j, :, :].diagonal()
#           plot_data['Quu_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Quu'][i, j, :, :]))[::-1]
#       # Get Jacobian
#       plot_data['J_rank'] = self.J_rank']
#       # Get solve regs
#       plot_data['xreg'] = self.xreg']
#       plot_data['ureg'] = self.ureg']

#     return plot_data






# #### Classical OCP
# def extract_ddp_data(ddp, ee_frame_name, ct_frame_name, ref=pin.LOCAL): 
#     '''
#     Record relevant data from ddp solver in order to plot 
#       ddp           : DDP solver object
#       ee_frame_name : name of frame for which ee plots will be generated
#       ct_frame_name : name of frame for which force plots will be generated
#       REF           : name of the reference frame for contact models (TODO: deduce it from solver parsing)
#     '''
#     logger.info("Extracting DDP data...")
#     # Store data
#     ddp_data = {}
#     # OCP params
#     ddp_data['T'] = ddp.problem.T
#     ddp_data['dt'] = ddp.problem.runningModels[0].dt
#     ddp_data['nq'] = ddp.problem.runningModels[0].state.nq
#     ddp_data['nv'] = ddp.problem.runningModels[0].state.nv
#     ddp_data['nu'] = ddp.problem.runningModels[0].differential.actuation.nu
#     ddp_data['nx'] = ddp.problem.runningModels[0].state.nx
#     # Pin model
#     ddp_data['pin_model'] = ddp.problem.runningModels[0].differential.pinocchio
#     ddp_data['armature'] = ddp.problem.runningModels[0].differential.armature
#     ddp_data['frame_id'] = ddp_data['pin_model'].getFrameId(ee_frame_name)
#     # Solution trajectories
#     ddp_data['xs'] = ddp.xs
#     ddp_data['us'] = ddp.us
#     ddp_data['CONTACT_TYPE'] = None
#     # Extract force at EE frame and contact info
#     if(hasattr(ddp.problem.runningModels[0].differential, 'contacts')):
#       # Get refs for contact model
#       contactModelRef0 = ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.reference
#       # Case 6D contact (x,y,z,Ox,Oy,Oz)
#       if(hasattr(contactModelRef0, 'rotation')):
#         ddp_data['contact_rotation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.rotation for i in range(ddp.problem.T)]
#         ddp_data['contact_rotation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.rotation)
#         ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.translation for i in range(ddp.problem.T)]
#         ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.translation)
#         ddp_data['CONTACT_TYPE'] = '6D'
#         PIN_REF_FRAME = pin.LOCAL
#       # Case 3D contact (x,y,z)
#       elif(np.size(contactModelRef0)==3):
#         if(ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 3):
#           # Get ref translation for 3D 
#           ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(ddp.problem.T)]
#           ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
#           ddp_data['CONTACT_TYPE'] = '3D'
#         elif(ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 1):
#           # Case 1D contact
#           ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(ddp.problem.T)]
#           ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
#           ddp_data['CONTACT_TYPE'] = '1D'
#         else: 
#           print(ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 3)
#           logger.error("Contact must be 1D or 3D !")
#         # Check which reference frame is used 
#         if(ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.type == pin.pinocchio_pywrap.ReferenceFrame.LOCAL):
#           PIN_REF_FRAME = pin.LOCAL
#         else:
#           PIN_REF_FRAME = pin.LOCAL_WORLD_ALIGNED
#       # Get contact force
#       datas = [ddp.problem.runningDatas[i].differential.multibody.contacts.contacts[ct_frame_name] for i in range(ddp.problem.T)]
#       # data.f = force exerted at parent joint expressed in WORLD frame (oMi)
#       # express it in LOCAL contact frame using jMf 
#       ee_forces = [data.jMf.actInv(data.f).vector for data in datas] 
#       ddp_data['fs'] = [ee_forces[i] for i in range(ddp.problem.T)]
#       # Express in WORLD aligned frame otherwise
#       if(PIN_REF_FRAME == pin.LOCAL_WORLD_ALIGNED or PIN_REF_FRAME == pin.WORLD):
#         ct_frame_id = ddp_data['pin_model'].getFrameId(ct_frame_name)
#         Ms = [pin_utils.get_SE3_(ddp.xs[i][:ddp_data['nq']], ddp_data['pin_model'], ct_frame_id) for i in range(ddp.problem.T)]
#         ddp_data['fs'] = [Ms[i].action @ ee_forces[i] for i in range(ddp.problem.T)]
#     # Extract refs for active costs 
#     # TODO : active costs may change along horizon : how to deal with that when plotting? 
#     ddp_data['active_costs'] = ddp.problem.runningModels[0].differential.costs.active.tolist()
#     if('stateReg' in ddp_data['active_costs']):
#         ddp_data['stateReg_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['stateReg'].cost.residual.reference for i in range(ddp.problem.T)]
#         ddp_data['stateReg_ref'].append(ddp.problem.terminalModel.differential.costs.costs['stateReg'].cost.residual.reference)
#     if('ctrlReg' in ddp_data['active_costs']):
#         ddp_data['ctrlReg_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].cost.residual.reference for i in range(ddp.problem.T)]
#     if('ctrlRegGrav' in ddp_data['active_costs']):
#         ddp_data['ctrlRegGrav_ref'] = [pin_utils.get_u_grav(ddp.xs[i][:ddp_data['nq']], ddp_data['pin_model'], ddp_data['armature']) for i in range(ddp.problem.T)]
#     if('stateLim' in ddp_data['active_costs']):
#         ddp_data['stateLim_ub'] = [ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.ub for i in range(ddp.problem.T)]
#         ddp_data['stateLim_lb'] = [ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.lb for i in range(ddp.problem.T)]
#         ddp_data['stateLim_ub'].append(ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.ub)
#         ddp_data['stateLim_lb'].append(ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.lb)
#     if('ctrlLim' in ddp_data['active_costs']):
#         ddp_data['ctrlLim_ub'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub for i in range(ddp.problem.T)]
#         ddp_data['ctrlLim_lb'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb for i in range(ddp.problem.T)]
#         ddp_data['ctrlLim_ub'].append(ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub)
#         ddp_data['ctrlLim_lb'].append(ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb)
#     if('placement' in ddp_data['active_costs']):
#         ddp_data['translation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.translation for i in range(ddp.problem.T)]
#         ddp_data['translation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.translation)
#         ddp_data['rotation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.rotation for i in range(ddp.problem.T)]
#         ddp_data['rotation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.rotation)
#     if('translation' in ddp_data['active_costs']):
#         ddp_data['translation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(ddp.problem.T)]
#         ddp_data['translation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
#     if('velocity' in ddp_data['active_costs']):
#         ddp_data['velocity_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['velocity'].cost.residual.reference.vector for i in range(ddp.problem.T)]
#         ddp_data['velocity_ref'].append(ddp.problem.terminalModel.differential.costs.costs['velocity'].cost.residual.reference.vector)
#         # ddp_data['frame_id'] = ddp.problem.runningModels[0].differential.costs.costs['velocity'].cost.residual.id
#     if('rotation' in ddp_data['active_costs']):
#         ddp_data['rotation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['rotation'].cost.residual.reference for i in range(ddp.problem.T)]
#         ddp_data['rotation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['rotation'].cost.residual.reference)
#     if('force' in ddp_data['active_costs']): 
#         ddp_data['force_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['force'].cost.residual.reference.vector for i in range(ddp.problem.T)]
#     return ddp_data


# #### Low Pass Filter OCP
# def extract_ddp_data_LPF(ddp, ee_frame_name='contact', ct_frame_name='contact'):
#     '''
#     Record relevant data from ddp solver in order to plot 
#     '''
#     logger.info("Extracting DDP data (LPF)...")
#     ddp_data = extract_ddp_data(ddp, ee_frame_name=ee_frame_name, ct_frame_name=ct_frame_name)
#     # Add terminal regularization references on filtered torques
#     if('ctrlReg' in ddp_data['active_costs']):
#         ddp_data['ctrlReg_ref'].append(ddp.problem.terminalModel.differential.costs.costs['ctrlReg'].cost.residual.reference)
#     if('ctrlRegGrav' in ddp_data['active_costs']):
#         ddp_data['ctrlRegGrav_ref'].append(pin_utils.get_u_grav(ddp.xs[-1][:ddp_data['nq']], ddp_data['pin_model'], ddp_data['armature']))
#     return ddp_data








# # Extract directly plot data 
# def extract_plot_data_from_npz(file, LPF=False):
#   sim_data = load_data(file)
#   if(not LPF):
#     plot_data = extract_plot_data_from_sim_data(sim_data)
#   else:
#     plot_data = extract_plot_data_from_sim_data_LPF(sim_data)
#   return plot_data

