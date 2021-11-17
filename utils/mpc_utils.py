from os import name
import numpy as np
from utils import pin_utils
np.random.seed(1)


class ActuationModel:

    def __init__(self, config, nu=7):
        '''
        Actuation model with parameters defined in config YAML file
        '''
        self.config = config
        self.nu = nu
        # Scaling of desired torque
        self.alpha = np.random.uniform(low=self.config['alpha_min'], high=self.config['alpha_max'], size=(nu,))
        self.beta = np.random.uniform(low=self.config['beta_min'], high=self.config['beta_max'], size=(nu,))
        # PI gains for inner control loop [NOT READY]   
        self.gain_P = self.config['Kp']*np.eye(nu)      
        self.gain_I = self.config['Ki']*np.eye(nu)
        self.err_I = np.zeros(nu)
        # Delays
        self.delay_sim_cycle = int(self.config['delay_sim_cycle'])       # in simu cycles
        self.buffer_sim   = []                                           # buffer for measured torque delayed by e.g. actuation and/or sensing 
        # Actuation model options
        self.DELAY_SIM         = config['DELAY_SIM']                     # Add delay in reference torques (low-level)
        self.SCALE_TORQUES     = config['SCALE_TORQUES']                 # Affinescaling of reference torque
        self.FILTER_TORQUES    = config['FILTER_TORQUES']                # Moving average smoothing of reference torques
        self.TORQUE_TRACKING   = False                                   # NOT READY


    def step(self, i, reference_torque):
        '''
        Transforms reference torque into measured torque
        '''
        measured_torque = reference_torque
        # Affine scaling
        if(self.SCALE_TORQUES):
          measured_torque = self.alpha * measured_torque + self.beta
        # Filtering (moving average)
        if(self.FILTER_TORQUES):
          n_sum = min(i, self.config['u_avg_filter_length'])
          for k in range(n_sum):
            measured_torque += self.data['U_des_SIMU'][i-k-1, :]
          measured_torque = measured_torque / (n_sum + 1)
        # Delay application of torque 
        if(self.DELAY_SIM):
          self.buffer_sim.append(measured_torque)            
          if(len(self.buffer_sim)<self.delay_sim_cycle):    
            pass
          else:                          
            measured_torque = self.buffer_sim.pop(-self.delay_sim_cycle)
        # Inner PID torque control loop [NOT READY]
        if(self.TORQUE_TRACKING):
            self.err_P = measured_torque - reference_torque              
            self.err_I += measured_torque                             
            measured_torque = reference_torque - self.Kp.dot(self.err_P) - self.Ki.dot(self.err_I)
        return measured_torque



class CommunicationModel:

    def __init__(self, config):
        '''
        Communication model with parameters defined in config YAML file
        '''
        self.config = config
        # Delay OCP computation
        self.x_buffer_OCP = []                                           # buffer for desired states delayed by OCP computation time
        self.u_buffer_OCP = []                                           # buffer for desired controls delayed by OCP computation time
        # Sensing model options
        self.DELAY_OCP         = config['DELAY_OCP']                     # Add delay in OCP solution (i.e. ~1ms resolution time)

    def step(self, ddp):
        '''
        Delays input predicted state and current control by 
        using a buffer. Returns the delayed input variables
        '''
        # Delay OCP solution due to computation time
        predicted_state = ddp.xs[1]
        current_control = ddp.us[0] 
        if(self.DELAY_OCP):
          delay = int(self.config['delay_OCP_ms'] * 1e-3 * self.config['plan_freq']) # in planning cycles
          self.x_buffer_OCP.append(predicted_state)
          self.u_buffer_OCP.append(current_control)
          if(len(self.x_buffer_OCP) < delay): 
            pass
          else:                            
            predicted_state = self.x_buffer_OCP.pop(-delay)
          if(len(self.u_buffer_OCP) < delay): 
            pass
          else:
            current_control = self.u_buffer_OCP.pop(-delay)
        return predicted_state, current_control



class SensingModel:

    def __init__(self, config, nq=7, nv=7):
        '''
        Sensing model with parameters defined in config YAML file
        '''
        self.config = config
        self.nq = nq
        self.nv = nv
        # White noise on desired torque and measured state
        self.var_q = np.asarray(self.config['var_q'])
        self.var_v = np.asarray(self.config['var_v'])
        self.var_u = 0.5*np.asarray(self.config['var_u']) 
        # Sensing model options
        self.NOISE_STATE       = config['NOISE_STATE']                   # Add Gaussian noise on the measured state 
        self.FILTER_STATE      = config['FILTER_STATE']                  # Moving average smoothing of reference torques

    def step(self, i, measured_state, memory):
        '''
        Transforms simulator state into a measured state
        '''
        # Optional Gaussian noise on measured state 
        if(self.NOISE_STATE):
          noise_q = np.random.normal(0., self.var_q, self.nq)
          noise_v = np.random.normal(0., self.var_v, self.nv)
          measured_state += np.concatenate([noise_q, noise_v]).T
        # Optional filtering on measured state
        if(self.FILTER_STATE):
          n_sum = min(i, self.config['x_avg_filter_length'])
          for k in range(n_sum):
            measured_state += memory[i-k-1, :]
          measured_state = measured_state / (n_sum + 1)
        return measured_state



class MPCDataRecorder:

  def __init__(self, config, robot, x0):
  
    self.config = config
    self.robot = robot
    self.x0 = x0
    # allocate and initialize data
    self.data = {}
    self.set_params()
    self.init_data()


  def set_params(self):
    '''
    Initialize parameters from config file and robot model
    '''
    # MPC & simulation parameters
    self.data['T_tot'] = self.config['T_tot']                               # Total duration of simulation (s)
    self.data['simu_freq'] = self.config['simu_freq']                       # Simulation frequency
    self.data['ctrl_freq'] = self.config['ctrl_freq']                       # Control frequency (reference sent to motors)
    self.data['plan_freq'] = self.config['plan_freq']                       # Planning frequency (OCP solution update rate)
    self.data['N_plan'] = int(self.data['T_tot']*self.data['plan_freq'])    # Total number of planning steps in the simulation
    self.data['N_ctrl'] = int(self.data['T_tot']*self.data['ctrl_freq'])    # Total number of control steps in the simulation 
    self.data['N_simu'] = int(self.data['T_tot']*self.data['simu_freq'])    # Total number of simulation steps 
    self.data['T_h'] = self.config['N_h']*self.config['dt']                 # Duration of the MPC horizon (s)
    self.data['N_h'] = self.config['N_h']                                   # Number of nodes in MPC horizon
    self.data['dt_ctrl'] = float(1./self.data['ctrl_freq'])                 # Duration of 1 control cycle (s)
    self.data['dt_plan'] = float(1./self.data['plan_freq'])                 # Duration of 1 planning cycle (s)
    self.data['dt_simu'] = float(1./self.data['simu_freq'])                 # Duration of 1 simulation cycle (s)
    # Misc params
    self.data['pin_model'] = self.robot.model
    self.data['nq'] = self.data['pin_model'].nq
    self.data['nv'] = self.data['pin_model'].nv
    self.data['nu'] = self.data['pin_model'].nq
    self.data['nx'] = self.data['nq'] + self.data['nv']
    self.data['id_endeff'] = self.data['pin_model'].getFrameId('contact') # hard-coded contact frame here !!!
    dt_ocp            = self.config['dt']                                 # OCP sampling rate 
    dt_mpc            = float(1./self.data['plan_freq'])                  # planning rate
    self.OCP_TO_PLAN_RATIO = dt_mpc / dt_ocp                              # ratio


  def init_data(self):
    '''
    Main initialization of simulation data 
    '''
    # Predictions
    self.data['X_pred'] = np.zeros((self.data['N_plan'], self.config['N_h']+1, self.data['nx'])) # Predicted states  ( ddp.xs : {x* = (q*, v*)} )
    self.data['U_pred'] = np.zeros((self.data['N_plan'], self.config['N_h'], self.data['nu']))   # Predicted torques ( ddp.us : {u*} )
    self.data['F_pred'] = np.zeros((self.data['N_plan'], self.config['N_h'], 6))                # Predicted EE contact forces
    self.data['X_des_PLAN'] = np.zeros((self.data['N_plan']+1, self.data['nx']))            # Predicted states at planner frequency  ( x* interpolated at PLAN freq )
    self.data['U_des_PLAN'] = np.zeros((self.data['N_plan'], self.data['nu']))              # Predicted torques at planner frequency ( u* interpolated at PLAN freq )
    self.data['F_des_PLAN'] = np.zeros((self.data['N_plan'], 6))                           # Predicted EE contact forces planner frequency  
    self.data['X_des_CTRL'] = np.zeros((self.data['N_ctrl']+1, self.data['nx']))            # Reference state at motor drivers freq ( x* interpolated at CTRL freq )
    self.data['U_des_CTRL'] = np.zeros((self.data['N_ctrl'], self.data['nu']))              # Reference input at motor drivers freq ( u* interpolated at CTRL freq )
    self.data['F_des_CTRL'] = np.zeros((self.data['N_ctrl'], 6))                           # Reference EE contact force at motor drivers freq
    self.data['X_des_SIMU'] = np.zeros((self.data['N_simu']+1, self.data['nx']))            # Reference state at actuation freq ( x* interpolated at SIMU freq )
    self.data['U_des_SIMU'] = np.zeros((self.data['N_simu'], self.data['nu']))              # Reference input at actuation freq ( u* interpolated at SIMU freq )
    self.data['F_des_SIMU'] = np.zeros((self.data['N_simu'], 6))                           # Reference EE contact force at actuation freq
    # Measurements
    self.data['X_mea_SIMU'] = np.zeros((self.data['N_simu']+1, self.data['nx']))            # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq )
    self.data['X_mea_no_noise_SIMU'] = np.zeros((self.data['N_simu']+1, self.data['nx']))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq ) without noise
    self.data['F_mea_SIMU'] = np.zeros((self.data['N_simu'], 6)) 
    self.data['X_mea_SIMU'][0, :] = self.x0
    self.data['X_mea_no_noise_SIMU'][0, :] = self.x0
    # References for reg / goal terms
    # xReg
    if('stateReg' in self.config['WHICH_COSTS']):
      self.data['state_ref'] = np.zeros((self.data['N_plan']+1, self.data['nx']))
    # uReg
    if('ctrlReg' or 'ctrlRegGrav' in self.config['WHICH_COSTS']):
      self.data['ctrl_ref'] = np.zeros((self.data['N_plan']+1, self.data['nu']))
    # EE position
    if('translation' or 'placement' in self.config['WHICH_COSTS']):
      self.data['p_ee_ref'] = np.zeros((self.data['N_plan']+1, 3))
    # EE velocity 
    if('velocity' in self.config['WHICH_COSTS']):
      self.data['v_ee_ref'] = np.zeros((self.data['N_plan']+1, 3))
    # EE force
    if('force' in self.config['WHICH_COSTS']):
      self.data['f_ee_ref'] = np.zeros((self.data['N_plan'], 6)) 
    # Other stuff
    self.data['RECORD_SOLVER_DATA'] = self.config['RECORD_SOLVER_DATA']
    if(self.data['RECORD_SOLVER_DATA']):
      self.data['K'] = np.zeros((self.data['N_plan'], self.config['N_h'], self.data['nq'], self.data['nx']))     # Ricatti gains (K_0)
      self.data['Vxx'] = np.zeros((self.data['N_plan'], self.config['N_h']+1, self.data['nx'], self.data['nx'])) # Hessian of the Value Function  
      self.data['Quu'] = np.zeros((self.data['N_plan'], self.config['N_h'], self.data['nu'], self.data['nu']))   # Hessian of the Value Function 
      self.data['xreg'] = np.zeros(self.data['N_plan'])                                                   # State reg in solver (diag of Vxx)
      self.data['ureg'] = np.zeros(self.data['N_plan'])                                                   # Control reg in solver (diag of Quu)
      self.data['J_rank'] = np.zeros(self.data['N_plan'])                                                 # Rank of Jacobian
  

  def record_ddp_data(self, i, ddp):
    '''
    Record simulation data at the current MPC cycle (to be called each time OCP is re-solved)
      Records predicted state, control over whole horizon 
      Records solver data optionally
    '''
    # Predictions
    self.data['X_pred'][i, :, :] = np.array(ddp.xs)
    self.data['U_pred'][i, :, :] = np.array(ddp.us)
    # Relevant data for interpolation to CTRL frequency
    self.x_curr = self.data['X_pred'][i, 0, :]    
    # Record current references of 1st cost model (node 0)
    dam = ddp.problem.runningModels[0].differential
    if('ctrlReg' in dam.costs.costs.todict().keys()):
      self.data['ctrl_ref'][i, :] = dam.costs.costs['ctrlReg'].cost.residual.reference
    elif('ctrlRegGrav' in dam.costs.costs.todict().keys()):
      self.data['ctrl_ref'][i, :] = pin_utils.get_u_grav(self.x_curr[:self.data['nq']], self.robot.model)
    if('stateReg' in dam.costs.costs.todict().keys()):
      self.data['state_ref'][i, :] = dam.costs.costs['stateReg'].cost.residual.reference
    if('translation' in dam.costs.costs.todict().keys()):
      self.data['p_ee_ref'][i, :] = dam.costs.costs['translation'].cost.residual.reference
    elif('placement' in dam.costs.costs.todict().keys()):
      self.data['p_ee_ref'][i, :] = dam.costs.costs['translation'].cost.residual.reference.translation
    if('velocity' in dam.costs.costs.todict().keys()):
      self.data['v_ee_ref'][i, :] = dam.costs.costs['velocity'].cost.residual.reference.vector[:3]
    if('force' in dam.costs.costs.todict().keys()):
      self.data['f_ee_ref'][i, :] = dam.costs.costs['force'].cost.residual.reference.vector
    # Record solver data (optional)
    if(self.config['RECORD_SOLVER_DATA']):
      self.data['K'][i, :, :, :] = np.array(ddp.K)         # Ricatti gains
      self.data['Vxx'][i, :, :, :] = np.array(ddp.Vxx)     # Hessians of V.F. 
      self.data['Quu'][i, :, :, :] = np.array(ddp.Quu)     # Hessians of Q 
      self.data['xreg'][i] = ddp.x_reg                     # Reg solver on x
      self.data['ureg'][i] = ddp.u_reg                     # Reg solver on u
      self.data['J_rank'][i] = np.linalg.matrix_rank(ddp.problem.runningDatas[0].differential.pinocchio.J)
    

  def record_des_PLAN(self, i, x_pred, u_curr):
    '''
    Record desired state and control 
        i : plan step 
    '''
    # Select reference control and state for the current PLAN cycle
    self.x_ref_PLAN  = self.x_curr + self.OCP_TO_PLAN_RATIO * (x_pred - self.x_curr)
    self.u_ref_PLAN  = u_curr 
    # If first planning step, desired = initial state
    if(i==0):
      self.data['X_des_PLAN'][i, :] = self.x_curr  
    self.data['U_des_PLAN'][i, :]   = self.u_ref_PLAN   
    self.data['X_des_PLAN'][i+1, :] = self.x_ref_PLAN    
  

  def record_des_CTRL(self, i):
    '''
    Record data at each CTRL cycle:
      State ref = current (initial) state + "tiny" step in the direction of next OCP predicted state
      "tiny"    = ratio (OCP sampling frequency / replanning frequency)
      i         : ctrl step
    '''
    # COEF       = float(i%int(freq_CTRL/freq_PLAN)) / float(freq_CTRL/freq_PLAN) # for interpolation PLAN->CTRL
    self.x_ref_CTRL = self.u_ref_PLAN   #self.x_curr + self.OCP_TO_PLAN_RATIO * (self.x_pred - self.x_curr)
    self.u_ref_CTRL = self.x_ref_PLAN   #self.u_curr
    # First prediction = measurement = initialization of MPC
    if(i==0):
      self.data['X_des_CTRL'][i, :] = self.x_curr  
    self.data['U_des_CTRL'][i, :]   = self.u_ref_CTRL  
    self.data['X_des_CTRL'][i+1, :] = self.x_ref_CTRL   


  def record_des_SIMU(self, i):
    '''
    Record data at each SIMU cycle:
      State ref = current (initial) state + "tiny" step in the direction of next OCP predicted state
      "tiny"    = ratio (OCP sampling frequency / replanning frequency)
      i         : simu step
    '''
    # COEF        = float(i%int(freq_SIMU/freq_PLAN)) / float(freq_SIMU/freq_PLAN) # for interpolation CTRL->SIMU
    self.x_ref_SIMU  = self.x_curr + self.OCP_TO_PLAN_RATIO * (self.x_pred - self.x_curr)
    self.u_ref_SIMU  = self.u_curr 
    # First prediction = measurement = initialization of MPC
    if(i==0):
      self.data['X_des_SIMU'][i, :] = self.x_curr  
    self.data['U_des_SIMU'][i, :]   = self.u_ref_SIMU  
    self.data['X_des_SIMU'][i+1, :] = self.x_ref_SIMU 


  def extract_plot_data(self):
    '''
    Extract plot data from simu data
    '''
    print('Extracting plotting data from simulation data...')
    
    plot_data = {}
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
    plot_data['p_ee_ref'] = sim_data['p_ee_ref']
    plot_data['v_ee_ref'] = sim_data['v_ee_ref']
    plot_data['f_ee_ref'] = sim_data['f_ee_ref']
    # Control predictions
    plot_data['u_pred'] = sim_data['U_pred']
    plot_data['u_des_PLAN'] = sim_data['U_des_PLAN']
    plot_data['u_des_CTRL'] = sim_data['U_des_CTRL']
    plot_data['u_des_SIMU'] = sim_data['U_des_SIMU']
    # State predictions (at PLAN freq)
    plot_data['q_pred'] = sim_data['X_pred'][:,:,:nq]
    plot_data['v_pred'] = sim_data['X_pred'][:,:,nq:nq+nv]
    plot_data['q_des_PLAN'] = sim_data['X_des_PLAN'][:,:nq]
    plot_data['v_des_PLAN'] = sim_data['X_des_PLAN'][:,nq:nq+nv] 
    plot_data['q_des_CTRL'] = sim_data['X_des_CTRL'][:,:nq] 
    plot_data['v_des_CTRL'] = sim_data['X_des_CTRL'][:,nq:nq+nv]
    plot_data['q_des_SIMU'] = sim_data['X_des_SIMU'][:,:nq]
    plot_data['v_des_SIMU'] = sim_data['X_des_SIMU'][:,nq:nq+nv]
    # State measurements (at SIMU freq)
    plot_data['q_mea'] = sim_data['X_mea_SIMU'][:,:nq]
    plot_data['v_mea'] = sim_data['X_mea_SIMU'][:,nq:nq+nv]
    plot_data['q_mea_no_noise'] = sim_data['X_mea_no_noise_SIMU'][:,:nq]
    plot_data['v_mea_no_noise'] = sim_data['X_mea_no_noise_SIMU'][:,nq:nq+nv]
    # Extract gravity torques
    plot_data['grav'] = np.zeros((sim_data['N_simu']+1, plot_data['nq']))
    for i in range(plot_data['N_simu']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'])
    # EE predictions (at PLAN freq)
    plot_data['p_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
    plot_data['v_ee_pred'] = np.zeros((sim_data['N_plan'], sim_data['N_h']+1, 3))
    for node_id in range(sim_data['N_h']+1):
        plot_data['p_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['v_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
    # EE measurements (at SIMU freq)
    plot_data['p_ee_mea'] = pin_utils.get_p_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_mea'] = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['p_ee_mea'] = pin_utils.get_p_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['p_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], sim_data['id_endeff'])
    # EE des
    plot_data['p_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['p_ee_des_CTRL'] = pin_utils.get_p_(plot_data['q_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_des_CTRL'] = pin_utils.get_v_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['p_ee_des_SIMU'] = pin_utils.get_p_(plot_data['q_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['v_ee_des_SIMU'] = pin_utils.get_v_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], sim_data['pin_model'], sim_data['id_endeff'])
    # Extract EE force
    plot_data['f_ee_pred'] = sim_data['F_pred']
    plot_data['f_ee_mea'] = sim_data['F_mea_SIMU']
    plot_data['f_ee_des_PLAN'] = sim_data['F_des_PLAN']
    plot_data['f_ee_des_CTRL'] = sim_data['F_des_CTRL']
    plot_data['f_ee_des_SIMU'] = sim_data['F_des_SIMU']

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


