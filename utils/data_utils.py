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




# # Extract directly plot data 
# def extract_plot_data_from_npz(file, LPF=False):
#   sim_data = load_data(file)
#   if(not LPF):
#     plot_data = extract_plot_data_from_sim_data(sim_data)
#   else:
#     plot_data = extract_plot_data_from_sim_data_LPF(sim_data)
#   return plot_data

