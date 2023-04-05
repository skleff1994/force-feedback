
from curses import noqiflush
import numpy as np

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



class ActuationModel:

    def __init__(self, config, nu, SEED=1):
        '''
        Actuation model with parameters defined in config YAML file
        Simulates (optionally) 
         - affine bias on torques a*tau_ref(i) + b
         - moving avg filter on tau_ref(i)
         - delay tau_ref(i) = tau_ref(i-delay)
         - torque PI control (tau_mea, tau_ref)
        '''
        np.random.seed(SEED)
        self.config = config
        self.nu = nu
        # Scaling of desired torque
        self.alpha = np.random.uniform(low=self.config['alpha_min'], high=self.config['alpha_max'], size=(nu,))
        self.beta = np.random.uniform(low=self.config['beta_min'], high=self.config['beta_max'], size=(nu,))
        # PI gains for inner control loop [NOT READY]   
        self.gain_P = self.config['Kp_low']*np.eye(nu)      
        self.gain_I = self.config['Ki_low']*np.eye(nu)
        # self.gain_D = self.config['Kd_low']*np.eye(nu)
        self.err_I = np.zeros(nu)
        # Delays
        self.delay_sim_cycle = int(self.config['delay_sim_cycle'])       # in simu cycles
        self.buffer_sim   = []                                           # buffer for measured torque delayed by e.g. actuation and/or sensing 
        # Noise
        self.var_u = np.asarray(self.config['var_u'])
        # Actuation model options
        self.DELAY_SIM         = config['DELAY_SIM']                     # Add delay in reference torques (low-level)
        self.SCALE_TORQUES     = config['SCALE_TORQUES']                 # Affinescaling of reference torque
        self.FILTER_TORQUES    = config['FILTER_TORQUES']                # Moving average smoothing of reference torques
        self.NOISE_TORQUES     = config['NOISE_TORQUES']                # Moving average smoothing of reference torques
        self.TORQUE_TRACKING   = config['TORQUE_TRACKING']                # NOT READY
        logger.info("Created ActuationModel(DELAY_SIM="+str(self.DELAY_SIM)+
                    ", SCALE_TORQUES="+str(self.SCALE_TORQUES)+
                    ", FILTER_TORQUES="+str(self.FILTER_TORQUES)+
                    ", NOISE_TORQUES="+str(self.NOISE_TORQUES)+").")
        if(self.SCALE_TORQUES):
          logger.info("Torques scaling : alpha = "+str(self.alpha)+" | beta = "+str(self.beta))

    def step(self, i, reference_torque, memory=None):
        '''
        Transforms reference torque into measured torque
        Simulates (optionally) 
         - affine bias on torques a*tau_ref(i) + b
         - moving avg filter on tau_ref(i)
         - delay tau_ref(i) = tau_ref(i-delay)
         - Gaussian noise on tau_ref(i)
         - torque PI control (tau_mea, tau_ref)
        '''
        measured_torque = reference_torque.copy()
        # Affine scaling
        if(self.SCALE_TORQUES and len(measured_torque) !=0):
          measured_torque = self.alpha * measured_torque + self.beta
        # Filtering (moving average)
        if(self.FILTER_TORQUES and len(memory)>0):
          n_sum = min(i, self.config['u_avg_filter_length'])
          for k in range(n_sum):
            measured_torque += memory[i-k-1, :]
          measured_torque = measured_torque / (n_sum + 1)
        # Delay application of torque 
        if(self.DELAY_SIM):
          self.buffer_sim.append(measured_torque)            
          if(len(self.buffer_sim)<self.delay_sim_cycle):    
            pass
          else:                          
            measured_torque = self.buffer_sim.pop(-self.delay_sim_cycle)
        # Optional Gaussian noise on desired torque 
        if(self.NOISE_TORQUES and len(measured_torque) !=0):
            noise_u = np.random.normal(0., self.var_u, self.nu)
            measured_torque += noise_u
        # Inner PID torque control loop [NOT READY]
        if(self.TORQUE_TRACKING and len(measured_torque) !=0):
            self.err_P = measured_torque - reference_torque              
            self.err_I += measured_torque    
            # self.err_D = (measured_torque - memory[-1, :])/5e-3                         
            measured_torque = reference_torque - self.gain_P.dot(self.err_P) - self.gain_I.dot(self.err_I) #- self.gain_D.dot(self.err_D)
        return measured_torque



class CommunicationModel:

    def __init__(self, config):
        '''
        Communication model with parameters defined in config YAML file
        Simulates (optionally)
         - delay in OCP solution (x*,u*)
        '''
        self.config = config
        # Delay OCP computation
        self.x_buffer_OCP = []                                           # buffer for desired states delayed by OCP computation time
        self.u_buffer_OCP = []                                           # buffer for desired controls delayed by OCP computation time
        # Sensing model options
        self.DELAY_OCP         = config['DELAY_OCP']                     # Add delay in OCP solution (i.e. ~1ms resolution time)
        logger.info("Created CommunicationModel(DELAY_OCP="+str(self.DELAY_OCP)+").")

    def step(self, predicted_state, current_control):
        '''
        Delays input predicted state and current control by 
        using a buffer. Returns the delayed input variables
        Simulates (optionally)
         - delay in OCP solution (x*,u*)
        '''
        # Delay OCP solution due to computation time
        if(self.DELAY_OCP):
          delay = int(self.config['delay_OCP_ms'] * 1e-3 * self.config['plan_freq']) # in planning cycles
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



class SensorModel:

    def __init__(self, config, naug=0, SEED=1):
        '''
        Sensing model with parameters defined in config YAML file
        Simulates (optionally)
         - gaussian noise on measured state
         - moving avg filtering on measured state
         naug : for augmented state (lpf, soft contact, etc..). 0 by default
        '''
        np.random.seed(SEED)
        self.config = config
        self.nq = len(config['q0']) 
        self.nv = len(config['dq0'])
        self.naug = naug
        # White noise on desired torque and measured state
        self.var_q = np.asarray(self.config['var_q'])
        self.var_v = np.asarray(self.config['var_v'])
        if(self.naug > 0):
          self.var_aug = np.asarray(self.config['var_aug'])[:naug]
        # Sensing model options
        self.NOISE_STATE       = config['NOISE_STATE']                   # Add Gaussian noise on the measured state 
        self.FILTER_STATE      = config['FILTER_STATE']                  # Moving average smoothing of reference torques
        logger.info("Created SensorModel(NOISE_STATE="+str(self.NOISE_STATE)+", FILTER_STATE="+str(self.FILTER_STATE)+").")

    def step(self, i, measured_state, memory):
        '''
        Transforms simulator state into a measured state
        Simulates (optionally)
         - gaussian noise on measured state
         - moving avg filtering on measured state
        '''
        # Optional Gaussian noise on measured state 
        if(self.NOISE_STATE):
          noise_q = np.random.normal(0., self.var_q, self.nq)
          noise_v = np.random.normal(0., self.var_v, self.nv)
          if(self.naug > 0):
            noise_tau = np.random.normal(0., self.var_aug, self.naug)
            measured_state += np.concatenate([noise_q, noise_v, noise_tau]).T
          else:
            measured_state += np.concatenate([noise_q, noise_v]).T
        # Optional filtering on measured state
        if(self.FILTER_STATE and len(memory)>0):
          n_sum = min(i, self.config['x_avg_filter_length'])
          for k in range(n_sum):
            measured_state += memory[i-k-1, :]
          measured_state = measured_state / (n_sum + 1)
        return measured_state


class VelocityEstimator:
  def __init__(self, dt): #, order=1):
    self.dt = dt
    self.q_prev = None
    # self.q_prev2 
    # self.order = order
    # self.nv = len(self.q_prev)
  
  def FD1_estimate(self, q):
    if(self.q_prev is None):
      v = np.zeros(len(q))
    else:
      v = (q - self.q_prev)/self.dt
    self.q_prev = q
    return v

    # else:
  #   return (q - self.q_prev)/self.dt

  # def FD2_estimate(self, q, dt=self.dt):
  #   return q - q_prev

  #   # Moving Average Filter
  # def moving_average_filter(input_data, filter_size=1):
  #     '''
  #     moving average on 1st dimension of some array ((N,n))
  #     '''
  #     output_data = input_data.copy() 
  #     # Filter them with moving average
  #     for i in range( output_data.shape[0] ):
  #         n_sum = min(i, filter_size)
  #         # Sum up over window
  #         for k in range(n_sum):
  #             output_data[i,:] += input_data[i-k-1, :]
  #         #  Divide by number of samples
  #         output_data[i,:] = output_data[i,:] / (n_sum + 1)
  #     return output_data 