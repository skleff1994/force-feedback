
import numpy as np

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



class ActuationModel:

    def __init__(self, config, nu=7, SEED=1):
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
        self.gain_P = self.config['Kp']*np.eye(nu)      
        self.gain_I = self.config['Ki']*np.eye(nu)
        self.err_I = np.zeros(nu)
        # Delays
        self.delay_sim_cycle = int(self.config['delay_sim_cycle'])       # in simu cycles
        self.buffer_sim   = []                                           # buffer for measured torque delayed by e.g. actuation and/or sensing 
        # Actuation model options
        self.DELAY_SIM         = config['DELAY_SIM']                     # Add delay in reference torques (low-level)
        self.SCALE_TORQUES     = config['SCALE_TORQUES']                 # Affinescaling of reference torque
        self.FILTER_TORQUES    = config['FILTER_TORQUES']                # Moving average smoothing of reference torques
        self.TORQUE_TRACKING   = config['TORQUE_TRACKING']                # NOT READY
        logger.info("Created ActuationModel(DELAY_SIM="+str(self.DELAY_SIM)+
                    ", SCALE_TORQUES="+str(self.SCALE_TORQUES)+
                    ", FILTER_TORQUES="+str(self.FILTER_TORQUES)+").")
        if(self.SCALE_TORQUES):
          logger.info("Torques scaling : alpha = "+str(self.alpha)+" | beta = "+str(self.beta))

    def step(self, i, reference_torque, memory):
        '''
        Transforms reference torque into measured torque
        Simulates (optionally) 
         - affine bias on torques a*tau_ref(i) + b
         - moving avg filter on tau_ref(i)
         - delay tau_ref(i) = tau_ref(i-delay)
         - torque PI control (tau_mea, tau_ref)
        '''
        measured_torque = reference_torque.copy()
        # Affine scaling
        if(self.SCALE_TORQUES):
          measured_torque = self.alpha * measured_torque + self.beta
        # Filtering (moving average)
        if(self.FILTER_TORQUES):
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
        # Inner PID torque control loop [NOT READY]
        if(self.TORQUE_TRACKING):
            self.err_P = measured_torque - reference_torque              
            self.err_I += measured_torque                             
            measured_torque = reference_torque - self.gain_P.dot(self.err_P) - self.gain_I.dot(self.err_I)
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

    def __init__(self, config, nq=7, nv=7, ntau=0, SEED=1):
        '''
        Sensing model with parameters defined in config YAML file
        Simulates (optionally)
         - gaussian noise on measured state
         - moving avg filtering on measured state
        '''
        np.random.seed(SEED)
        self.config = config
        self.nq = nq
        self.nv = nv
        self.ntau = ntau
        # White noise on desired torque and measured state
        self.var_q = np.asarray(self.config['var_q'])
        self.var_v = np.asarray(self.config['var_v'])
        self.var_u = 0.5*np.asarray(self.config['var_u']) 
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
          if(self.ntau != 0):
            noise_tau = np.random.normal(0., self.var_u, self.ntau)
            measured_state += np.concatenate([noise_q, noise_v, noise_tau]).T
          else:
            measured_state += np.concatenate([noise_q, noise_v]).T
        # Optional filtering on measured state
        if(self.FILTER_STATE):
          n_sum = min(i, self.config['x_avg_filter_length'])
          for k in range(n_sum):
            measured_state += memory[i-k-1, :]
          measured_state = measured_state / (n_sum + 1)
        return measured_state
