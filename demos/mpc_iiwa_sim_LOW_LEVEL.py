# Number of runs
"""
@package force_feedback
@file mpc_iiwa_sim_LOW_LEVEL.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Closed-loop MPC for force task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl in closed-loop MPC (feedback from state x=(q,v))
Using PyBullet simulator for rigid-body dynamics 
Using PyBullet GUI for visualization

The goal of this script is to simulate the low-level torque control
as well at higher frequency (5 to 20kHz) . In face of noise we should 
still recover the performance of closed-loop MPC (ICRA 2021) because 
the KUKA had a low-level torque control
'''

import numpy as np  
from utils import path_utils, sim_utils, plot_utils, ocp_utils, data_utils, pin_utils
import pybullet as p
import time 

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task3')
# Create a Pybullet simulation environment + set simu freq
simu_freq = config['simu_freq']  
dt_simu = 1./simu_freq
q0 = np.asarray(config['q0'])
dq0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, dq0])   
pybullet_simulator = sim_utils.init_kuka_simulator(dt=dt_simu, x0=x0)
# Get pin wrapper
robot = pybullet_simulator.pin_robot
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")

# # # # # # # # #
### SETUP OCP ### 
# # # # # # # # #
ddp = ocp_utils.init_DDP(robot, config, x0)

# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
sim_data = {}
# MPC & simulation parameters
plan_freq = config['plan_freq']                         # MPC planning frequency (Hz)
ctrl_freq = config['ctrl_freq']                         # Control - simulation - frequency (Hz)
sim_data['T_tot'] = config['T_tot']                     # Total duration of simulation (s)
sim_data['N_plan'] = int(sim_data['T_tot']*plan_freq)   # Total number of planning steps in the simulation
sim_data['N_ctrl'] = int(sim_data['T_tot']*ctrl_freq)   # Total number of control steps in the simulation 
sim_data['N_simu'] = int(sim_data['T_tot']*simu_freq)   # Total number of simulation steps 
sim_data['T_h'] = config['N_h']*config['dt']            # Duration of the MPC horizon (s)
sim_data['N_h'] = config['N_h']                         # Number of nodes in MPC horizon
sim_data['dt_ctrl'] = float(1./ctrl_freq)               # Duration of 1 control cycle (s)
sim_data['dt_plan'] = float(1./plan_freq)               # Duration of 1 planning cycle (s)
sim_data['dt_simu'] = dt_simu                           # Duration of 1 simulation cycle (s)
# Misc params
sim_data['nq'] = nq
sim_data['nv'] = nv
sim_data['nx'] = nx
sim_data['p_ref'] = M_ee.translation.copy()
# Main data to record 
sim_data['X_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, nx))     # Predicted states (output of DDP, i.e. ddp.xs)
sim_data['U_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], nu))       # Predicted torques (output of DDP, i.e. ddp.us)
sim_data['U_ref'] = np.zeros((sim_data['N_ctrl'], nu))             # Reference torque for motor drivers (i.e. ddp.us[0] interpolated to control frequency)
sim_data['U_mea'] = np.zeros((sim_data['N_simu'], nu))             # Actuation torques (i.e. disturbed reference sent to PyBullet at simu/HF)
sim_data['X_mea'] = np.zeros((sim_data['N_simu']+1, nx))           # Measured states (i.e. measured from PyBullet at simu/HF)
sim_data['X_mea_no_noise'] = np.zeros((sim_data['N_simu']+1, nx))  # Measured states (i.e measured from PyBullet at simu/HF) without noise
vel_U_ref = np.zeros((sim_data['N_ctrl'], nu))                     # Desired torques (current ff output by DDP)
vel_U_mea = np.zeros((sim_data['N_simu'], nu))                     # Actuation torques (sent to PyBullet)
vel_U_ref_HF = np.zeros((sim_data['N_simu'], nu))                  # Actuation torques (sent to PyBullet)
vel_U_mea[0,:] = np.zeros(nq)
  # Initialize PID errors
err_u = np.zeros(nq)
vel_err_u = np.zeros(nq)
int_err_u = np.zeros(nq)
  # Initialize average acceleration tracking error (avg over 1ms)
sim_data['A_err'] = np.zeros((sim_data['N_ctrl'], nx))
  # Initialize measured state 
sim_data['X_mea'][0, :] = x0
sim_data['X_mea_no_noise'][0, :] = x0
p0 = robot.data.oMf[id_endeff].translation.T.copy()
  # Replan & control counters
nb_plan = 0
nb_ctrl = 0
# Low-level simulation parameters (actuation model)
  # Scaling of desired torque
alpha = np.random.uniform(low=config['alpha_min'], high=config['alpha_max'], size=(nq,))
beta = np.random.uniform(low=config['beta_min'], high=config['beta_max'], size=(nq,))
sim_data['alpha'] = alpha
sim_data['beta'] = beta
  # White noise on desired torque and measured state
var_u = 0.001*(2*np.asarray(config['u_lim'])) #u_np.asarray(config['var_u']) 0.5% of range on the joint
var_q = np.asarray(config['var_q'])
var_v = np.asarray(config['var_v'])
  # Buffers for delays
delay_OCP_ms = config['delay_OCP_ms']                   # in ms
delay_OCP_cycle = int(delay_OCP_ms * 1e-3 * plan_freq)  # in planning cycles
delay_sim_cycle = int(config['delay_sim_cycle'])        # in simu cycles
buffer_OCP = []                                         # buffer for desired torques
buffer_sim = []                                         # buffer for measured torque
  # Proportional-integral torque control gains
Kp = config['Kp']*np.eye(nq)
Ki = config['Ki']*np.eye(nq)
Kd = config['Kd']*np.eye(nq)
  # Moving avg filter
u_avg_filter_length = config['u_avg_filter_length']    # in HF cycles
x_avg_filter_length = config['x_avg_filter_length']    # in HF cycles
  # Sim options
WHICH_PLOTS = ['x','u','p']
TORQUE_TRACKING = config['TORQUE_TRACKING']       # Activate low-level reference torque tracking (PID) 
DELAY_SIM = config['DELAY_SIM']                   # Add delay in reference torques (low-level)
DELAY_OCP = config['DELAY_OCP']                   # Add delay in OCP solution (i.e. ~1ms resolution time)
SCALE_TORQUES = config['SCALE_TORQUES']           # Affine scaling of reference torque
NOISE_TORQUES = config['NOISE_TORQUES']           # Add Gaussian noise on reference torques
FILTER_TORQUES = config['FILTER_TORQUES']         # Moving average smoothing of reference torques
NOISE_STATE = config['NOISE_STATE']               # Add Gaussian noise on the measured state 
FILTER_STATE = config['FILTER_STATE']             # Moving average smoothing of reference torques
INTERPOLATE_PLAN = config['INTERPOLATE_PLAN']     # Interpolate DDP desired feedforward torque to control frequency
INTERPOLATE_CTRL = config['INTERPOLATE_CTRL']     # Interpolate motor driver reference torque and time-derivatives to low-level frequency 

# # # # # # # # # # # #
### SIMULATION LOOP ###
# # # # # # # # # # # #
if(config['INIT_LOGS']):
  print('                  ***********************')
  print('                  * Simulation is ready *') 
  print('                  ***********************')        
  print('---------------------------------------------------------')
  print('- Total simulation duration            : T_tot  = '+str(sim_data['T_tot'])+' s')
  print('- Simulation frequency                 : f_simu = '+str(float(simu_freq/1000.))+' kHz')
  print('- Control frequency                    : f_ctrl = '+str(float(ctrl_freq/1000.))+' kHz')
  print('- Replanning frequency                 : f_plan = '+str(float(plan_freq/1000.))+' kHz')
  print('- Total # of simulation steps          : N_ctrl = '+str(sim_data['N_simu']))
  print('- Total # of control steps             : N_ctrl = '+str(sim_data['N_ctrl']))
  print('- Total # of planning steps            : N_plan = '+str(sim_data['N_plan']))
  print('- Duration of MPC horizon              : T_ocp  = '+str(sim_data['T_h'])+' s')
  print('- OCP integration step                 : dt     = '+str(config['dt'])+' s')
  print('---------------------------------------------------------')
  print('- Simulate low-level torque control?   : TORQUE_TRACKING  = '+str(TORQUE_TRACKING))
  if(TORQUE_TRACKING):
    print('    - PID gains = \n'
        +'      Kp ='+str(Kp)+'\n'
        +'      Ki ='+str(Ki)+'\n'
        +'      Kd ='+str(Kd)+'\n')
  print('- Simulate delay in low-level torque?  : DELAY_SIM        = '+str(DELAY_SIM)+' ('+str(delay_sim_cycle)+' cycles)')
  print('- Simulate delay in OCP solution?      : DELAY_OCP        = '+str(DELAY_OCP)+' ('+str(delay_OCP_ms)+' ms)')
  print('- Affine scaling of ref. ctrl torque?  : SCALE_TORQUES    = '+str(SCALE_TORQUES))
  if(SCALE_TORQUES):
    print('    a='+str(alpha)+'\n')
    print('    b='+str(beta)+')')
  print('- Noise on torques?                    : NOISE_TORQUES    = '+str(NOISE_TORQUES))
  print('- Filter torques?                      : FILTER_TORQUES   = '+str(FILTER_TORQUES))
  print('- Noise on state?                      : NOISE_STATE      = '+str(NOISE_STATE))
  print('- Filter state?                        : FILTER_STATE     = '+str(FILTER_STATE))
  print('- Interpolate planned torque?          : INTERPOLATE_PLAN = '+str(INTERPOLATE_PLAN))
  print('- Interpolate control torque?          : INTERPOLATE_CTRL = '+str(INTERPOLATE_CTRL))
  print('---------------------------------------------------------')
  print("Simulation will start...")
  time.sleep(config['log_display_time'])

# SIMULATE
log_rate = 10000
time_stop_noise = sim_data['T_tot'] #/2. # in sec

for i in range(sim_data['N_simu']): 

    if(i%log_rate==0): 
      print("  ")
      print("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))

  # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(simu_freq/plan_freq) == 0):
        # print("  PLAN ("+str(nb_plan)+"/"+str(N_plan)+")")
        # # Updtate OCP if necessary
        # for k,m in enumerate(ddp.problem.runningModels[:]):
        #     # m.differential.costs.costs["placement"].weight += (i/N_simu)*config['frameWeight']   
        #     # print(m.differential.costs.costs["placement"].weight)
        #     m.differential.costs.costs["placement"].weight = utils.cost_weight_tanh(i, N_simu, max_weight=1000, alpha=5, alpha_cut=0.1)
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = sim_data['X_mea'][i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['X_mea'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        sim_data['X_pred'][nb_plan, :, :] = np.array(ddp.xs)
        sim_data['U_pred'][nb_plan, :, :] = np.array(ddp.us)
        # Extract desired control torque + prepare interpolation to control frequency
        x_pred_1 = sim_data['X_pred'][nb_plan, 1, :]
        u_pred_0 = sim_data['U_pred'][nb_plan, 0, :]
        # Delay due to OCP resolution time 
        if(DELAY_OCP):
          buffer_OCP.append(u_pred_0)
          if(len(buffer_OCP)<delay_OCP_cycle): 
            pass
          else:                            
            u_pred_0 = buffer_OCP.pop(-delay_OCP_cycle)
        # Optionally interpolate to control frequency
        if(nb_plan >= 1 and INTERPOLATE_PLAN==True):
          u_pred_0_next = sim_data['U_pred'][nb_plan, 1, :]
        else:
          u_pred_0_next = u_pred_0 
        # Increment planning counter
        nb_plan += 1
        
  # If we are in a control cycle select reference torque to send to motors
    if(i%int(simu_freq/ctrl_freq) == 0):
        # print("  CTRL ("+str(nb_ctrl)+"/"+str(N_ctrl)+")")
        # Optionally interpolate desired torque to control frequency
        if(INTERPOLATE_PLAN):
          coef = float(i % int(ctrl_freq/plan_freq)) / (float(ctrl_freq/plan_freq))
          u_ref = (1-coef)*u_pred_0 + coef*u_pred_0_next   
        else:
          u_ref = u_pred_0
        # Record reference torque
        sim_data['U_ref'][nb_ctrl, :] = u_ref 
        # Optionally interpolate to HF
        if(nb_ctrl >= 1 and INTERPOLATE_CTRL):
          u_ref_prev = sim_data['U_ref'][nb_ctrl-1, :]
          vel_u_ref_prev = vel_U_ref[nb_ctrl-1, :]
        else:
          u_ref_prev = u_ref
          vel_u_ref_prev = np.zeros(nq)
        # Estimate reference torque time-derivative by finite-differences for low-level PID
        vel_u_ref = ( u_ref - u_ref_prev ) / sim_data['dt_ctrl']
        vel_U_ref[nb_ctrl, :] = vel_u_ref
        # vel_u_des = (U_des[nb_ctrl-4, :] - 8*U_des[nb_ctrl-3, :] + U_des[nb_ctrl-1, :] - U_des[nb_ctrl, :]) / (12*dt_ctrl)
        # Increment control counter
        nb_ctrl += 1
        
  # Simulate actuation with PI torque tracking controller (low-level control frequency)
    # Optionally interpolate reference torque to HF / let constant
    if(INTERPOLATE_CTRL):
      coef = float(i%int(simu_freq/ctrl_freq)) / float(simu_freq/ctrl_freq)
      u_ref_HF = (1-coef)*u_ref_prev + coef*u_ref  
      vel_u_ref_HF = (1-coef)*vel_u_ref_prev + coef*vel_u_ref  
    else:
      u_ref_HF = u_ref  
      vel_u_ref_HF = vel_u_ref
    vel_U_ref_HF[i,:] = vel_u_ref_HF
    # Initialize measured torque to reference torque
    if(TORQUE_TRACKING):
      u_mea = u_ref_HF - Kp.dot(err_u) - Ki.dot(int_err_u) - Kd.dot(vel_err_u)
    else:
      u_mea = u_ref_HF 
    # Actuation = scaling + noise + filtering + delay
    if(SCALE_TORQUES):
      u_mea = alpha*u_mea + beta
    if(NOISE_TORQUES):
      u_mea += np.random.normal(0., var_u)
    if(FILTER_TORQUES):
      n_sum = min(i, u_avg_filter_length)
      for k in range(n_sum):
        u_mea += sim_data['U_mea'][i-k-1, :]
      u_mea = u_mea / (n_sum + 1)
    if(DELAY_SIM):
      buffer_sim.append(u_mea)            
      if(len(buffer_sim)<delay_sim_cycle):    
        pass
      else:                          
        u_mea = buffer_sim.pop(-delay_sim_cycle)
    # Record measured torque & step simulator
    sim_data['U_mea'][i, :] = u_mea
    pybullet_simulator.send_joint_command(sim_data['U_mea'][i, :])
    p.stepSimulation()
    # Measure new state from simulation 
    q_mea, v_mea = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea, v_mea)
    # Record data (unnoised)
    x_mea = np.concatenate([q_mea, v_mea]).T 
    sim_data['X_mea_no_noise'][i+1, :] = x_mea
    # Accumulate acceleration error over the control cycle
    sim_data['A_err'][nb_ctrl-1,:] += (np.abs(x_mea - x_pred_1))/float(simu_freq/ctrl_freq)
    # Optional noise + filtering
    if(NOISE_STATE and float(i)/simu_freq <= time_stop_noise):
      wq = np.random.normal(0., var_q, nq)
      wv = np.random.normal(0., var_v, nv)
      x_mea += np.concatenate([wq, wv]).T
    if(FILTER_STATE):
      n_sum = min(i, x_avg_filter_length)
      for k in range(n_sum):
        x_mea += sim_data['X_mea'][i-k-1, :]
      x_mea = x_mea / (n_sum + 1)
    # Record noised data
    sim_data['X_mea'][i+1, :] = x_mea 
    # Estimate torque time-derivative
    if(i>=1):
      vel_U_mea[i, :] = (u_mea - sim_data['U_mea'][i-1, :]) / (dt_simu)
      # vel_u_mea = (U_mea[i-4, :] - 8*U_mea[i-3, :] + U_mea[i-1, :] - U_mea[i, :]) / (12*dt_simu)
    else:
      vel_U_mea[i, :] = np.zeros(nq)
    # Update PID errors
    if(TORQUE_TRACKING):
      err_u = sim_data['U_mea'][i, :] - u_ref_HF              
      int_err_u += err_u                             
      vel_err_u = vel_U_mea[i, :] #- vel_u_ref_HF #vel_u_ref_HF # vs vel_u_ref  

print('--------------------------------')
print('Simulation exited successfully !')
print('--------------------------------')

# # # # # # # # # # # #
# PROCESS SIM RESULTS #
# # # # # # # # # # # #
# Post-process EE trajectories + record in sim data
print('Post-processing end-effector trajectories...')
sim_data['P_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, 3))
for node_id in range(config['N_h']+1):
  sim_data['P_pred'][:, node_id, :] = pin_utils.get_p(sim_data['X_pred'][:, node_id, :nq], robot, id_endeff) - np.array([sim_data['p_ref']]*sim_data['N_plan'])
sim_data['P_mea'] = pin_utils.get_p(sim_data['X_mea'][:,:nq], robot, id_endeff)
q_des = np.vstack([sim_data['X_mea'][0,:nq], sim_data['X_pred'][:,1,:nq]])
sim_data['P_des'] = pin_utils.get_p(q_des, robot, id_endeff)
sim_data['P_mea_no_noise'] = pin_utils.get_p(sim_data['X_mea_no_noise'][:,:nq], robot, id_endeff)

# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = 'tracking='+str(TORQUE_TRACKING)+'_'+str(plan_freq)+'Hz'
# Extract plot data from sim data
plot_data = data_utils.extract_plot_data(sim_data)
# Plot results
plot_utils.plot_mpc_results(plot_data, which_plots=WHICH_PLOTS,
                              PLOT_PREDICTIONS=True, 
                              pred_plot_sampling=int(plan_freq/10),
                              SAVE=True,
                              SAVE_DIR=save_dir,
                              SAVE_NAME=save_name,
                              AUTOSCALE=True)
# Save optionally
if(config['SAVE_DATA']):
  data_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)