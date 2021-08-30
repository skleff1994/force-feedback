"""
@package force_feedback
@file mpc_iiwa_sim_LPF.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop 'LPF torque feedback' MPC for static target task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateLPF y=(q,v,tau), control w = unfiltered torque)
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with torque feedback where
the actuation dynamics is modeled as a low pass filter (LPF) in the optimization.
  - The letter y denotes the augmented state of joint positions, velocities
    and filtered torques while the letter 'w' denotes the unfiltered torque 
    input to the actuation model. 
  - We optimize (y*,w*) using Crocoddyl but we send tau* to the simulator (NOT w*)
  - Simulator = custom actuation model (not LPF) + PyBullet RBD
'''

import numpy as np  
from utils import path_utils, sim_utils, ocp_utils, pin_utils, plot_utils, data_utils
import pybullet as p
import time 

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_lpf')
# Create a Pybullet simulation environment + set simu freq
simu_freq = config['simu_freq']  
dt_simu = 1./simu_freq
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
pybullet_simulator = sim_utils.init_kuka_simulator(dt=dt_simu, x0=x0)
# Get pin wrapper
robot = pybullet_simulator.pin_robot
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]
nq, nv = robot.model.nq, robot.model.nv
ny = nq+nv+nq
nu = nq
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")


#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
# u0 = np.asarray(config['tau0'])
ug = pin_utils.get_u_grav(q0, robot)
y0 = np.concatenate([x0, ug])
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, f_c=config['f_c'], cost_w=1e-4)

#  Schedule weights for target reaching
for k,m in enumerate(ddp.problem.runningModels):
    m.differential.costs.costs['placement'].weight = 10. + ocp_utils.cost_weight_tanh(k, N_h, max_weight=100., alpha=5., alpha_cut=0.65)
    m.differential.costs.costs['stateReg'].weight = ocp_utils.cost_weight_parabolic(k, N_h, min_weight=0.01, max_weight=config['xRegWeight'])
    print("IAM["+str(k)+"].ee = "+str(m.differential.costs.costs['placement'].weight)+
    " | IAM["+str(k)+"].xReg = "+str(m.differential.costs.costs['stateReg'].weight))

# Plot
# xs_init = [y0 for i in range(N_h+1)]
# us_init = [ug for i in range(N_h)]# ddp.problem.quasiStatic(xs_init[:-1])
# ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
# plot_utils.plot_ddp_results_LPF(ddp, robot, id_endeff)

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
sim_data['ny'] = ny
sim_data['p_ref'] = M_ee.translation.copy()
# Main data to record 
sim_data['Y_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, ny)) # Predicted states  (ddp.xs : {y* = (q*, v*, tau*)} )
sim_data['W_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], nu))   # Predicted torques (ddp.us : {w*} )
sim_data['Tau_ref'] = np.zeros((sim_data['N_ctrl'], nu))                 # Reference torque for motor drivers (tau* interpolated at ctrl freq)
sim_data['Tau_mea'] = np.zeros((sim_data['N_simu'], nu))                 # Actuation torques (output of actuator sent to PyBullet at simu freq)
sim_data['Y_mea'] = np.zeros((sim_data['N_simu']+1, ny))                 # Measured states (measured y = (q, v, tau) from PyBullet at simu freq)
sim_data['Y_mea_no_noise'] = np.zeros((sim_data['N_simu']+1, ny))        # Measured states (measured y = (q, v, tau) from PyBullet at simu freq) without noise
dTau_ref = np.zeros((sim_data['N_ctrl'], nu))                            # Desired torques (current ff output by DDP)
dTau_mea = np.zeros((sim_data['N_simu'], nu))                            # Actuation torques (sent to PyBullet)
dTau_ref_HF = np.zeros((sim_data['N_simu'], nu))                         # Actuation torques (sent to PyBullet)
dTau_mea[0,:] = np.zeros(nq)
  # Initialize PID errors
err_u = np.zeros(nq)
err_du = np.zeros(nq)
int_err_u = np.zeros(nq)
  # Initialize measured state 
sim_data['Y_mea'][0, :] = y0
sim_data['Y_mea_no_noise'][0, :] = y0
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
  # Proportional-integral-derivative torquecontrol gains  
Kp = config['Kp']*np.eye(nq)
Ki = config['Ki']*np.eye(nq)
Kd = config['Kd']*np.eye(nq)
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
        # print("PLAN ("+str(nb_plan)+"/"+str(N_plan)+")")
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = sim_data['Y_mea'][i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['Y_mea'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        sim_data['Y_pred'][nb_plan, :, :] = np.array(ddp.xs)
        sim_data['W_pred'][nb_plan, :, :] = np.array(ddp.us)
        # Extract predictions and prepare interpolation to control frequency
        y_pred_0 = sim_data['Y_pred'][nb_plan, 0, :] # measured  (q,v,tau)
        y_pred_1 = sim_data['Y_pred'][nb_plan, 1, :] # predicted (q,v,tau)
        w_pred_0 = sim_data['W_pred'][nb_plan, 0, :] # optimal unfiltered torque (w)
        # Delay due to OCP resolution time 
        if(DELAY_OCP):
          buffer_OCP.append(w_pred_0)
          if(len(buffer_OCP)<delay_OCP_cycle): 
            pass
          else:                            
            w_pred_0 = buffer_OCP.pop(-delay_OCP_cycle)
        # Optionally interpolate to control frequency
        if(nb_plan >= 1 and INTERPOLATE_PLAN==True):
          w_pred_1 = sim_data['W_pred'][nb_plan, 1, :]
        else:
          w_pred_1 = w_pred_0 
        # Increment planning counter
        nb_plan += 1
        
  # If we are in a control cycle select reference torque to send to motors
    if(i%int(simu_freq/ctrl_freq) == 0):
        # print("  CTRL ("+str(nb_ctrl)+"/"+str(N_ctrl)+")")
        # Optionally interpolate state and control to control frequency
        if(INTERPOLATE_PLAN):
          coef = float(i % int(ctrl_freq/plan_freq)+1) / (float(ctrl_freq/plan_freq))
          u_ref = (1-coef)*w_pred_0 + coef*w_pred_1   
          x_ref = (1-coef)*y_pred_0 + coef*y_pred_1   # or shift +1 in time?
        else:
          u_ref = w_pred_0
          x_ref = y_pred_1
        # Record reference torque
        sim_data['Tau_ref'][nb_ctrl, :] = u_ref 
        # Optionally interpolate to HF
        if(nb_ctrl >= 1 and INTERPOLATE_CTRL):
          u_ref_prev = sim_data['Tau_ref'][nb_ctrl-1, :]
          x_ref_prev = sim_data['X_ref'][nb_ctrl-1, :]
          du_ref_prev = dTau_ref[nb_ctrl-1, :]
        else:
          u_ref_prev = u_ref
          x_ref_prev = x_ref
          du_ref_prev = np.zeros(nq)
        # Estimate reference torque time-derivative by finite-differences for low-level PID
        du_ref = ( u_ref - u_ref_prev ) / sim_data['dt_ctrl']
        dTau_ref[nb_ctrl, :] = du_ref
        # vel_u_des = (U_des[nb_ctrl-4, :] - 8*U_des[nb_ctrl-3, :] + U_des[nb_ctrl-1, :] - U_des[nb_ctrl, :]) / (12*dt_ctrl)
        # Increment control counter
        nb_ctrl += 1
        
  # Simulate actuation with PI torque tracking controller (low-level control frequency)
    # Optionally interpolate reference torque to HF / let constant
    if(INTERPOLATE_CTRL):
      coef = float(i%int(simu_freq/ctrl_freq)) / float(simu_freq/ctrl_freq)
      u_ref_HF = (1-coef)*u_ref_prev + coef*u_ref  
      x_ref_HF = (1-coef)*x_ref_prev + coef*x_ref  
      du_ref_HF = (1-coef)*du_ref_prev + coef*du_ref
    else:
      u_ref_HF = u_ref 
      x_ref_HF = x_ref 
      du_ref_HF = du_ref
    dTau_ref_HF[i,:] = du_ref_HF
    # Initialize measured torque to reference torque
    if(TORQUE_TRACKING):
      u_mea = u_ref_HF - Kp.dot(err_u) - Ki.dot(int_err_u) - Kd.dot(err_du)
    else:
      u_mea = u_ref_HF 
    tau_mea = x_ref_HF[-nq:] # send interp tau*_0 --> tau*_1 to the robot
    # Actuation = scaling + noise + filtering + delay
    if(SCALE_TORQUES):
      tau_mea = alpha*tau_mea + beta
    if(NOISE_TORQUES):
      u_mea += np.random.normal(0., var_u)
    if(FILTER_TORQUES):
      n_sum = min(i, config['u_avg_filter_length'])
      for k in range(n_sum):
        u_mea += sim_data['Tau_mea'][i-k-1, :]
      u_mea = u_mea / (n_sum + 1)
    if(DELAY_SIM):
      buffer_sim.append(u_mea)            
      if(len(buffer_sim)<delay_sim_cycle):    
        pass
      else:                          
        u_mea = buffer_sim.pop(-delay_sim_cycle)
    # Record measured torque & step simulator
    sim_data['Tau_mea'][i, :] = tau_mea

    # # Actuation model = LPF on interpolated values?
    # alpha = float(1./(1+2*np.pi*5e-5*config['f_c']))
    # tau_mea = alpha*tau_des + (1-alpha)*u_mea # in fact u_des as long as old actuation model is desactivated
    # tau_mea = alpha*tau_des + beta
    
    pybullet_simulator.send_joint_command(tau_mea) #u_mea 
    p.stepSimulation()
    # Measure new state from simulation :
    q_mea, v_mea = pybullet_simulator.get_state()
    

    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea, v_mea)
    # Record data (unnoised)
    x_mea = np.concatenate([q_mea, v_mea, tau_mea]).T 
    sim_data['Y_mea_no_noise'][i+1, :] = x_mea
    # Optional noise + filtering
    if(NOISE_STATE and float(i)/simu_freq <= time_stop_noise):
      wq = np.random.normal(0., var_q, nq)
      wv = np.random.normal(0., var_v, nv)
      x_mea += np.concatenate([wq, wv, 0.]).T
    if(FILTER_STATE):
      n_sum = min(i, config['x_avg_filter_length'])
      for k in range(n_sum):
        x_mea += sim_data['Y_mea'][i-k-1, :]
      x_mea = x_mea / (n_sum + 1)
    # Record noised data
    sim_data['Y_mea'][i+1, :] = x_mea 
    # Estimate torque time-derivative
    if(i>=1):
      dTau_mea[i, :] = (u_mea - sim_data['Tau_mea'][i-1, :]) / (dt_simu)
      # vel_u_mea = (Tau_mea[i-4, :] - 8*Tau_mea[i-3, :] + Tau_mea[i-1, :] - Tau_mea[i, :]) / (12*dt_simu)
    else:
      dTau_mea[i, :] = np.zeros(nq)
    # Update PID errors
    if(TORQUE_TRACKING):
      err_u = sim_data['Tau_mea'][i, :] - u_ref_HF              
      int_err_u += err_u                             
      err_du = dTau_mea[i, :] #- vel_u_ref_HF #vel_u_ref_HF # vs vel_u_ref  

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
  sim_data['P_pred'][:, node_id, :] = pin_utils.get_p(sim_data['Y_pred'][:, node_id, :nq], robot, id_endeff) - np.array([sim_data['p_ref']]*sim_data['N_plan'])
sim_data['P_mea'] = pin_utils.get_p(sim_data['Y_mea'][:,:nq], robot, id_endeff)
q_des = np.vstack([sim_data['Y_mea'][0,:nq], sim_data['Y_pred'][:,1,:nq]])
sim_data['P_des'] = pin_utils.get_p(q_des, robot, id_endeff)
sim_data['P_mea_no_noise'] = pin_utils.get_p(sim_data['Y_mea_no_noise'][:,:nq], robot, id_endeff)

# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = 'tracking='+str(TORQUE_TRACKING)+'_'+str(plan_freq)+'Hz'
# Extract plot data from sim data
plot_data = data_utils.extract_plot_data(sim_data)
# Plot results
plot_utils.plot_mpc_results_lpf(plot_data, which_plots=WHICH_PLOTS,
                              PLOT_PREDICTIONS=True, 
                              pred_plot_sampling=int(plan_freq/20),
                              SAVE=True,
                              SAVE_DIR=save_dir,
                              SAVE_NAME=save_name,
                              AUTOSCALE=True)
# Save optionally
if(config['SAVE_DATA']):
  data_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)