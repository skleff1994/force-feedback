"""
@package force_feedback
@file iiwa_mpc_lpf_sim.py
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
config = path_utils.load_config_file('static_reaching_task_lpf_mpc')
# Create a Pybullet simulation environment + set simu freq
dt_simu = 1./float(config['simu_freq'])  
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
pybullet_simulator = sim_utils.init_kuka_simulator(dt=dt_simu, x0=x0)
# Get pin wrapper
robot = pybullet_simulator.pin_robot
# Get dimensions 
nq, nv = robot.model.nq, robot.model.nv; ny = nq+nv+nq; nu = nq
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")


#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
ug = pin_utils.get_u_grav(q0, robot)
y0 = np.concatenate([x0, ug])


LPF_TYPE = 1
# Approx. LPF obtained from Z.O.H. discretization on CT LPF 
if(LPF_TYPE==0):
    alpha = np.exp(-2*np.pi*config['f_c']*dt)
# Approx. LPF obtained from 1st order Euler int. on CT LPF
if(LPF_TYPE==1):
    alpha = 1./float(1+2*np.pi*config['f_c']*dt)
# Exact LPF obtained from E.M.A model (IIR)
if(LPF_TYPE==2):
    y = np.cos(2*np.pi*config['f_c']*dt)
    alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 
print("--------------------------------------")
print("              INIT OCP                ")
print("--------------------------------------")
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=False, 
                                                cost_w_reg=0., 
                                                cost_w_lim=1.,
                                                tau_plus=True, 
                                                lpf_type=LPF_TYPE,
                                                WHICH_COSTS=config['WHICH_COSTS'] ) 

WEIGHT_PROFILE = False
SOLVE_AND_PLOT_INIT = False

if(WEIGHT_PROFILE):
  #  Schedule weights for target reaching
  for k,m in enumerate(ddp.problem.runningModels):
      m.differential.costs.costs['placement'].weight = 10. + ocp_utils.cost_weight_tanh(k, N_h, max_weight=10., alpha=5., alpha_cut=0.65)
      m.differential.costs.costs['stateReg'].weight = ocp_utils.cost_weight_parabolic(k, N_h, min_weight=0.01, max_weight=config['xRegWeight'])
      m.differential.costs.costs['ctrlReg'].weight  = 10./(k**2+1)
      # print("IAM["+str(k)+"].ee = "+str(m.differential.costs.costs['placement'].weight)+
      # " | IAM["+str(k)+"].xReg = "+str(m.differential.costs.costs['stateReg'].weight))


if(SOLVE_AND_PLOT_INIT):
  xs_init = [y0 for i in range(N_h+1)]
  us_init = [ug for i in range(N_h)]# ddp.problem.quasiStatic(xs_init[:-1])
  ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
  ddp_data = data_utils.extract_ddp_data_LPF(ddp)
  fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, markers=['.'], SHOW=True)

# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
sim_data = data_utils.init_sim_data_LPF(config, robot, y0)
  # Get frequencies
freq_PLAN = sim_data['plan_freq']
freq_CTRL = sim_data['ctrl_freq']
freq_SIMU = sim_data['simu_freq']
  # Initialize PID errors and control gains
err_u_P = np.zeros(nq)
err_u_I = np.zeros(nq)
err_u_D = np.zeros(nq)
  # Replan & control counters
nb_plan = 0
nb_ctrl = 0
  # Buffers for delays
y_buffer_OCP = []                                             # buffer for desired controls delayed by OCP computation time
w_buffer_OCP = []                                             # buffer for desired states delayed by OCP computation time
buffer_sim = []                                               # buffer for measured torque delayed by e.g. actuation and/or sensing 
  # Sim options
WHICH_PLOTS = ['y','w', 'p']                                       # Which plots to generate ? ('y':state, 'w':control, 'p':end-eff, etc.)
TORQUE_TRACKING = config['TORQUE_TRACKING']                   # Activate low-level reference torque tracking (PID) 
DELAY_SIM = config['DELAY_SIM']                               # Add delay in reference torques (low-level)
DELAY_OCP = config['DELAY_OCP']                               # Add delay in OCP solution (i.e. ~1ms resolution time)
SCALE_TORQUES = config['SCALE_TORQUES']                       # Affine scaling of reference torque
NOISE_TORQUES = config['NOISE_TORQUES']                       # Add Gaussian noise on reference torques
FILTER_TORQUES = config['FILTER_TORQUES']                     # Moving average smoothing of reference torques
NOISE_STATE = config['NOISE_STATE']                           # Add Gaussian noise on the measured state 
FILTER_STATE = config['FILTER_STATE']                         # Moving average smoothing of reference torques
INTERPOLATE_PLAN_TO_CTRL = config['INTERPOLATE_PLAN_TO_CTRL'] # Interpolate MPC prediction to control frequency
INTERPOLATE_CTRL_TO_SIMU = config['INTERPOLATE_CTRL_TO_SIMU'] # Interpolate motor driver reference torque to low-level frequency 
# INTERPOLATE_PLAN_TO_SIMU = config['INTERPOLATE_PLAN_TO_SIMU'] # Interpolate MPC prediction to low-level frequency 
#                                                               # !!! automatically sets to True *_PLAN_TO_CTRL and *_CTRL_TO_SIMU
dt_ocp = dt                               # OCP sampling rate 
dt_mpc = float(1./sim_data['plan_freq'])  # planning rate
dt_ctr = float(1./sim_data['ctrl_freq'])  # control rate 
dt_sim = float(1./sim_data['simu_freq'])  # sampling rate 
OCP_TO_PLAN_RATIO = dt_mpc / dt_ocp
OCP_TO_CTRL_RATIO = dt_ctr / dt_ocp
OCP_TO_SIMU_RATIO = dt_sim / dt_ocp

print("Scaling OCP-->PLAN : ", OCP_TO_PLAN_RATIO)
print("Scaling OCP-->CTRL : ", OCP_TO_CTRL_RATIO)
print("Scaling OCP-->SIMU : ", OCP_TO_SIMU_RATIO)


buffer_Y_PLAN_TO_CTRL = []
buffer_W_PLAN_TO_CTRL = []

# # # # # # # # # # # #
### SIMULATION LOOP ###
# # # # # # # # # # # #
if(config['INIT_LOG']):
  print('                  ***********************')
  print('                  * Simulation is ready *') 
  print('                  ***********************')        
  print("-------------------------------------------------------------------")
  print('- Total simulation duration            : T_tot  = '+str(sim_data['T_tot'])+' s')
  print('- Simulation frequency                 : f_simu = '+str(float(freq_SIMU/1000.))+' kHz')
  print('- Control frequency                    : f_ctrl = '+str(float(freq_CTRL/1000.))+' kHz')
  print('- Replanning frequency                 : f_plan = '+str(float(freq_PLAN/1000.))+' kHz')
  print('- Total # of simulation steps          : N_ctrl = '+str(sim_data['N_simu']))
  print('- Total # of control steps             : N_ctrl = '+str(sim_data['N_ctrl']))
  print('- Total # of planning steps            : N_plan = '+str(sim_data['N_plan']))
  print('- Duration of MPC horizon              : T_ocp  = '+str(sim_data['T_h'])+' s')
  print('- OCP integration step                 : dt     = '+str(config['dt'])+' s')
  print("-------------------------------------------------------------------")
  print('- Simulate low-level torque control?   : TORQUE_TRACKING          = '+str(TORQUE_TRACKING))
  if(TORQUE_TRACKING):
    print('    - PID gains = \n'
        +'      Kp ='+str(sim_data['gain_P'])+'\n'
        +'      Ki ='+str(sim_data['gain_I'])+'\n'
        +'      Kd ='+str(sim_data['gain_D'])+'\n')
  print('- Simulate delay in low-level torque?  : DELAY_SIM                = '+str(DELAY_SIM)+' ('+str(sim_data['delay_sim_cycle'])+' cycles)')
  print('- Simulate delay in OCP solution?      : DELAY_OCP                = '+str(DELAY_OCP)+' ('+str(config['delay_OCP_ms'])+' ms)')
  print('- Affine scaling of ref. ctrl torque?  : SCALE_TORQUES            = '+str(SCALE_TORQUES))
  if(SCALE_TORQUES):
    print('    a='+str(sim_data['alpha'])+'\n')
    print('    b='+str(sim_data['beta'])+')')
  print('- Noise on torques?                    : NOISE_TORQUES            = '+str(NOISE_TORQUES))
  print('- Filter torques?                      : FILTER_TORQUES           = '+str(FILTER_TORQUES))
  print('- Noise on state?                      : NOISE_STATE              = '+str(NOISE_STATE))
  print('- Filter state?                        : FILTER_STATE             = '+str(FILTER_STATE))
  print('- Interpolate planned torque?          : INTERPOLATE_PLAN_TO_CTRL = '+str(INTERPOLATE_PLAN_TO_CTRL))
  print('- Interpolate control torque?          : INTERPOLATE_CTRL_TO_SIMU = '+str(INTERPOLATE_CTRL_TO_SIMU))
  print("-------------------------------------------------------------------")
  print("Simulation will start...")
  time.sleep(config['init_log_display_time'])


#      y_0         y_1         y_2                     --> pred(MPC=O) size N_h
# OCP : O           O           O                           ref_O = y_1
# MPC : M     M     M     M     M                           ref_M = y_0 + Interp_[O->M] (y_1 - y_0)
# CTR : C  C  C  C  C  C  C  C  C                           ref_C = y_0 + Interp_[O->C] (y_1 - y_0)
# SIM : SSSSSSSSSSSSSSSSSSSSSSSSS                           ref_S = y_0 + Interp_[O->S] (y_1 - y_0)
#
#            y_0         y_1         y_2               --> pred(MPC=1) size N_h
#             O           O           O 
#             M     M     M     M     M
#             C  C  C  C  C  C  C  C  C
#             SSSSSSSSSSSSSSSSSSSSSSSSS  
#
#                        y_0         y_1         y_2   --> pred(MPC=2) size N_h
#                         O           O           O 
#                         M     M     M     M     M
#                         C  C  C  C  C  C  C  C  C
#                         SSSSSSSSSSSSSSSSSSSSSSSSS  


# SIMULATE
for i in range(sim_data['N_simu']): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print("  ")
      print("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))

  # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        
        print("PLAN ("+str(nb_plan)+"/"+str(sim_data['N_plan'])+")")

        # reg_u_factor = 1e4/float(i+1) # decreases throughout simulation 
        # reg_x_factor = 1e4/float(i+1) # decreases throughout simulation 
        # coef = float(i+1)/sim_data['N_simu']
        # for k,m in enumerate(ddp.problem.runningModels):
        #   # m.differential.costs.costs['placement'].weight = ocp_utils.cost_weight_tanh(i, sim_data['N_simu'], max_weight=config['frameWeight'], alpha=1., alpha_cut=0.65)
        #   # m.differential.costs.costs['stateReg'].weight = max(10/float(i+1), config['xRegWeight'])
        #   m.differential.costs.costs['ctrlReg'].weight = max(1e3/float(i+1), config['uRegWeight'])

        # if(i%1000==0):
          # print("Placement = ", ddp.problem.runningModels[0].differential.costs.costs['placement'].weight )
          # print("stateReg  = ", ddp.problem.runningModels[0].differential.costs.costs['stateReg'].weight )
          # print("ctrlReg decreasing from ", ddp.problem.runningModels[0].differential.costs.costs['ctrlReg'].weight, " to ", ddp.problem.runningModels[-1].differential.costs.costs['ctrlReg'].weight )

        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = sim_data['Y_mea_SIMU'][i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['Y_mea_SIMU'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        sim_data['Y_pred'][nb_plan, :, :] = np.array(ddp.xs)
        sim_data['W_pred'][nb_plan, :, :] = np.array(ddp.us)
        # Extract relevant predictions for interpolations
        y_curr = sim_data['Y_pred'][nb_plan, 0, :]  # y0* = measured state    (q^,  v^ , tau^ )
        y_pred = sim_data['Y_pred'][nb_plan, 1, :]  # y1* = predicted state   (q1*, v1*, tau1*) 
        w_curr = sim_data['W_pred'][nb_plan, 0, :]  # w0* = optimal control   (w0*) !! UNFILTERED TORQUE !!
        # Initialize control prediction
        if(nb_plan==0):
          w_pred_prev = w_curr
        else:
          w_pred_prev = sim_data['W_pred'][nb_plan-1, 1, :]

        # Optionally delay due to OCP resolution time 
        if(DELAY_OCP):
          y_buffer_OCP.append(y_pred)
          w_buffer_OCP.append(w_curr)
          if(len(y_buffer_OCP)<sim_data['delay_OCP_cycle']): 
            pass
          else:                            
            y_pred = y_buffer_OCP.pop(-sim_data['delay_OCP_cycle'])
          if(len(w_buffer_OCP)<sim_data['delay_OCP_cycle']): 
            pass
          else:
            w_curr = w_buffer_OCP.pop(-sim_data['delay_OCP_cycle'])

        # Select reference control and state for the current PLAN cycle
        y_ref_PLAN          = y_curr      + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
        w_ref_PLAN          = w_pred_prev + OCP_TO_PLAN_RATIO * (w_curr - w_pred_prev)
        if(nb_plan==0):
          sim_data['Y_des_PLAN'][nb_plan, :] = y_curr  
        sim_data['W_des_PLAN'][nb_plan, :]   = w_ref_PLAN   
        sim_data['Y_des_PLAN'][nb_plan+1, :] = y_ref_PLAN    

        # Increment planning counter
        nb_plan += 1



  # If we are in a control cycle select reference torque to send to the actuator
    if(i%int(freq_SIMU/freq_CTRL) == 0):        
        # print("  CTRL ("+str(nb_ctrl)+"/"+str(sim_data['N_ctrl'])+")")

        # Select reference control and state for the current CTRL cycle
        COEF                = float(i%int(freq_CTRL/freq_PLAN)) / float(freq_CTRL/freq_PLAN)
        y_ref_CTRL          = y_curr      + COEF * OCP_TO_CTRL_RATIO * (y_pred - y_curr)
        w_ref_CTRL          = w_pred_prev + COEF * OCP_TO_CTRL_RATIO * (w_curr - w_pred_prev)
        # First prediction = measurement = initialization of MPC
        if(nb_ctrl==0):
          sim_data['Y_des_CTRL'][nb_ctrl, :] = y_curr  
        sim_data['W_des_CTRL'][nb_ctrl, :]   = w_ref_CTRL  
        sim_data['Y_des_CTRL'][nb_ctrl+1, :] = y_ref_CTRL   

        # Increment control counter
        nb_ctrl += 1
        

  # Simulate actuation with PI torque tracking controller (low-level control frequency)

    # Select reference control and state for the current SIMU cycle
    COEF = float(i%int(freq_SIMU/freq_PLAN)) / float(freq_SIMU/freq_PLAN)
    y_ref_SIMU         = y_curr      + COEF * OCP_TO_SIMU_RATIO * (y_pred - y_curr)
    w_ref_SIMU         = w_pred_prev + COEF * OCP_TO_SIMU_RATIO * (w_curr - w_pred_prev)
    # First prediction = measurement = initialization of MPC
    if(i==0):
      sim_data['Y_des_SIMU'][i, :] = y_curr  
    sim_data['W_des_SIMU'][i, :]   = w_ref_SIMU  
    sim_data['Y_des_SIMU'][i+1, :] = y_ref_SIMU 

    # Torque applied by motor on actuator : interpolate current torque and predicted torque 
    tau_ref_SIMU = y_ref_SIMU[-nu:] # y_curr[-nu:]+ COEF * (y_pred[-nu:] - y_curr[-nu:]) # (dt_sim/dt_mpc) *

    # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU )    
    tau_mea_SIMU = tau_ref_SIMU 

    if(TORQUE_TRACKING):
      tau_mea_SIMU = tau_ref_SIMU - sim_data['gain_P'].dot(err_u_P) - sim_data['gain_I'].dot(err_u_I) - sim_data['gain_D'].dot(err_u_D)
    else:
      tau_mea_SIMU = tau_ref_SIMU 
    if(SCALE_TORQUES):
      tau_mea_SIMU = sim_data['alpha']*tau_mea_SIMU + sim_data['beta']
    if(FILTER_TORQUES):
      n_sum = min(i, config['u_avg_filter_length'])
      for k in range(n_sum):
        tau_mea_SIMU += sim_data['Y_mea_SIMU'][i-k-1, -nu:]
      tau_mea_SIMU = tau_mea_SIMU / (n_sum + 1)
    if(DELAY_SIM):
      buffer_sim.append(tau_mea_SIMU)            
      if(len(buffer_sim)<sim_data['delay_sim_cycle']):    
        pass
      else:                          
        tau_mea_SIMU = buffer_sim.pop(-sim_data['delay_sim_cycle'])

    # # Actuation model = LPF on interpolated values?
    # alpha = float(1./(1+2*np.pi*5e-5*config['f_c']))
    # tau_mea_SIMU = alpha*tau_mea_SIMU + (1-alpha)*w_curr # in fact u_des as long as old actuation model is desactivated
    
    # Send output of actuation torque to the RBD simulator 
    pybullet_simulator.send_joint_command(tau_mea_SIMU)  
    p.stepSimulation()
    # Measure new state from simulation :
    q_mea_SIMU, v_mea_SIMU = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
    # Record data (unnoised)
    y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU]).T 
    sim_data['Y_mea_no_noise_SIMU'][i+1, :] = y_mea_SIMU

    # Optional noise + filtering
    if(NOISE_STATE): # and float(i)/freq_SIMU <= time_stop_noise):
      noise_q = np.random.normal(0., sim_data['var_q'], nq)
      noise_v = np.random.normal(0., sim_data['var_v'], nv)
      noise_tau = np.random.normal(0., sim_data['var_u'], nu)
      y_mea_SIMU += np.concatenate([noise_q, noise_v, noise_tau]).T
    if(FILTER_STATE):
      n_sum = min(i, config['x_avg_filter_length'])
      for k in range(n_sum):
        y_mea_SIMU += sim_data['Y_mea_SIMU'][i-k-1, :]
      y_mea_SIMU = y_mea_SIMU / (n_sum + 1)
    # Record noised data
    sim_data['Y_mea_SIMU'][i+1, :] = y_mea_SIMU 
    # # Estimate torque time-derivative
    # if(i>=1):
    #   sim_data['dY_mea_SIMU'][i, :] = (y_mea_SIMU - sim_data['dY_mea_SIMU'][i-1, :]) / (dt_simu)
      # vel_u_mea = (Tau_mea[i-4, :] - 8*Tau_mea[i-3, :] + Tau_mea[i-1, :] - Tau_mea[i, :]) / (12*dt_simu)
    # Update PID errors
    if(TORQUE_TRACKING):
      err_u_P = sim_data['Y_mea_SIMU'][i, -nu:] - tau_ref_SIMU              
      err_u_I += err_u_P                             
      err_u_D = sim_data['dY_mea_SIMU'][i, :] #- vel_u_ref_HF #vel_u_ref_HF # vs vel_u_ref  

print('--------------------------------')
print('Simulation exited successfully !')
print('--------------------------------')

# # # # # # # # # # # #
# PROCESS SIM RESULTS #
# # # # # # # # # # # #
# Post-process EE trajectories + record in sim data
# print('Post-processing end-effector trajectories...')
# id_endeff = robot.model.getFrameId('contact')
# sim_data['P_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, 3))
# for node_id in range(config['N_h']+1):
#   sim_data['P_pred'][:, node_id, :] = pin_utils.get_p(sim_data['Y_pred'][:, node_id, :nq], robot, id_endeff) - np.array([sim_data['p_ref']]*sim_data['N_plan'])
# sim_data['P_mea_SIMU'] = pin_utils.get_p(sim_data['Y_mea_SIMU'][:,:nq], robot, id_endeff)
# q_des = np.vstack([y0[:nq], sim_data['Y_pred'][:,1,:nq]])
# sim_data['P_des_PLAN'] = pin_utils.get_p(q_des, robot, id_endeff)
# sim_data['P_mea_no_noise_SIMU'] = pin_utils.get_p(sim_data['Y_mea_no_noise_SIMU'][:,:nq], robot, id_endeff)

# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = 'tracking='+str(TORQUE_TRACKING)+'_'+str(freq_PLAN)+'Hz_LPF'
# Extract plot data from sim data
plot_data = data_utils.extract_plot_data_from_sim_data_LPF(sim_data)
# Plot results
plot_utils.plot_mpc_results_LPF(plot_data, which_plots=WHICH_PLOTS,
                                PLOT_PREDICTIONS=True, 
                                pred_plot_sampling=int(freq_PLAN/10),
                                SAVE=True,
                                SAVE_DIR=save_dir,
                                SAVE_NAME=save_name,
                                AUTOSCALE=True)
# Save optionally
if(config['SAVE_DATA']):
  data_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)