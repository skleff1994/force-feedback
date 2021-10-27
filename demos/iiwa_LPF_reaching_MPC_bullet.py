"""
@package force_feedback
@file iiwa_LPF_reaching_MPC_bullet.py
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

# Fix seed 
np.random.seed(1)

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config_name = 'iiwa_LPC_reaching_MPC'
config = path_utils.load_config_file(config_name)
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


LPF_TYPE = 0
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
                                                cost_w_reg=1e-6, 
                                                cost_w_lim=10.,
                                                tau_plus=True, 
                                                lpf_type=LPF_TYPE,
                                                WHICH_COSTS=config['WHICH_COSTS']) 

WEIGHT_PROFILE = False
SOLVE_AND_PLOT_INIT = False

if(WEIGHT_PROFILE):
  #  Schedule weights for target reaching
  for k,m in enumerate(ddp.problem.runningModels):
      m.differential.costs.costs['translation'].weight = ocp_utils.cost_weight_tanh(k, N_h, max_weight=10., alpha=5., alpha_cut=0.65)
      m.differential.costs.costs['stateReg'].weight = ocp_utils.cost_weight_parabolic(k, N_h, min_weight=0.001, max_weight=0.1)
      m.differential.costs.costs['ctrlReg'].weight  = ocp_utils.cost_weight_parabolic(k, N_h, min_weight=0.001, max_weight=0.1)
      # print("IAM["+str(k)+"].ee = "+str(m.differential.costs.costs['placement'].weight)+
      # " | IAM["+str(k)+"].xReg = "+str(m.differential.costs.costs['stateReg'].weight))

xs_init = [y0 for i in range(N_h+1)]
us_init = [ug for i in range(N_h)]# ddp.problem.quasiStatic(xs_init[:-1])

if(SOLVE_AND_PLOT_INIT):
  xs_init = [y0 for i in range(N_h+1)]
  us_init = [ug for i in range(N_h)]# ddp.problem.quasiStatic(xs_init[:-1])
  ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
  # for i in range(N_h):
  #   print(ddp.problem.runningDatas[i].differential.costs.costs['ctrlReg'].activation.a_value)
  # print(ddp.problem.terminalData.differential.costs.costs['ctrlReg'].activation.a_value)
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
WHICH_PLOTS = ['y','w', 'p']                                  # Which plots to generate ? ('y':state, 'w':control, 'p':end-eff, etc.)
DELAY_SIM = config['DELAY_SIM']                               # Add delay in reference torques (low-level)
DELAY_OCP = config['DELAY_OCP']                               # Add delay in OCP solution (i.e. ~1ms resolution time)
SCALE_TORQUES = config['SCALE_TORQUES']                       # Affine scaling of reference torque
NOISE_TORQUES = config['NOISE_TORQUES']                       # Add Gaussian noise on reference torques
FILTER_TORQUES = config['FILTER_TORQUES']                     # Moving average smoothing of reference torques
NOISE_STATE = config['NOISE_STATE']                           # Add Gaussian noise on the measured state 
FILTER_STATE = config['FILTER_STATE']                         # Moving average smoothing of reference torques
dt_ocp = dt                                                   # OCP sampling rate 
dt_mpc = float(1./sim_data['plan_freq'])                      # planning rate
OCP_TO_PLAN_RATIO = dt_mpc / dt_ocp                           # ratio
print("Scaling OCP-->PLAN : ", OCP_TO_PLAN_RATIO) 

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
  print('- Total # of simulation steps          : N_simu = '+str(sim_data['N_simu']))
  print('- Total # of control steps             : N_ctrl = '+str(sim_data['N_ctrl']))
  print('- Total # of planning steps            : N_plan = '+str(sim_data['N_plan']))
  print('- Duration of MPC horizon              : T_ocp  = '+str(sim_data['T_h'])+' s')
  print('- OCP integration step                 : dt     = '+str(config['dt'])+' s')
  print("-------------------------------------------------------------------")
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
  print("-------------------------------------------------------------------")
  print("Simulation will start...")
  time.sleep(config['init_log_display_time'])

# Interpolation  

 # ^ := MPC computations
 # | := current MPC computation

 # MPC ITER #1
  #      y_0         y_1         y_2 ...                    --> pred(MPC=O) size N_h
  # OCP : O           O           O                           ref_O = y_1
  # MPC : M     M     M     M     M                           ref_M = y_0 + Interp_[O->M] (y_1 - y_0)
  # CTR : C  C  C  C  C  C  C  C  C                           ref_C = y_0 + Interp_[O->C] (y_1 - y_0)
  # SIM : SSSSSSSSSSSSSSSSSSSSSSSSS                           ref_S = y_0 + Interp_[O->S] (y_1 - y_0)
  #       |     ^     ^     ^     ^  ...
 # MPC ITER #2
  #            y_0         y_1         y_2 ...              --> pred(MPC=1) size N_h
  #             O           O           O                     ...
  #             M     M     M     M     M
  #             C  C  C  C  C  C  C  C  C
  #             SSSSSSSSSSSSSSSSSSSSSSSSS  
  #             |     ^     ^     ^     ^  ...
 # MPC ITER #3
  #                        y_0         y_1         y_2 ...  --> pred(MPC=2) size N_h
  #                         O           O           O         ...
  #                         M     M     M     M     M
  #                         C  C  C  C  C  C  C  C  C
  #                         SSSSSSSSSSSSSSSSSSSSSSSSS  
  #                         |     ^     ^     ^     ^  ...
 # ...

# SIMULATE
for i in range(sim_data['N_simu']): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print("  ")
      print("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))

  # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        # print("PLAN ("+str(nb_plan)+"/"+str(sim_data['N_plan'])+")")
        # for k,m in enumerate(ddp.problem.runningModels):
        #   m.differential.costs.costs['translation'].weight = ocp_utils.cost_weight_tanh(i, sim_data['N_simu'], max_weight=1., alpha=5., alpha_cut=0.65)
          # m.differential.costs.costs['stateReg'].weight = ocp_utils.cost_weight_parabolic(i, sim_data['N_simu'], min_weight=0.001, max_weight=0.1)
          # m.differential.costs.costs['ctrlReg'].weight  = ocp_utils.cost_weight_parabolic(i, sim_data['N_simu'], min_weight=0.01, max_weight=0.1)
        # if(i%1000==0):
        #   print("Placement = ", ddp.problem.runningModels[0].differential.costs.costs['translation'].weight )
        #   print("stateReg  = ", ddp.problem.runningModels[0].differential.costs.costs['stateReg'].weight )
        #   print("ctrlReg decreasing from ", ddp.problem.runningModels[0].differential.costs.costs['ctrlReg'].weight, " to ", ddp.problem.runningModels[-1].differential.costs.costs['ctrlReg'].weight )
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
        y_curr = sim_data['Y_pred'][nb_plan, 0, :]    # y0* = measured state    (q^,  v^ , tau^ )
        y_pred = sim_data['Y_pred'][nb_plan, 1, :]    # y1* = predicted state   (q1*, v1*, tau1*) 
        w_curr = sim_data['W_pred'][nb_plan, 0, :]    # w0* = optimal control   (w0*) !! UNFILTERED TORQUE !!
        # w_pred = sim_data['W_pred'][nb_plan, 1, :]  # w1* = predicted optimal control   (w1*) !! UNFILTERED TORQUE !!
        # Record solver data (optional)
        if(config['RECORD_SOLVER_DATA']):
          sim_data['K'][nb_plan, :, :, :] = np.array(ddp.K)         # Ricatti gains
          sim_data['Vxx'][nb_plan, :, :, :] = np.array(ddp.Vxx)     # Hessians of V.F. 
          sim_data['Quu'][nb_plan, :, :, :] = np.array(ddp.Quu)     # Hessians of Q 
          sim_data['xreg'][nb_plan] = ddp.x_reg                     # Reg solver on x
          sim_data['ureg'][nb_plan] = ddp.u_reg                     # Reg solver on u
          sim_data['J_rank'][nb_plan] = np.linalg.matrix_rank(ddp.problem.runningDatas[0].differential.pinocchio.J)
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
        y_ref_PLAN  = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
        w_ref_PLAN  = w_curr #w_pred_prev + OCP_TO_PLAN_RATIO * (w_curr - w_pred_prev)
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
        COEF       = float(i%int(freq_CTRL/freq_PLAN)) / float(freq_CTRL/freq_PLAN)
        y_ref_CTRL = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)# y_curr + COEF * OCP_TO_PLAN_RATIO * (y_pred - y_curr)
        w_ref_CTRL = w_curr #w_pred_prev + OCP_TO_PLAN_RATIO * (w_curr - w_pred_prev) #w_pred_prev + COEF * OCP_TO_PLAN_RATIO * (w_curr - w_pred_prev)
        # First prediction = measurement = initialization of MPC
        if(nb_ctrl==0):
          sim_data['Y_des_CTRL'][nb_ctrl, :] = y_curr  
        sim_data['W_des_CTRL'][nb_ctrl, :]   = w_ref_CTRL  
        sim_data['Y_des_CTRL'][nb_ctrl+1, :] = y_ref_CTRL   
        # Increment control counter
        nb_ctrl += 1
        
  # Simulate actuation with PI torque tracking controller (low-level control frequency)

    # Select reference control and state for the current SIMU cycle
    COEF        = float(i%int(freq_SIMU/freq_PLAN)) / float(freq_SIMU/freq_PLAN)
    y_ref_SIMU  = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)# y_curr + COEF * OCP_TO_PLAN_RATIO * (y_pred - y_curr)
    w_ref_SIMU  = w_curr #w_pred_prev + OCP_TO_PLAN_RATIO * (w_curr - w_pred_prev)# w_pred_prev + COEF * OCP_TO_PLAN_RATIO * (w_curr - w_pred_prev)

    # First prediction = measurement = initialization of MPC
    if(i==0):
      sim_data['Y_des_SIMU'][i, :] = y_curr  
    sim_data['W_des_SIMU'][i, :]   = w_ref_SIMU  
    sim_data['Y_des_SIMU'][i+1, :] = y_ref_SIMU 

    # Torque applied by motor on actuator : interpolate current torque and predicted torque 
    tau_ref_SIMU =  y_ref_SIMU[-nu:] # y_curr[-nu:]+ COEF * EPS * (y_pred[-nu:] - y_curr[-nu:]) # (dt_sim/dt_mpc) # 
    # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU )    
    tau_mea_SIMU = tau_ref_SIMU 
    if(SCALE_TORQUES):
      tau_mea_SIMU = sim_data['alpha'] * tau_mea_SIMU + sim_data['beta']
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
    # alpha_ = 1./float(1+2*np.pi*config['f_c']*dt)
    # tau_mea_SIMU = alpha_*tau_mea_SIMU + (1-alpha_)*w_curr # in fact u_des as long as old actuation model is desactivated
    #  Send output of actuation torque to the RBD simulator 
    pybullet_simulator.send_joint_command(tau_mea_SIMU)#w_curr)  #y_ref_CTRL[-nu:]
    p.stepSimulation()
    # Measure new state from simulation :
    q_mea_SIMU, v_mea_SIMU = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
    # Record data (unnoised)
    y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU]).T 
    sim_data['Y_mea_no_noise_SIMU'][i+1, :] = y_mea_SIMU
    # Optional noise + filtering
    if(NOISE_STATE):# and float(i)/freq_SIMU <= 0.2):
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

print('--------------------------------')
print('Simulation exited successfully !')
print('--------------------------------')

# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = config_name+'_BIAS='+str(SCALE_TORQUES)+\
                        '_NOISE='+str(NOISE_STATE or NOISE_TORQUES)+\
                        '_DELAY='+str(DELAY_OCP or DELAY_SIM)+\
                        '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)
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