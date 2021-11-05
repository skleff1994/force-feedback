"""
@package force_feedback
@file iiwa_LPF_contact_MPC_raisim.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop 'LPF torque feedback' MPC for force task with the KUKA iiwa 
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
import time
from utils import raisim_utils, path_utils, ocp_utils, pin_utils, plot_utils, data_utils
np.set_printoptions(precision=4, linewidth=180)
np.random.seed(1)




# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
print("--------------------------------------")
print("              INIT SIM                ")
print("--------------------------------------")
# Read config file
config_name = 'iiwa_LPF_contact_MPC'
config = path_utils.load_config_file(config_name)
# Initialize simulator and reset robot model to intial state
dt_simu = 1./float(config['simu_freq']) 
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])  
env, robot = raisim_utils.init_kuka_RAISIM(dt=dt_simu, x0=x0) 
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv; nu = nq
# Display contact surface
id_endeff = robot.model.getFrameId('contact')
  # Placement of reference of the contact in Crocoddyl (Baumgarte integration and friction cost)
M_ct              = robot.data.oMf[id_endeff].copy() 
  # Initial placement of contacted object in simulator
contact_placement = M_ct.copy()
offset = robot.config.tennis_ball_radius #+ 0.001  
contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
# env.display_ball(contact_placement, radius=0.1) 
env.display_wall(contact_placement)
print("-----------------------------------------")
print("[Raisim] Created robot and contact object")
print("-----------------------------------------")




# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
print("--------------------------------------")
print("              INIT OCP                ")
print("--------------------------------------")
# Create DDP solver + compute warm start torque
f_ext = pin_utils.get_external_joint_torques(M_ct, config['frameForceRef'], robot)
u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model)
y0 = np.concatenate([x0, u0])
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=False, 
                                                w_reg_ref='gravity',
                                                TAU_PLUS=False, 
                                                LPF_TYPE=config['LPF_TYPE'],
                                                WHICH_COSTS=config['WHICH_COSTS'] ) 
# Warmstart and solve
xs_init = [y0 for i in range(config['N_h']+1)]
us_init = [u0 for i in range(config['N_h'])]
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
# Plot initial solution
PLOT_INIT = False
if(PLOT_INIT):
  ddp_data = data_utils.extract_ddp_data_LPF(ddp)
  fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, markers=['.'], SHOW=True)




# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
print("--------------------------------------")
print("              INIT MPC                ")
print("--------------------------------------")
sim_data = data_utils.init_sim_data_LPF(config, robot, y0)
  # Get frequencies
freq_PLAN = sim_data['plan_freq']
freq_CTRL = sim_data['ctrl_freq']
freq_SIMU = sim_data['simu_freq']
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
dt_ocp = config['dt']                                         # OCP sampling rate 
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


# SIMULATE
for i in range(sim_data['N_simu']): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print("  ")
      print("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))

  # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        # print("PLAN ("+str(nb_plan)+"/"+str(sim_data['N_plan'])+")")
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
        w_ref_PLAN  = w_curr
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
        y_ref_CTRL = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
        w_ref_CTRL = w_curr 
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
    y_ref_SIMU  = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
    w_ref_SIMU  = w_curr 

    # First prediction = measurement = initialization of MPC
    if(i==0):
      sim_data['Y_des_SIMU'][i, :] = y_curr  
    sim_data['W_des_SIMU'][i, :]   = w_ref_SIMU  
    sim_data['Y_des_SIMU'][i+1, :] = y_ref_SIMU 

    # Torque applied by motor on actuator : interpolate current torque and predicted torque 
    tau_ref_SIMU =  y_ref_SIMU[-nu:]  
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

    #  Send output of actuation torque to the RBD simulator 
    robot.send_joint_command(tau_mea_SIMU)
    env.step()
    # Measure new state from simulation :
    q_mea_SIMU, v_mea_SIMU = robot.get_state()
    # Measure force from simulation
    f_mea_SIMU = robot.get_contact_forces()
    if(i%50==0): 
      print(f_mea_SIMU)
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
    sim_data['F_mea_SIMU'][i, :] = f_mea_SIMU

print('--------------------------------')
print('Simulation exited successfully !')
print('--------------------------------')

# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = config_name+'_bullet_'+\
                        '_BIAS='+str(SCALE_TORQUES)+\
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