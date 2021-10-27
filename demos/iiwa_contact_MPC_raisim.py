
"""
@package force_feedback
@file iiwa_contact_MPC_raisim.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop MPC for static target task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from state x=(q,v), control u = tau) 
Using Raisim simulator for rigid-body dynamics & RaisimUnityOpenGL GUI visualization
The goal of this script is to simulate closed-loop MPC on a simple reaching task 
'''

import numpy as np
import time
from utils import raisim_utils, path_utils, ocp_utils, pin_utils, plot_utils, data_utils
np.set_printoptions(precision=4, linewidth=180)

# Fix seed 
np.random.seed(1)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config_name = 'iiwa_contact_MPC'
config = path_utils.load_config_file(config_name)
# Load Kuka config from URDF
urdf_path = "/home/skleff/robot_properties_kuka_RAISIM/iiwa_test.urdf"
mesh_path = "/home/skleff/robot_properties_kuka_RAISIM"
iiwa_config = raisim_utils.IiwaMinimalConfig(urdf_path, mesh_path)

# Load Raisim environment
LICENSE_PATH = '/home/skleff/.raisim/activation.raisim'
env = raisim_utils.RaiEnv(LICENSE_PATH, dt=1./config['simu_freq'])
robot = env.add_robot(iiwa_config, init_config=None)
env.launch_server()

# Initialize simulation
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv; nu = nq
# Update robot model with initial state
robot.reset_state(q0, v0)
robot.forward_robot(q0, v0)
print(robot.get_state())
M_ee = robot.data.oMf[id_endeff]
print("Initial placement : \n")
print(M_ee)


# Display contact surface
id_endeff = robot.model.getFrameId('contact')
  # Placement of reference of the contact in Crocoddyl (Baumgarte integration and friction cost)
M_ct              = robot.data.oMf[id_endeff].copy() 
  # Initial placement of contacted object in simulator
contact_placement = robot.data.oMf[id_endeff].copy()
offset = iiwa_config.tennis_ball_radius 
contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
env.display_ball(contact_placement, radius=0.1) 
# env.display_wall(contact_placement)
print("-----------------------")
print("[Raisim] Created robot ")
print("-----------------------")

#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
ug = pin_utils.get_u_grav(q0, robot)
# Warm start and reg
import pinocchio as pin
f_ext = []
for i in range(nq+1):
    # CONTACT --> WORLD
    W_M_ct = contact_placement.copy()
    f_WORLD = W_M_ct.act(pin.Force(np.asarray(config['frameForceRef'])))
    # WORLD --> JOINT
    j_M_W = robot.data.oMi[i].copy().inverse()
    f_JOINT = j_M_W.act(f_WORLD)
    f_ext.append(f_JOINT)
# print(f_ext)
u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model)


print("--------------------------------------")
print("              INIT OCP                ")
print("--------------------------------------")
ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=False, 
                                            WHICH_COSTS=config['WHICH_COSTS']) 
SOLVE_AND_PLOT_INIT = True


xs_init = [x0 for i in range(N_h+1)]
us_init = [u0 for i in range(N_h)]

if(SOLVE_AND_PLOT_INIT):
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
  # ddp_data = data_utils.extract_ddp_data(ddp)
  # fig, ax = plot_utils.plot_ddp_results(ddp_data, markers=['.'], which_plots=['x','u','p', 'f'], SHOW=True)
        # for k,m in enumerate(ddp.problem.runningModels):
        #   m.differential.

# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
sim_data = data_utils.init_sim_data(config, robot, x0)

# print(sim_data['f_ee_ref'])
# time.sleep(10)
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
x_buffer_OCP = []                                             # buffer for desired controls delayed by OCP computation time
u_buffer_OCP = []                                             # buffer for desired states delayed by OCP computation time
buffer_sim = []                                               # buffer for measured torque delayed by e.g. actuation and/or sensing 
  # Sim options
WHICH_PLOTS = ['x','u', 'p', 'f']                                  # Which plots to generate ? ('y':state, 'w':control, 'p':end-eff, etc.)
DELAY_SIM = config['DELAY_SIM']                               # Add delay in reference torques (low-level)
DELAY_OCP = config['DELAY_OCP']                               # Add delay in OCP solution (i.e. ~1ms resolution time)
SCALE_TORQUES = config['SCALE_TORQUES']                       # Affinescaling of reference torque
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


# SIMULATE
for i in range(sim_data['N_simu']): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print("  ")
      print("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))

  # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        # print("PLAN ("+str(nb_plan)+"/"+str(sim_data['N_plan'])+")")
        # Update OCP 
        # for k,m in enumerate(ddp.problem.runningModels):
        #   m.differential.
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = sim_data['X_mea_SIMU'][i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['X_mea_SIMU'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        sim_data['X_pred'][nb_plan, :, :] = np.array(ddp.xs)
        sim_data['U_pred'][nb_plan, :, :] = np.array(ddp.us)
        sim_data ['F_pred'][nb_plan, :, :] = np.array([ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(N_h)])
        # Extract relevant predictions for interpolations
        x_curr = sim_data['X_pred'][nb_plan, 0, :]    # x0* = measured state    (q^,  v^ , tau^ )
        x_pred = sim_data['X_pred'][nb_plan, 1, :]    # x1* = predicted state   (q1*, v1*, tau1*) 
        u_curr = sim_data['U_pred'][nb_plan, 0, :]    # u0* = optimal control   
        f_curr = sim_data['F_pred'][nb_plan, 0, :]    # f0* = current contact force
        f_pred = sim_data['F_pred'][nb_plan, 1, :]    # f1* = predicted contact force
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
          u_pred_prev = u_curr
        else:
          u_pred_prev = sim_data['U_pred'][nb_plan-1, 1, :]
        # Optionally delay due to OCP resolution time 
        if(DELAY_OCP):
          x_buffer_OCP.append(x_pred)
          u_buffer_OCP.append(u_curr)
          if(len(x_buffer_OCP)<sim_data['delay_OCP_cycle']): 
            pass
          else:                            
            x_pred = x_buffer_OCP.pop(-sim_data['delay_OCP_cycle'])
          if(len(u_buffer_OCP)<sim_data['delay_OCP_cycle']): 
            pass
          else:
            u_curr = u_buffer_OCP.pop(-sim_data['delay_OCP_cycle'])
        # Select reference control and state for the current PLAN cycle
        x_ref_PLAN  = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)
        u_ref_PLAN  = u_curr #u_pred_prev + OCP_TO_PLAN_RATIO * (u_curr - u_pred_prev)
        f_ref_PLAN  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)
        if(nb_plan==0):
          sim_data['X_des_PLAN'][nb_plan, :] = x_curr  
        sim_data['U_des_PLAN'][nb_plan, :]   = u_ref_PLAN   
        sim_data['X_des_PLAN'][nb_plan+1, :] = x_ref_PLAN    
        sim_data['F_des_PLAN'][nb_plan, :] = f_ref_PLAN 

        # Increment planning counter
        nb_plan += 1

  # If we are in a control cycle select reference torque to send to the actuator
    if(i%int(freq_SIMU/freq_CTRL) == 0):        
        # print("  CTRL ("+str(nb_ctrl)+"/"+str(sim_data['N_ctrl'])+")")
        # Select reference control and state for the current CTRL cycle
        COEF       = float(i%int(freq_CTRL/freq_PLAN)) / float(freq_CTRL/freq_PLAN)
        x_ref_CTRL = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)# x_curr + COEF * OCP_TO_PLAN_RATIO * (x_pred - x_curr)
        u_ref_CTRL = u_curr 
        f_ref_CTRL = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)
        # First prediction = measurement = initialization of MPC
        if(nb_ctrl==0):
          sim_data['X_des_CTRL'][nb_ctrl, :] = x_curr  
        sim_data['U_des_CTRL'][nb_ctrl, :]   = u_ref_CTRL  
        sim_data['X_des_CTRL'][nb_ctrl+1, :] = x_ref_CTRL  
        sim_data['F_des_CTRL'][nb_ctrl, :] = f_ref_CTRL    
        # Increment control counter
        nb_ctrl += 1
        
  # Simulate actuation with PI torque tracking controller (low-level control frequency)

    # Select reference control and state for the current SIMU cycle
    COEF        = float(i%int(freq_SIMU/freq_PLAN)) / float(freq_SIMU/freq_PLAN)
    x_ref_SIMU  = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)# x_curr + COEF * OCP_TO_PLAN_RATIO * (x_pred - x_curr)
    u_ref_SIMU  = u_curr 
    f_ref_SIMU  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)

    # First prediction = measurement = initialization of MPC
    if(i==0):
      sim_data['X_des_SIMU'][i, :] = x_curr  
    sim_data['U_des_SIMU'][i, :]   = u_ref_SIMU  
    sim_data['X_des_SIMU'][i+1, :] = x_ref_SIMU 
    sim_data['F_des_SIMU'][i, :] = f_ref_SIMU 

    # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU )    
    tau_mea_SIMU = u_ref_SIMU 
    if(SCALE_TORQUES):
      tau_mea_SIMU = sim_data['alpha'] * tau_mea_SIMU + sim_data['beta']
    if(FILTER_TORQUES):
      n_sum = min(i, config['u_avg_filter_length'])
      for k in range(n_sum):
        tau_mea_SIMU += sim_data['X_mea_SIMU'][i-k-1, -nu:]
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
    # Update pinocchio model
    robot.forward_robot(q_mea_SIMU, v_mea_SIMU)
    # Record data (unnoised)
    x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
    sim_data['X_mea_no_noise_SIMU'][i+1, :] = x_mea_SIMU
    # Optional noise + filtering
    if(NOISE_STATE):# and float(i)/freq_SIMU <= 0.2):
      noise_q = np.random.normal(0., sim_data['var_q'], nq)
      noise_v = np.random.normal(0., sim_data['var_v'], nv)
      noise_tau = np.random.normal(0., sim_data['var_u'], nu)
      x_mea_SIMU += np.concatenate([noise_q, noise_v]).T
    if(FILTER_STATE):
      n_sum = min(i, config['x_avg_filter_length'])
      for k in range(n_sum):
        x_mea_SIMU += sim_data['X_mea_SIMU'][i-k-1, :]
      x_mea_SIMU = x_mea_SIMU / (n_sum + 1)
    # Record noised data
    sim_data['X_mea_SIMU'][i+1, :] = x_mea_SIMU 
    sim_data['F_mea_SIMU'][i, :] = f_mea_SIMU

print('--------------------------------')
print('Simulation exited successfully !')
print('--------------------------------')

# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = config_name+'_raisim_'+\
                        '_BIAS='+str(SCALE_TORQUES)+\
                        '_NOISE='+str(NOISE_STATE or NOISE_TORQUES)+\
                        '_DELAY='+str(DELAY_OCP or DELAY_SIM)+\
                        '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)+\
                        '_RAISIM'
# Extract plot data from sim data
plot_data = data_utils.extract_plot_data_from_sim_data(sim_data)
# Plot results
plot_utils.plot_mpc_results(plot_data, which_plots=WHICH_PLOTS,
                                PLOT_PREDICTIONS=True, 
                                pred_plot_sampling=int(freq_PLAN/10),
                                SAVE=True,
                                SAVE_DIR=save_dir,
                                SAVE_NAME=save_name,
                                AUTOSCALE=True)
# Save optionally
if(config['SAVE_DATA']):
  data_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)

env.server.killServer()