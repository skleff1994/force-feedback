"""
@package force_feedback
@file iiwa_contact_MPC_bullet.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2021-10-28
@brief Closed-loop MPC for force task with the KUKA iiwa (PyBullet)
"""

'''
The robot is tasked with exerting a constant normal force  
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from state x=(q,v), control u = tau) 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with state feedback, optionally
imperfect actuation (bias, noise, delays) at higher frequency
'''


import sys
sys.path.append('.')

import numpy as np  
from utils import path_utils, sim_utils, ocp_utils, pin_utils, plot_utils, data_utils, mpc_utils
import time 
np.set_printoptions(precision=4, linewidth=180)
np.random.seed(1)


import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config_name = 'iiwa_contact_MPC'
config = path_utils.load_config_file(config_name)
# Create a Pybullet simulation environment + set simu freq
dt_simu = 1./float(config['simu_freq'])  
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
env, pybullet_simulator = sim_utils.init_kuka_simulator(dt=dt_simu, x0=x0)
# Get pin wrapper
robot = pybullet_simulator.pin_robot
# Get dimensions 
nq, nv = robot.model.nq, robot.model.nv; nu = nq
# Display contact surface
id_endeff = robot.model.getFrameId('contact')
contact_placement = robot.data.oMf[id_endeff].copy()
M_ct = robot.data.oMf[id_endeff].copy()
offset = 0.036 #0.0335
contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
sim_utils.display_contact_surface(contact_placement, with_collision=True)




# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
# Setup Croco OCP and create solver
ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=False, 
                                            WHICH_COSTS=config['WHICH_COSTS']) 
# Warmstart and solve
f_ext = pin_utils.get_external_joint_torques(M_ct, config['frameForceRef'], robot)
u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model)
xs_init = [x0 for i in range(config['N_h']+1)]
us_init = [u0 for i in range(config['N_h'])]
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
# Plot
PLOT_INIT = False
if(PLOT_INIT):
  ddp_data = data_utils.extract_ddp_data(ddp)
  fig, ax = plot_utils.plot_ddp_results(ddp_data, markers=['.'], SHOW=True)


# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
sim_data = data_utils.init_sim_data(config, robot, x0)
  # Get frequencies
freq_PLAN = sim_data['plan_freq']
freq_CTRL = sim_data['ctrl_freq']
freq_SIMU = sim_data['simu_freq']
  # Replan & control counters
nb_plan = 0
nb_ctrl = 0
  # Sim options
WHICH_PLOTS       = ['x','u', 'p']                          # Which plots to generate ? ('y':state, 'w':control, 'p':end-eff, etc.)
dt_ocp            = config['dt']                            # OCP sampling rate 
dt_mpc            = float(1./sim_data['plan_freq'])         # planning rate
OCP_TO_PLAN_RATIO = dt_mpc / dt_ocp                         # ratio

# Additional simulation blocks 
communication = mpc_utils.CommunicationModel(config)
actuation     = mpc_utils.ActuationModel(config)
sensing       = mpc_utils.SensorModel(config)


# # # # # # # # # # # #
### SIMULATION LOOP ###
# # # # # # # # # # # #


# SIMULATE
for i in range(sim_data['N_simu']): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print('')
      logger.info("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))
      print('')


  # Solve OCP if we are in a planning cycle (MPC/planning frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        # print("PLAN ("+str(nb_plan)+"/"+str(sim_data['N_plan'])+")")
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = sim_data['X_mea_SIMU'][i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['X_mea_SIMU'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        sim_data['X_pred'][nb_plan, :, :] = np.array(ddp.xs)
        sim_data['U_pred'][nb_plan, :, :] = np.array(ddp.us)
        sim_data ['F_pred'][nb_plan, :, :] = np.array([ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(config['N_h'])])
        # Extract relevant predictions for interpolations
        x_curr = sim_data['X_pred'][nb_plan, 0, :]    # x0* = measured state    (q^,  v^ , tau^ )
        x_pred = sim_data['X_pred'][nb_plan, 1, :]    # x1* = predicted state   (q1*, v1*, tau1*) 
        u_curr = sim_data['U_pred'][nb_plan, 0, :]    # u0* = optimal control   
        f_curr = sim_data['F_pred'][nb_plan, 0, :]
        f_pred = sim_data['F_pred'][nb_plan, 1, :]
        # Record cost references
        data_utils.record_cost_references(ddp, sim_data, nb_plan)
        # Record solver data (optional)
        if(config['RECORD_SOLVER_DATA']):
          data_utils.record_solver_data(ddp, sim_data, nb_plan) 
        # Model communication between computer --> robot
        x_pred, u_curr = communication.step(x_pred, u_curr)
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

  # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
    if(i%int(freq_SIMU/freq_CTRL) == 0):        
        # Select reference control and state for the current CTRL cycle
        x_ref_CTRL = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)
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
        
  # Simulate actuation/sensing and step simulator (physics simulation frequency)

    # Select reference control and state for the current SIMU cycle
    x_ref_SIMU  = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)
    u_ref_SIMU  = u_curr 
    f_ref_SIMU  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)

    # First prediction = measurement = initialization of MPC
    if(i==0):
      sim_data['X_des_SIMU'][i, :] = x_curr  
    sim_data['U_des_SIMU'][i, :]   = u_ref_SIMU  
    sim_data['X_des_SIMU'][i+1, :] = x_ref_SIMU 
    sim_data['F_des_SIMU'][i, :] = f_ref_SIMU 

    # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU )    
    tau_mea_SIMU = actuation.step(i, u_ref_SIMU, sim_data['U_des_SIMU'])  
    #  Send output of actuation torque to the RBD simulator 
    pybullet_simulator.send_joint_command(tau_mea_SIMU)
    env.step()
    # Measure new state from simulation 
    q_mea_SIMU, v_mea_SIMU = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
    f_mea_SIMU = sim_utils.get_contact_wrench(pybullet_simulator, id_endeff)
    if(i%50==0): 
      print(f_mea_SIMU)
    # Record data (unnoised)
    x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
    sim_data['X_mea_no_noise_SIMU'][i+1, :] = x_mea_SIMU
    # Sensor model (optional noise + filtering)
    sim_data['X_mea_SIMU'][i+1, :] = sensing.step(i, x_mea_SIMU, sim_data['X_mea_SIMU'])
    sim_data['F_mea_SIMU'][i, :] = f_mea_SIMU

# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = config_name+'_bullet_'+\
                        '_BIAS='+str(config['SCALE_TORQUES'])+\
                        '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                        '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                        '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)
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