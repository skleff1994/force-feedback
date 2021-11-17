"""
@package force_feedback
@file iiwa_tracking_circle_MPC_bullet.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop MPC for static target task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateLPF x=(q,v), control u = tau 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate closed-loop MPC on a simple reaching task 
'''

import numpy as np  
from utils import path_utils, sim_utils, ocp_utils, pin_utils, plot_utils, data_utils, mpc_utils
import pybullet as p
import time 
np.random.seed(1)
np.set_printoptions(precision=4, linewidth=180)



# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
print("--------------------------------------")
print("              INIT SIM                ")
print("--------------------------------------")
# Read config file
config_name = 'iiwa_tracking_circle_MPC'
config      = path_utils.load_config_file(config_name)
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
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]


# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
print("--------------------------------------")
print("              INIT OCP                ")
print("--------------------------------------")
ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=False, 
                                            WHICH_COSTS=config['WHICH_COSTS']) 
# Create circle trajectory (WORLD frame)
EE_ref = ocp_utils.circle_trajectory_WORLD(M_ee.copy(), dt=config['dt'], 
                                                        radius=config['frameCircleTrajectoryRadius'], 
                                                        omega=config['frameCircleTrajectoryVelocity'])
# Set EE translation cost model references (i.e. setup tracking problem)
models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
for k,m in enumerate(models):
    if(k<EE_ref.shape[0]):
        ref = EE_ref[k]
    else:
        ref = EE_ref[-1]
    m.differential.costs.costs['translation'].cost.residual.reference = ref
print("Setup tracking problem.")
# Warm start state = IK of circle trajectory
print("Computing warm-start...")
WARM_START_IK = True
if(WARM_START_IK):
    xs_init = [] 
    us_init = []
    q_ws = q0
    for k,m in enumerate(models):
        ref = m.differential.costs.costs['translation'].cost.residual.reference
        q_ws, v_ws, eps = pin_utils.IK_position(robot, q_ws, id_endeff, ref, DT=1e-2, IT_MAX=100)
        # print(q_ws, v_ws)
        xs_init.append(np.concatenate([q_ws, v_ws]))
    us_init = [pin_utils.get_u_grav(xs_init[i][:nq], robot.model) for i in range(config['N_h'])]
# Classical warm start using initial config
else:
    ug  = pin_utils.get_u_grav(q0, robot.model)
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [ug for i in range(config['N_h'])]
print("Initial OCP solving...")
# solve
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
# Plot initial solution
PLOT_INIT = False
if(PLOT_INIT):
  ddp_data = data_utils.extract_ddp_data(ddp)
  fig, ax = plot_utils.plot_ddp_results(ddp_data, markers=['.'], SHOW=True)




# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
print("--------------------------------------")
print("              INIT MPC                ")
print("--------------------------------------")
sim = mpc_utils.MPCDataRecorder(config, robot, x0) #data_utils.init_sim_data(config, robot, x0)
  # Get frequencies
freq_PLAN = sim.data['plan_freq']
freq_CTRL = sim.data['ctrl_freq']
freq_SIMU = sim.data['simu_freq']
plan_steps_counter = 0        # number of plan steps
ctrl_steps_counter = 0        # number of control steps

actuator      = mpc_utils.ActuationModel(config, nu=nq)
sensor        = mpc_utils.SensingModel(config, nq=nq, nv=nv) 
communication = mpc_utils.CommunicationModel(config) 


# # # # # # # # # # # #
### SIMULATION LOOP ###
# # # # # # # # # # # #
if(config['INIT_LOG']):
  print('                  ***********************')
  print('                  * Simulation is ready *') 
  print('                  ***********************')        
  print("-------------------------------------------------------------------")
  print('- Total simulation duration            : T_tot  = '+str(sim.data['T_tot'])+' s')
  print('- Simulation frequency                 : f_simu = '+str(float(freq_SIMU/1000.))+' kHz')
  print('- Control frequency                    : f_ctrl = '+str(float(freq_CTRL/1000.))+' kHz')
  print('- Replanning frequency                 : f_plan = '+str(float(freq_PLAN/1000.))+' kHz')
  print('- Total # of simulation steps          : N_simu = '+str(sim.data['N_simu']))
  print('- Total # of control steps             : N_ctrl = '+str(sim.data['N_ctrl']))
  print('- Total # of planning steps            : N_plan = '+str(sim.data['N_plan']))
  print('- Duration of MPC horizon              : T_ocp  = '+str(sim.data['T_h'])+' s')
  print('- OCP integration step                 : dt     = '+str(config['dt'])+' s')
  print("-------------------------------------------------------------------")
  print('- Simulate delay in low-level torque?  : DELAY_SIM                = '+str(config['DELAY_SIM'])+' ('+str(config['delay_sim_cycle'])+' cycles)')
  print('- Simulate delay in OCP solution?      : DELAY_OCP                = '+str(config['DELAY_OCP'])+' ('+str(config['delay_OCP_ms'])+' ms)')
  print('- Affine scaling of ref. ctrl torque?  : SCALE_TORQUES            = '+str(config['SCALE_TORQUES']))
  if(config['SCALE_TORQUES']):
    print('    a='+str(sim.data['alpha'])+'\n')
    print('    b='+str(sim.data['beta'])+')')
  print('- Noise on torques?                    : NOISE_TORQUES            = '+str(config['NOISE_TORQUES']))
  print('- Filter torques?                      : FILTER_TORQUES           = '+str(config['FILTER_TORQUES']))
  print('- Noise on state?                      : NOISE_STATE              = '+str(config['NOISE_STATE']))
  print('- Filter state?                        : FILTER_STATE             = '+str(config['FILTER_STATE']))
  print("-------------------------------------------------------------------")
  print("Simulation will start...")
  time.sleep(config['init_log_display_time'])

# SIMULATE
for i in range(sim.data['N_simu']): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print("  ")
      print("SIMU step "+str(i)+"/"+str(sim.data['N_simu']))

  # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        # Shift references of EE translation CostModelResiduals if next OCP interval is reached
        if(plan_steps_counter%int(1./sim.OCP_TO_PLAN_RATIO)==0):
          models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
          for k,m in enumerate(models):
              if(k+plan_steps_counter<EE_ref.shape[0]):
                ref = EE_ref[k+plan_steps_counter]
              else:
                ref = EE_ref[-1]
              m.differential.costs.costs['translation'].cost.residual.reference = ref
        # Reset x0 to measured state 
        ddp.problem.x0 = sim.data['X_mea_SIMU'][i, :]
        # Warm-start = previous solution
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim.data['X_mea_SIMU'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP 
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        # Record predictions
        sim.record_ddp_data(plan_steps_counter, ddp)
        # Communication model ( remote machine <-> robot ) 
        x_pred, u_curr = communication.step(ddp)
        # Record sim data
        sim.record_des_PLAN(plan_steps_counter, x_pred, u_curr)
        # Increment counter 
        plan_steps_counter += 1


  # If we are in a control cycle select reference torque to send to the actuator
    if(i%int(freq_SIMU/freq_CTRL) == 0):        
        # Select reference control and state for the current CTRL cycle
        sim.record_des_CTRL(ctrl_steps_counter) 
        # Increment counter 
        ctrl_steps_counter += 1


  # Simulate actuation with PI torque tracking controller (low-level control frequency)
    # Record desired 
    sim.record_des_SIMU(i)
    # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU )    
    tau_mea_SIMU = actuator.step(i, sim.u_ref_SIMU)
    # Send output of actuation torque to the RBD simulator 
    pybullet_simulator.send_joint_command(tau_mea_SIMU)
    p.stepSimulation()
    # Measure new state from simulation :
    q_mea_SIMU, v_mea_SIMU = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
    # Record data (unnoised)
    x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
    sim.data['X_mea_no_noise_SIMU'][i+1, :] = x_mea_SIMU
    # Sensing model
    sim.data['X_mea_SIMU'][i+1, :] = sensor.step(i, x_mea_SIMU, sim.data['X_mea_SIMU'])


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
plot_data = data_utils.extract_plot_data_from_sim_data(sim)
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
  data_utils.save_data(sim, save_name=save_name, save_dir=save_dir)