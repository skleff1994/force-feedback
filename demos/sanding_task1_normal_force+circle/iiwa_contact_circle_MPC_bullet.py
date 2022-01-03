"""
@package force_feedback
@file iiwa_contact_circle_MPC_bullet.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop MPC for static target task with the KUKA iiwa 
"""

'''
The robot is tasked with exerting a constant normal force at its EE
while drawing a circle on the contact surface
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from state x=(q,v), control u = tau 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate closed-loop MPC
'''

import sys
sys.path.append('.')

import numpy as np  
from utils import path_utils, sim_utils, ocp_utils, pin_utils, plot_utils, data_utils, mpc_utils
import pybullet as p
import time 
np.random.seed(1)
np.set_printoptions(precision=4, linewidth=180)

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
config_name = 'iiwa_contact_circle_MPC'
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
# Get EE frame id and placement
id_endeff = robot.model.getFrameId('contact')
ee_frame_placement = robot.data.oMf[id_endeff].copy()
# Compute contact frame placement 
contact_placement = robot.data.oMf[id_endeff].copy()
M_ct = robot.data.oMf[id_endeff].copy()
offset = 0.03348 #0.0335 gold number = 0.03348 (NO IMPACT, NO PENETRATION)
contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
# Optionally tilt the contact surface
if(config['TILT_SURFACE']):
  TILT_RPY = [0., config['TILT_PITCH_LOCAL_DEG']*np.pi/180, 0.]
  contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY)
# Create the contact surface in PyBullet simulator 
contact_surface_bulletId = sim_utils.display_contact_surface(contact_placement.copy(), with_collision=True)
# Set lateral friction coefficient of the contact surface
sim_utils.set_friction_coef(contact_surface_bulletId, 0.5)



# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
# Init shooting problem and solver
ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=False, 
                                            WHICH_COSTS=config['WHICH_COSTS']) 
# Setup tracking problem with circle ref EE trajectory
models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
RADIUS = config['frameCircleTrajectoryRadius'] 
OMEGA  = config['frameCircleTrajectoryVelocity']
for k,m in enumerate(models):
    # Ref
    t = min(k*config['dt'], 2*np.pi/OMEGA)
    p_ee_ref = ocp_utils.circle_point_WORLD(t, ee_frame_placement, 
                                               radius=RADIUS,
                                               omega=OMEGA)
    # Cost translation
    m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
    # Contact model 1D update z ref (WORLD frame)
    m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref[2]
    # m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref
    
# Warm start state = IK of circle trajectory
WARM_START_IK = True
if(WARM_START_IK):
    logger.info("Computing warm-start using Inverse Kinematics...")
    xs_init = [] 
    us_init = []
    q_ws = q0
    for k,m in enumerate(list(ddp.problem.runningModels) + [ddp.problem.terminalModel]):
        # Get ref placement
        p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
        Mref = ee_frame_placement.copy()
        Mref.translation = p_ee_ref
        q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
        xs_init.append(np.concatenate([q_ws, v_ws]))
    us_init = [pin_utils.get_u_grav(xs_init[i][:nq], robot.model) for i in range(config['N_h'])]
# Classical warm start using initial config
else:
    ug  = pin_utils.get_u_grav(q0, robot.model)
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [ug for i in range(config['N_h'])]

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
sim_data = data_utils.init_sim_data(config, robot, x0)
  # Get frequencies
freq_PLAN = sim_data['plan_freq']
freq_CTRL = sim_data['ctrl_freq']
freq_SIMU = sim_data['simu_freq']
  # Replan & control counters
nb_plan = 0
nb_ctrl = 0
  # Sim options
WHICH_PLOTS       = config['WHICH_PLOTS']                   # Which plots to generate ? ('y':state, 'w':control, 'p':end-eff, etc.)
dt_ocp            = config['dt']                            # OCP sampling rate 
dt_mpc            = float(1./sim_data['plan_freq'])         # planning rate
OCP_TO_PLAN_RATIO  = dt_mpc / dt_ocp                         # ratio
PLAN_TO_SIMU_RATIO = dt_simu / dt_mpc                        # Must be an integer !!!!
OCP_TO_SIMU_RATIO  = dt_simu / dt_ocp                        # Must be an integer !!!!
if(1./PLAN_TO_SIMU_RATIO%1 != 0):
  logger.warning("SIMU->MPC ratio not an integer ! (1./PLAN_TO_SIMU_RATIO = "+str(1./PLAN_TO_SIMU_RATIO)+")")
if(1./OCP_TO_SIMU_RATIO%1 != 0):
  logger.warning("SIMU->OCP ratio not an integer ! (1./OCP_TO_SIMU_RATIO  = "+str(1./OCP_TO_SIMU_RATIO)+")")


# Additional simulation blocks 
communication = mpc_utils.CommunicationModel(config)
actuation     = mpc_utils.ActuationModel(config)
sensing       = mpc_utils.SensorModel(config)


# Display target circle  trajectory (reference)
nb_points = 20 
for i in range(nb_points):
  t = (i/nb_points)*2*np.pi/OMEGA
  pl = pin_utils.rotate(ee_frame_placement, rpy=TILT_RPY)
  pos = ocp_utils.circle_point_WORLD(t, pl, radius=RADIUS, omega=OMEGA)
  sim_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])

draw_rate = 200




# SIMULATE
for i in range(sim_data['N_simu']): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print('')
      logger.info("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))
      print('')
  

  # Solve OCP if we are in a planning cycle (MPC/planning frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        # Current simulation time
        t_simu = i*dt_simu 
        # Setup tracking problem with circle ref EE trajectory
        models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
        for k,m in enumerate(models):
            # Ref
            t = min(t_simu + k*dt_ocp, 2*np.pi/OMEGA)
            p_ee_ref = ocp_utils.circle_point_WORLD(t, ee_frame_placement.copy(), 
                                                       radius=RADIUS,
                                                       omega=OMEGA)
            # Cost translation
            m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
            # Contact model
            m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref[2]

        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = sim_data['state_mea_SIMU'][i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['state_mea_SIMU'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        sim_data['state_pred'][nb_plan, :, :] = np.array(ddp.xs)
        sim_data['ctrl_pred'][nb_plan, :, :] = np.array(ddp.us)
        sim_data ['force_pred'][nb_plan, :, :] = np.array([ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(config['N_h'])])
        # Extract relevant predictions for interpolations
        x_curr = sim_data['state_pred'][nb_plan, 0, :]    # x0* = measured state    (q^,  v^ , tau^ )
        x_pred = sim_data['state_pred'][nb_plan, 1, :]    # x1* = predicted state   (q1*, v1*, tau1*) 
        u_curr = sim_data['ctrl_pred'][nb_plan, 0, :]    # u0* = optimal control   
        f_curr = sim_data['force_pred'][nb_plan, 0, :]
        f_pred = sim_data['force_pred'][nb_plan, 1, :]
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
          sim_data['state_des_PLAN'][nb_plan, :] = x_curr  
        sim_data['ctrl_des_PLAN'][nb_plan, :]   = u_ref_PLAN   
        sim_data['state_des_PLAN'][nb_plan+1, :] = x_ref_PLAN    
        sim_data['force_des_PLAN'][nb_plan, :] = f_ref_PLAN    
        
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
          sim_data['state_des_CTRL'][nb_ctrl, :] = x_curr  
        sim_data['ctrl_des_CTRL'][nb_ctrl, :]   = u_ref_CTRL  
        sim_data['state_des_CTRL'][nb_ctrl+1, :] = x_ref_CTRL   
        sim_data['force_des_CTRL'][nb_ctrl, :] = f_ref_CTRL   
        # Increment control counter
        nb_ctrl += 1

  # Simulate actuation/sensing and step simulator (physics simulation frequency)

    # Select reference control and state for the current SIMU cycle
    x_ref_SIMU  = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)
    u_ref_SIMU  = u_curr 
    f_ref_SIMU  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)

    # First prediction = measurement = initialization of MPC
    if(i==0):
      sim_data['state_des_SIMU'][i, :] = x_curr  
    sim_data['ctrl_des_SIMU'][i, :]   = u_ref_SIMU  
    sim_data['state_des_SIMU'][i+1, :] = x_ref_SIMU 
    sim_data['force_des_SIMU'][i, :] = f_ref_SIMU 

    # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU )    
    tau_mea_SIMU = actuation.step(i, u_ref_SIMU, sim_data['ctrl_des_SIMU'])  
    #  Send output of actuation torque to the RBD simulator 
    pybullet_simulator.send_joint_command(tau_mea_SIMU)
    env.step()
    # Measure new state from simulation 
    q_mea_SIMU, v_mea_SIMU = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
    f_mea_SIMU = sim_utils.get_contact_wrench(pybullet_simulator, id_endeff)
    if(i%100==0): 
      print(f_mea_SIMU)
    # Record data (unnoised)
    x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
    sim_data['state_mea_no_noise_SIMU'][i+1, :] = x_mea_SIMU
    # Sensor model (optional noise + filtering)
    sim_data['state_mea_SIMU'][i+1, :] = sensing.step(i, x_mea_SIMU, sim_data['state_mea_SIMU'])
    sim_data['force_mea_SIMU'][i, :] = f_mea_SIMU


    # Display real 
    if(i%draw_rate==0):
      pos = pybullet_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
      sim_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])


# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/force-feedback/data'
save_name = config_name+'_bullet_'+\
                        '_BIAS='+str(config['SCALE_TORQUES'])+\
                        '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                        '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                        '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)+\
                        '_'+str(time.time())
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