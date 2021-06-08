"""
@package force_feedback
@file mpc_iiwa_sim.py
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


import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np  
import pinocchio as pin
import crocoddyl
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot, IiwaConfig
from utils import utils 
import pybullet as p
import time 


############################################
### ROBOT MODEL & SIMULATION ENVIRONMENT ###
############################################
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config/'))
config_file = config_path+"/static_reaching_task2"+".yml"
config = utils.load_config_file(config_file)
    # Create a Pybullet simulation environment
simu_freq = config['simu_freq']         # simulation > control frequency 
dt_simu = 1./simu_freq
env = BulletEnvWithGround(p.GUI, dt=dt_simu)
pybullet_simulator = IiwaRobot()
env.add_robot(pybullet_simulator)
    # Create a robot instance. This initializes the simulator as well.
robot = pybullet_simulator.pin_robot
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
    # Initial state 
q0 = np.asarray(config['q0'])
dq0 = np.asarray(config['dq0']) 
print(q0, dq0)
    # Reset robot to initial state in PyBullet
pybullet_simulator.reset_state(q0, dq0)
    # Update pinocchio data accordingly 
pybullet_simulator.forward_robot(q0, dq0)
    # Get initial frame placement
M_ee = robot.data.oMf[id_endeff]
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")

time.sleep(1)

#################
### OCP SETUP ###
#################
  # OCP parameters 
dt = config['dt']                   # OCP integration step (s)               
N_h = config['N_h']                 # Number of knots in the horizon 
x0 = np.concatenate([q0, dq0])      # Initial state
print("Initial state : ", x0.T)
  # Construct cost function terms
   # State and actuation models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
   # State regularization
stateRegWeights = np.asarray(config['stateRegWeights'])
x_reg_ref = x0 #np.zeros(nq+nv)     
xRegCost = crocoddyl.CostModelState(state, 
                                    crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                    x_reg_ref, 
                                    actuation.nu)
print("Created state reg cost.")
   # Control regularization
ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
# ctrlRegWeights[-1] = 2
u_grav = pin.rnea(robot.model, robot.data, x0[:nq], np.zeros((nv,1)), np.zeros((nq,1))) #
uRegCost = crocoddyl.CostModelControl(state, 
                                      crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                      u_grav)
print("Created ctrl reg cost.")
   # State limits penalization
x_lim_ref  = np.zeros(nq+nv)
xLimitCost = crocoddyl.CostModelState(state, 
                                      crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(state.lb, state.ub)), 
                                      x_lim_ref, 
                                      actuation.nu)
print("Created state lim cost.")
   # Control limits penalization
u_min = -np.asarray(config['u_lim']) 
u_max = +np.asarray(config['u_lim']) 
u_lim_ref = np.zeros(nq)
uLimitCost = crocoddyl.CostModelControl(state, 
                                        crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                        u_lim_ref)
print("Created ctrl lim cost.")
   # End-effector placement 
p_target = np.asarray(config['p_des']) 
M_target = pin.SE3(M_ee.rotation.T, p_target)
desiredFramePlacement = M_ee #M_target
framePlacementWeights = np.asarray(config['framePlacementWeights'])
framePlacementCost = crocoddyl.CostModelFramePlacement(state, 
                                                       crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                       crocoddyl.FramePlacement(id_endeff, desiredFramePlacement), 
                                                       actuation.nu) 
print("Created frame placement cost.")
# Create IAMs
runningModels = []
for i in range(N_h):
  # Create IAM 
  runningModels.append(crocoddyl.IntegratedActionModelEuler( 
      crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                       actuation, 
                                                       crocoddyl.CostModelSum(state, nu=actuation.nu)), dt ) )
  # Add cost models
  runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['frameWeight'])
  runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['xRegWeight'])
  runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['uRegWeight'])
  runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['xLimWeight'])
  runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['uLimWeight'])
  # Add armature
  runningModels[i].differential.armature = np.asarray(config['armature'])
  # Terminal IAM + set armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                        actuation, 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
   # Add cost models
terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
terminalModel.differential.costs.addCost("stateReg", xRegCost, config['xRegWeightTerminal'])
terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['xLimWeightTerminal'])
  # Add armature
terminalModel.differential.armature = np.asarray(config['armature']) 
print("Initialized IAMs.")
# Create the shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
# Creating the DDP solver 
ddp = crocoddyl.SolverFDDP(problem)
print("OCP is ready to be solved.")
# Solve and extract solution trajectories
xs = [x0] * (N_h+1)
us = [ddp.problem.runningModels[0].quasiStatic(ddp.problem.runningDatas[0], x0)] * N_h
ddp.solve(xs, us, maxiter=100)
xs = np.array(ddp.xs)
us = np.array(ddp.us)


##################
# MPC SIMULATION #
##################
# MPC & simulation parameters
maxit = config['maxiter']
T_tot = config['T_tot']
plan_freq = config['plan_freq']       # MPC re-planning frequency (Hz)
ctrl_freq = config['ctrl_freq']       # Control - simulation - frequency (Hz)
N_plan = int(T_tot*plan_freq)         # Total number of planning steps in the simulation
N_ctrl = int(T_tot*ctrl_freq)         # Total number of control steps in the simulation 
N_simu = int(T_tot*simu_freq)         # Total number of simulation steps 
T_h = N_h*dt                          # Duration of the MPC horizon (s)
dt_ctrl = float(1./ctrl_freq)         # Time step duration of the control loop
dt_plan = float(1./plan_freq)         # Time step duration of planning loop
# Initialize data
nx = nq+nv
nu = nq
X_pred = np.zeros((N_plan, N_h+1, nx))   # Predicted states
U_pred = np.zeros((N_plan, N_h, nu))     # Predicted torques
X_des = np.zeros((N_ctrl+1, nx))         # Desired states (next predicted state)
U_des = np.zeros((N_ctrl, nu))           # Desired torques (current ff output by DDP)
U_mea = np.zeros((N_simu+1, nu))         # Actuation torques (sent to PyBullet)
X_mea = np.zeros((N_simu+1, nx))         # Measured states 
# Low-level simulation parameters (actuation model)
  # Scaling of desired torque
alpha = np.random.uniform(low=config['alpha_min'], high=config['alpha_max'], size=(nq,))
beta = np.random.uniform(low=config['beta_min'], high=config['beta_max'], size=(nq,))
  # White noise on desired torque and measured state
var_u = np.asarray(config['var_u'])
var_q = config['var_q']
var_v = config['var_v']
  # Buffer for torque delay
delay_ms = config['delay_ms'] # in ms
delay_cycle = 3 #int(delay_ms * 1e-3 * simu_freq)
buffer = []
  # Proportional-integral torque control gains
Kp = config['Kp']*np.eye(nq)
Ki = config['Ki']*np.eye(nq)
Kd = config['Kd']*np.eye(nq)
  # Moving avg filter
avg_length = config['avg_length']
# LOGS
print('                  ************************')
print('                  * MPC controller ready *') 
print('                  ************************')        
print('---------------------------------------------------------')
print('- Total simulation duration            : T_tot  = '+str(T_tot)+' s')
print('- Simulation frequency                 : f_simu = '+str(simu_freq)+' s')
print('- Control frequency                    : f_ctrl = '+str(ctrl_freq)+' Hz')
print('- Replanning frequency                 : f_plan = '+str(plan_freq)+' Hz')
print('- Total # of simulation steps          : N_ctrl = '+str(N_simu))
print('- Total # of control steps             : N_ctrl = '+str(N_ctrl))
print('- Total # of planning steps            : N_plan = '+str(N_plan))
print('- Duration of MPC horizon              : T_ocp  = '+str(T_h)+' s')
print('- OCP integration step                 : dt     = '+str(dt)+' s')
print('---------------------------------------------------------')
print("Simulation will start...")
# INIT
  # Measure initial state from simulation environment &init data
q_mea, v_mea = pybullet_simulator.get_state()
pybullet_simulator.forward_robot(q_mea, v_mea)
x0 = np.concatenate([q_mea, v_mea]).T
print("Initial state ", str(x0))
X_mea[0, :] = x0
X_des[0, :] = x0
  # Initial measurement ot torque
U_mea[0, :] = alpha*u_grav + beta
  # Replan counter
nb_plan = 0
  # Control counter
nb_ctrl = 0
  # Sim options
TORQUE_TRACKING = False
DELAY_TORQUES = False
NOISE_TORQUES = False
SCALE_TORQUES = True
FILTER_TORQUES = False
NOISE_STATE = False
INTERPOLATE_TORQUE = False
vel_U_des = np.zeros((N_ctrl, nu))           # Desired torques (current ff output by DDP)
vel_U_mea = np.zeros((N_simu+1, nu))         # Actuation torques (sent to PyBullet)
vel_U_mea[0,:] = np.zeros(nq)
  # Log
print("TORQUE_TRACKING = ", TORQUE_TRACKING)
print("DELAY_TORQUES   = ", DELAY_TORQUES, " (", delay_cycle, ") ")
print("NOISE_TORQUES = ", NOISE_TORQUES)
print("SCALE_TORQUES = ", SCALE_TORQUES)
print("FILTER_TORQUES  = ", FILTER_TORQUES)
time.sleep(2)
vel_err_u = np.zeros(nq)
vel_u_des = np.zeros(nq)
vel_u_mea = np.zeros(nq)
# SIMULATION LOOP
for i in range(N_simu): 
    print("  ")
    print("SIMU step "+str(i)+"/"+str(N_simu))

    # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(simu_freq/plan_freq) == 0):
        print("  PLAN ("+str(nb_plan)+"/"+str(N_plan)+")")
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = X_mea[i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = X_mea[i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=maxit, isFeasible=False)
        X_pred[nb_plan, :, :] = np.array(ddp.xs)
        U_pred[nb_plan, :, :] = np.array(ddp.us)
        # Extract desired control and predicted state
        u_des = U_pred[nb_plan, 0, :] 
        U_des[nb_ctrl, :] = u_des
        # For interpolation
        if(nb_ctrl>=1):
          u_des_prev = U_des[nb_ctrl-1, :]
        else:
          u_des_prev = u_des
        # Estimate torque vel with FD # point rule
        if(nb_ctrl>=1):
          vel_u_des = ( U_des[nb_ctrl, :] - U_des[nb_ctrl-1, :] ) / dt_ctrl
          # vel_u_des = (U_des[nb_ctrl-4, :] - 8*U_des[nb_ctrl-3, :] + U_des[nb_ctrl-1, :] - U_des[nb_ctrl, :]) / (12*dt_ctrl)
        else:
          vel_u_des = np.zeros(nq)
        vel_U_des[nb_ctrl, :] = vel_u_des
        X_des[nb_ctrl+1, :] = X_pred[nb_plan, 1, :]
        # Reset integral error 
        int_err_u = np.zeros(nq)
        # Increment replan cycles counter
        nb_plan += 1
        nb_ctrl += 1
        # Updtate OCP if necessary
        for k,m in enumerate(ddp.problem.runningModels[:]):
            m.differential.costs.costs["placement"].weight += (i/N_simu)*1e-1
    
    # Simulate actuation with PI torque tracking controller (low-level control frequency)
    if(INTERPOLATE_TORQUE):
      coef = float(i%20.)/19.
      u_des_interp = (1-coef)*u_des_prev + coef*u_des    # Initialize torque measurement to desired torque (perfect actuation) 
    else:
      u_des_interp = u_des
    # PI control law
    if(TORQUE_TRACKING):
      u_mea = U_mea[i, :] - u_des_interp                 # Initialize torque measurement to last measured torque (HF closed-loop)
      err_u = u_mea #U_mea[i, :] - u_des
      int_err_u += err_u
      vel_err_u = vel_u_mea - vel_u_des
      # print("DES : ", u_des)
      # print("MEA : ", U_mea[i, :])
      u_mea = u_des_interp - Kp.dot(err_u) - Ki.dot(int_err_u) - Kd.dot(vel_err_u)
    else:
      u_mea = u_des_interp
    # Scaling 
    if(SCALE_TORQUES):
      u_mea = alpha*u_mea + beta
    # Noise
    if(NOISE_TORQUES):
      u_mea += np.random.normal(0., var_u)
    # Filter out using moving average
    if(FILTER_TORQUES):
      n_sum = min(i, avg_length)
      for k in range(n_sum):
        u_mea += U_mea[i-k-1, :]
      u_mea = u_mea / (n_sum + 1)
    # Delay
    if(DELAY_TORQUES):
      buffer.append(u_mea)            # Add current desired torque to buffer
      if(len(buffer)<delay_cycle):    # If buffer too small use u_grav
        pass
      else:                           # Else pick an old desired control 
        u_mea = buffer.pop(-delay_cycle)
    # Record measured control
    U_mea[i+1, :] = u_mea
    # Estimate torque vel with 5 point rule
    if(i==0):
      vel_u_mea = (u_mea - U_mea[i, :]) / (dt_simu)
    else:
      vel_u_mea = (U_mea[i+1, :] - U_mea[i, :]) / (dt_simu)
      # vel_u_mea = (U_mea[i-4, :] - 8*U_mea[i-3, :] + U_mea[i-1, :] - U_mea[i, :]) / (12*dt_simu)
    # else:
    #   vel_u_mea = np.zeros(nq)
    vel_U_mea[i, :] = vel_u_mea
  
    # Step simulator
    pybullet_simulator.send_joint_command(U_mea[i+1, :])
    p.stepSimulation()
    # Measure new state from simulation 
    q_mea, v_mea = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea, v_mea)
    # Record data
    x_mea = np.concatenate([q_mea, v_mea]).T 
      # Optional noise
    if(NOISE_STATE):
      wq = np.random.normal(0., var_q, nq)
      wv = np.random.normal(0., var_v, nv)
      x_mea += np.concatenate([wq, wv]).T
    X_mea[i+1, :] = x_mea                    # Measured state
    # time.sleep(0.1)


print("ALPHA, BETA = ", alpha, beta)
####################################    
# GENERATE NICE PLOT OF SIMULATION #
####################################
PLOT_PREDICTIONS = False
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
# Reshape trajs if necessary 
q_pred = X_pred[:,:,:nq]
v_pred = X_pred[:,:,nv:]
q_mea = X_mea[:,:nq]
v_mea = X_mea[:,nv:]
q_des = X_des[:,:nq]
v_des = X_des[:,nv:]
p_mea = utils.get_p(q_mea, robot, id_endeff)
p_des = utils.get_p(q_des, robot, id_endeff) 
# Create time spans for X and U + Create figs and subplots
t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
t_span_simu_u = np.linspace(0, T_tot-dt_simu, N_simu)
t_span_ctrl_x = np.linspace(0, T_tot, N_ctrl+1)
t_span_ctrl_u = np.linspace(0, T_tot-dt_ctrl, N_ctrl)
fig_x, ax_x = plt.subplots(nq, 2)
fig_u, ax_u = plt.subplots(nq, 2)
fig_p, ax_p = plt.subplots(3,1)
# For each joint
for i in range(nq):
    # Extract state predictions of i^th joint
    q_pred_i = q_pred[:,:,i]
    v_pred_i = v_pred[:,:,i]
    u_pred_i = U_pred[:,:,i]
    # print(u_pred_i[0,0])
    if(PLOT_PREDICTIONS):
        # For each planning step in the trajectory
        for j in range(N_plan):
            # Receding horizon = [j,j+N_h]
            t0_horizon = j*dt_plan
            tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
            tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
            # Set up lists of (x,y) points for predicted positions and velocities
            points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
            points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
            points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
            # Set up lists of segments
            segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
            segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
            segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
            # Make collections segments
            cm = plt.get_cmap('Greys_r') 
            lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
            lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
            lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
            lc_q.set_array(tspan_x_pred)
            lc_v.set_array(tspan_x_pred) 
            lc_u.set_array(tspan_u_pred)
            # Customize
            lc_q.set_linestyle('-')
            lc_v.set_linestyle('-')
            lc_u.set_linestyle('-')
            lc_q.set_linewidth(1)
            lc_v.set_linewidth(1)
            lc_u.set_linewidth(1)
            # Plot collections
            ax_x[i,0].add_collection(lc_q)
            ax_x[i,1].add_collection(lc_v)
            ax_u[i].add_collection(lc_u)
            # Scatter to highlight points
            colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
            my_colors = cm(colors)
            ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
            ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
            ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
    

    # Desired joint position (interpolated from prediction)
    ax_x[i,0].plot(t_span_ctrl_x, q_des[:,i], 'b-', label='Desired')
    # Measured joint position (PyBullet)
    ax_x[i,0].plot(t_span_simu_x, q_mea[:,i], 'r-', label='Measured')
    ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{i}$ (rad)')
    ax_x[i,0].grid()

    # Desired joint velocity (interpolated from prediction)
    ax_x[i,1].plot(t_span_ctrl_x, v_des[:,i], 'b-', label='Desired')
    # Measured joint velocity (PyBullet)
    ax_x[i,1].plot(t_span_simu_x, v_mea[:,i], 'r-', label='Measured')
    ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{i}$ (rad/s)')
    ax_x[i,1].grid()

    # Desired joint torque (interpolated feedforward)
    ax_u[i, 0].plot(t_span_ctrl_u, U_des[:,i], 'b-', label='Desired')
    ax_u[i, 0].plot(t_span_ctrl_u, [u_grav[i]]*len(t_span_ctrl_u), 'k--', label='U_GRAV')
    ax_u[i, 0].plot(t_span_simu_u, [alpha[i]*U_mea[0,i] + beta[i]]*len(t_span_simu_u), 'g--', label='SCALING(u0)')
    # Total
    ax_u[i, 0].plot(t_span_simu_x, U_mea[:,i], 'r-', label='Measured') 
    ax_u[i, 0].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
    ax_u[i, 0].grid()

    # Desired joint torque (interpolated feedforward)
    ax_u[i, 1].plot(t_span_ctrl_u, vel_U_des[:,i], 'b-', label='Desired')
    ax_u[i, 1].plot(t_span_ctrl_u, [0]*len(t_span_ctrl_u), 'k--', label='0')
    # Total
    ax_u[i, 1].plot(t_span_simu_x, vel_U_mea[:,i], 'r-', label='Measured') 
    ax_u[i, 1].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
    ax_u[i, 1].grid()

    # Legend
    handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
    fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

    handles_u, labels_u = ax_u[i,0].get_legend_handles_labels()
    fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})


# Plot endeff
# x
ax_p[0].plot(t_span_ctrl_x, p_des[:,0], 'b-', label='x_des')
ax_p[0].plot(t_span_simu_x, p_mea[:,0], 'r-.', label='x_mea')
ax_p[0].set_title('x-position')
ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
ax_p[0].grid()
# y
ax_p[1].plot(t_span_ctrl_x, p_des[:,1], 'b-', label='y_des')
ax_p[1].plot(t_span_simu_x, p_mea[:,1], 'r-.', label='y_mea')
ax_p[1].set_title('y-position')
ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
ax_p[1].grid()
# z
ax_p[2].plot(t_span_ctrl_x, p_des[:,2], 'b-', label='z_des')
ax_p[2].plot(t_span_simu_x, p_mea[:,2], 'r-.', label='z_mea')
ax_p[2].set_title('z-position')
ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
ax_p[2].grid()
# Add frame ref if any
p_ref = desiredFramePlacement.translation
ax_p[0].plot(t_span_ctrl_x, [p_ref[0]]*(N_ctrl+1), 'ko', label='ref_pl', alpha=0.5)
ax_p[1].plot(t_span_ctrl_x, [p_ref[1]]*(N_ctrl+1), 'ko', label='ref_pl', alpha=0.5)
ax_p[2].plot(t_span_ctrl_x, [p_ref[2]]*(N_ctrl+1), 'ko', label='ref_pl', alpha=0.5)
handles_p, labels_p = ax_p[0].get_legend_handles_labels()
fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

# Titles
fig_x.suptitle('Joint trajectories: des. vs sim. (DDP-based MPC)', size=16)
fig_u.suptitle('Joint torques: des. vs sim. (DDP-based MPC)', size=16)
fig_p.suptitle('End-effector: ref. vs des. vs sim. (DDP-based MPC)', size=16)

plt.show() 