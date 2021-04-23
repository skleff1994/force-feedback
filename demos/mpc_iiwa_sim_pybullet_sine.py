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
The robot is tasked with creating contact between its end-effector and a wall and apply a constant normal force
Trajectory optimization using Crocoddyl in closed-loop MPC (feedback from state x=(q,v))
Using PyBullet simulator for rigid-body dynamics with contacts
Using PyBullet GUI for visualization
'''
# '''
# This demo file is used to compare several contact models for simulation, namely:
#     - PyBullet contact model
#     - Crocoddyl contact (rigid, i.e. no model --> only desired force based on predictions + KKT dynamics integration)
#     - Visco-elastic contact model (spring damper) integrated using Bilal's exponential integrator
# '''

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np  
import pinocchio as pin
from pinocchio import StdVec_Force
import crocoddyl
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot, IiwaConfig

from utils import utils

import pybullet as p
import time 


############################################
### ROBOT MODEL & SIMULATION ENVIRONMENT ###
############################################
#   # ROBOT 
# robot = IiwaConfig.buildRobotWrapper()
    # Create a Pybullet simulation environment
env = BulletEnvWithGround()
    # Create a robot instance. This initializes the simulator as well.
pybullet_simulator = env.add_robot(IiwaRobot)
robot = pybullet_simulator.pin_robot
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
    # Initial state 
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
dq0 = pin.utils.zero(nv)
    # Reset robot to initial state in PyBullet
pybullet_simulator.reset_state(q0, dq0)
    # Update pinocchio data accordingly 
pybullet_simulator.forward_robot(q0, dq0)
    # Get initial frame placement
M_ee = robot.data.oMf[id_endeff]
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
# print("Initial placement in WORLD frame : ")
# print(M_ee)
  # CONTACT
    # Set contact placement = M_ee with offset (cf. below)
M_ct = pin.SE3.Identity()
M_ct.rotation = M_ee.rotation 
offset = 0.03 + 0.003499998807875214  # 0.1 + 0.003499998807875214 
M_ct.translation = M_ee.act(np.array([0., 0., offset])) 
# print("Contact placement in WORLD frame : ")
# print(M_ct)

# Measure distance EE to contact surface using p.getContactPoints() 
# in order to avoid PyB repulsion due to penetration 
# Result = 0.03 + 0.003499998807875214. Problem : smaller than ball radius (changed urdf?) . 
contactId = utils.display_contact_surface(M_ct, pybullet_simulator.robotId, with_collision=True)
print("[PyBullet] Created contact plane (id = "+str(contactId)+")")
print("[PyBullet] Contact placement in WORLD frame : ")
print(M_ct)

#################
### OCP SETUP ###
#################
  # OCP parameters 
dt = 2e-2                      # OCP integration step (s)               
N_h = 50                       # Number of knots in the horizon 
x0 = np.concatenate([q0, dq0])  # Initial state
print("Initial state : ", x0.T)
  # Construct cost function terms
   # State and actuation models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
   # State regularization
stateRegWeights = np.array([1.]*nq + [2.]*nv)  
x_reg_ref = x0 #np.zeros(nq+nv)     
xRegCost = crocoddyl.CostModelState(state, 
                                    crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                    x_reg_ref, 
                                    actuation.nu)
print("Created state reg cost.")
   # Control regularization
ctrlRegWeights = np.ones(nq)
u_grav = pin.rnea(robot.model, robot.data, x0[:nq], np.zeros((nv,1)), np.zeros((nq,1)))
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
u_min = -np.array([320, 320, 176, 176, 110, 40, 40])
u_max = np.array([320, 320, 176, 176, 110, 40, 40])
u_lim_ref = np.zeros(nq)
uLimitCost = crocoddyl.CostModelControl(state, 
                                        crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                        u_lim_ref)
print("Created ctrl lim cost.")
   # End-effector contact force
desiredFrameForce = pin.Force(np.array([0., 0., -10., 0., 0., 0.]))
frameForceWeights = np.array([1.]*3 + [1.]*3)  
frameForceCost = crocoddyl.CostModelContactForce(state, 
                                                 crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                 crocoddyl.FrameForce(id_endeff, desiredFrameForce), 
                                                 actuation.nu) 
print("Created frame force cost.")
   # End-effector velocity 
desiredFrameMotion = pin.Motion(np.array([0.,0.,0.,0.,0.,0.]))
frameVelocityWeights = np.ones(6)
frameVelocityCost = crocoddyl.CostModelFrameVelocity(state, 
                                                     crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                     crocoddyl.FrameMotion(id_endeff, desiredFrameMotion), 
                                                     actuation.nu) 
print("Created frame velocity cost.")
   # End-effector placement 
desiredFramePlacement = M_ct #robot.data.oMf[id_endeff] #M_ct
framePlacementWeights = np.ones(6)
framePlacementCost = crocoddyl.CostModelFramePlacement(state, 
                                                       crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                       crocoddyl.FramePlacement(id_endeff, desiredFramePlacement), 
                                                       actuation.nu) 
print("Created frame placement cost.")
# Contact model
ref_placement = crocoddyl.FramePlacement(id_endeff, M_ct) #robot.data.oMf[id_endeff]) #pin.SE3.Identity()) #pin_robot.data.oMf[id_endeff])
contact6d = crocoddyl.ContactModel6D(state, ref_placement, gains=np.array([50.,10.]))

# Friction cone 
cone_rotation = robot.data.oMf[id_endeff].rotation.T
nsurf = cone_rotation.dot(np.matrix(np.array([0, 0, 1])).T)
mu = 0.7
# nsurf, mu = np.array([0.,0.,1.]), 0.7
frictionCone = crocoddyl.FrictionCone(nsurf, mu, 4, True, 0, 1000) #2000 ?
frictionConeCost = crocoddyl.CostModelContactFrictionCone(state,
                                                          crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(frictionCone.lb , frictionCone.ub)),
                                                          crocoddyl.FrameFrictionCone(id_endeff, frictionCone),
                                                          actuation.nu)

# Create IAMs
runningModels = []
for i in range(N_h):
  # Create IAM 
  runningModels.append(crocoddyl.IntegratedActionModelEuler( 
      crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                          actuation, 
                                                        #   contactModel, 
                                                          crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                          crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                          inv_damping=0., 
                                                          enable_force=True), dt) )
  # Add cost models
  runningModels[i].differential.costs.addCost("placement", framePlacementCost, 1.) 
  runningModels[i].differential.costs.addCost("force", frameForceCost, 1., active=False) 
  runningModels[i].differential.costs.addCost("frictionCone", frictionConeCost, 5e-2, active=False) 
  runningModels[i].differential.costs.addCost("stateReg", xRegCost, 1e-4) 
  runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, 1e-3)
  runningModels[i].differential.costs.addCost("stateLim", xLimitCost, 10) 
  runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, 1e-1) 
  # Add armature
  runningModels[i].differential.armature = np.array([.1]*7)
  # Add contact models
  runningModels[i].differential.contacts.addContact("contact", contact6d, active=False)
# Terminal IAM + set armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                        actuation, 
                                                        # contactModel,
                                                        crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                        inv_damping=0., 
                                                        enable_force=True) )
# Add cost models
terminalModel.differential.costs.addCost("placement", framePlacementCost, 0) #1e6) 
terminalModel.differential.costs.addCost("force", frameForceCost, 1., active=False)
terminalModel.differential.costs.addCost("stateReg", xRegCost, 1e-3) 
terminalModel.differential.costs.addCost("stateLim", xLimitCost, 10) 
# Add armature
terminalModel.differential.armature = np.array([.1]*7)
# Add contact model
terminalModel.differential.contacts.addContact("contact", contact6d, active=False)

print("Initialized IAMs.")
print("Running IAM cost.active  = ", runningModels[0].differential.costs.active.tolist())
print("Terminal IAM cost.active = ", terminalModel.differential.costs.active.tolist())

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
maxit = 1
T_tot = 5.
plan_freq = 1000                      # MPC re-planning frequency (Hz)
ctrl_freq = 1000                      # Control - simulation - frequency (Hz)
N_tot = int(T_tot*ctrl_freq)          # Total number of control steps in the simulation (s)
N_p = int(T_tot*plan_freq)            # Total number of OCPs (replan) solved during the simulation
T_h = N_h*dt                          # Duration of the MPC horizon (s)
# Initialize data
nx = nq+nv
nu = nq
X_mea = np.zeros((N_tot+1, nx))       # Measured states 
X_des = np.zeros((N_tot+1, nx))       # Desired states
U_des = np.zeros((N_tot, nu))         # Desired controls 
X_pred = np.zeros((N_p, N_h+1, nx))   # MPC predictions (state)
U_pred = np.zeros((N_p, N_h, nu))     # MPC predictions (control)
U_des = np.zeros((N_tot, nq))         # Feedforward torques planned by MPC (DDP) 
U_mea = np.zeros((N_tot, nq))         # Torques sent to PyBullet
contact_des = [False]*X_des.shape[0]                # Contact record for contact force
contact_mea = [False]*X_mea.shape[0]                # Contact record for contact force
contact_pred = np.zeros((N_p, N_h+1), dtype=bool)   # Contact record for contact force
F_des = np.zeros((N_tot, 6))        # Contact force computed with pinocchio (should be the same as desired)
F_mea_pyb = np.zeros((N_tot, 6))    # PyBullet measurement of contact force 
F_mea = np.zeros((N_tot, 6))        # Contact force calculated with Pinocchio from PyBullet joint measurements
F_pred = np.zeros((N_p, N_h, 6))    # MPC prediction of contact force (same as desired)
F_ref = np.zeros((N_tot, 6))        # Reference contact force
# Consim stuff
F_mea_consim = np.zeros((N_tot, 6)) # Contact force measured in ConSim (exponential integrator)
X_mea_consim = np.zeros((N_tot+1, nx)) # Measured state from ConSim 
# Logs
print('                  ************************')
print('                  * MPC controller ready *') 
print('                  ************************')        
print('---------------------------------------------------------')
print('- Total simulation duration            : T_tot  = '+str(T_tot)+' s')
print('- Control frequency                    : f_ctrl = '+str(ctrl_freq)+' Hz')
print('- Replanning frequency                 : f_plan = '+str(plan_freq)+' Hz')
print('- Total # of control steps             : N_tot  = '+str(N_tot))
print('- Duration of MPC horizon              : T_ocp  = '+str(T_h)+' s')
print('- Total # of replanning knots          : N_p    = '+str(N_p))
print('- OCP integration step                 : dt     = '+str(dt)+' s')
print('---------------------------------------------------------')
print("Simulation will start...")
time.sleep(1)

# Measure initial state from simulation environment &init data
q_mea, v_mea = pybullet_simulator.get_state()
pybullet_simulator.forward_robot(q_mea, v_mea)
x0 = np.concatenate([q_mea, v_mea]).T
print("Initial state ", str(x0))
X_mea[0, :] = x0
X_des[0, :] = x0

# For desired sine force
min_force = -10
max_force = -5
offset = (min_force + max_force)/2
amplitude = offset - min_force
freq = 1. 
print("max / min    = ", max_force, min_force)
print("offset / amp = ", offset, amplitude)
print("frequency = ", freq)
# time.sleep(10)
# Replan counter
nb_replan = 0
# SIMULATION LOOP
switch=False
for i in range(N_tot): 
    print("  ")
    print("Sim step "+str(i)+"/"+str(N_tot))
    # Solve OCP if we are in a planning cycle
    if(i%int(ctrl_freq/plan_freq) == 0):
        print("  Replan step "+str(nb_replan)+"/"+str(N_p))
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = X_mea[i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = X_mea[i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        ### HERE UPDATE OCP AS NEEDED ####
        normal_force = offset + (offset - min_force)*np.sin(2*np.pi*freq*i*1e-3)
        # STATE-based switch
        if(len(p.getContactPoints(pybullet_simulator.robotId, contactId))>0 and switch==False):
            switch=True
            ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.Mref.placement =  robot.data.oMf[id_endeff]
            ddp.problem.terminalModel.differential.contacts.changeContactStatus("contact", True)
            ddp.problem.terminalModel.differential.costs.changeCostStatus("force", True)
            normal_force_t = offset + (offset - min_force)*np.sin(2*np.pi*freq*(i*1e-3 + N_h*dt))
            ddp.problem.terminalModel.differential.costs.costs["force"].cost.reference = crocoddyl.FrameForce(id_endeff, pin.Force(np.array([0., 0., normal_force_t, 0., 0., 0.])))
            ddp.problem.terminalModel.differential.costs.costs["force"].weight = 100

            for k,m in enumerate(ddp.problem.runningModels[:]):
                # Activate contact and force cost
                m.differential.contacts.contacts["contact"].contact.Mref.placement =  robot.data.oMf[id_endeff]
                m.differential.contacts.changeContactStatus("contact", True)
                m.differential.costs.changeCostStatus("force", True)
                m.differential.costs.costs["force"].weight = 20.

                # Reference force trajectory  = sine signal between [-10,-5] at 0.5 Hz
                normal_force_k = offset + (offset - min_force)*np.sin(2*np.pi*freq*(i*1e-3+k*dt))
                new_ref_force = crocoddyl.FrameForce(id_endeff, pin.Force(np.array([0., 0., normal_force_k, 0., 0., 0.])))
                m.differential.costs.costs["force"].cost.reference = new_ref_force

                # m.differential.costs.changeCostStatus("frictionCone", True)
                # m.differential.costs.costs["frictionCone"].weight = 1
                
                # Update state reg cost
                m.differential.costs.costs["stateReg"].reference = xs_init[0]
                m.differential.costs.costs["stateReg"].weight = 0
                # Update control reg cost
                m.differential.costs.costs["ctrlReg"].reference = us_init[0]
                m.differential.costs.costs["ctrlReg"].weight = 1.

        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=maxit, isFeasible=False)
        X_pred[nb_replan, :, :] = np.array(ddp.xs)
        U_pred[nb_replan, :, :] = np.array(ddp.us)
        for j in range(N_h):
            F_pred[nb_replan, j, :] = ddp.problem.runningDatas[j].differential.multibody.contacts.contacts['contact'].f.vector
        # Extract 1st control and 2nd state
        u_des = U_pred[nb_replan, 0, :] 
        x_des = X_pred[nb_replan, 1, :]
        f_des = F_pred[nb_replan, 0, :]
        f_ref = np.array([0., 0., normal_force, 0., 0., 0.])
        # Increment replan counter
        nb_replan += 1

    # Record and apply the 1st control
    U_des[i, :] = u_des
    # Send control to simulation & step simulator # here plug : PyBullet or Consim
    pybullet_simulator.send_joint_command(u_des)
    p.stepSimulation()
    # Measure new state from simulation and record data
    q_mea, v_mea = pybullet_simulator.get_state()
    pybullet_simulator.forward_robot(q_mea, v_mea)
    x_mea = np.concatenate([q_mea, v_mea]).T 
    
    # Measure contact force in PyBullet    
    ids, forces = pybullet_simulator.get_force()
    # Express in local EE frame (minus because force env-->robot)
    if(switch==True):
        F_mea_pyb[i,:] = -robot.data.oMf[id_endeff].actionInverse.dot(forces[0])
    else:
        pass
    print("    [PyBullet] = ", F_mea_pyb[i,:])
    # FD estimate of joint accelerations
    if(i==0):
      a_mea = np.zeros(nq)
    else:
      a_mea = (v_mea - X_mea[i,nq:nq+nv])/1e-3
    # ID
    f = StdVec_Force()
    for j in range(robot.model.njoints):
      f.append(pin.Force.Zero())
    f[-1].linear = F_mea_pyb[i,:3]
    f[-1].angular = F_mea_pyb[i,3:]
    # Get corresponding measured torque 
    tau_mea = pin.rnea(robot.model, robot.data, q_mea, v_mea, a_mea, f)

    X_mea[i+1, :] = x_mea                    # Measured state
    X_des[i+1, :] = x_des                    # Desired state
    F_des[i, :] = f_des                      # Desired force
    F_ref[i, :] = f_ref
    U_mea[i, :] = tau_mea
    
# GENERATE NICE PLOT OF SIMULATION
with_predictions = False
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
# Time step duration of the control loop
dt_ctrl = float(1./ctrl_freq)
# Time step duration of planning loop
dt_plan = float(1./plan_freq)
# Reshape trajs if necessary 
q_pred = X_pred[:,:,:nq]
v_pred = X_pred[:,:,nv:]
q_mea = X_mea[:,:nq]
v_mea = X_mea[:,nv:]
q_des = X_des[:,:nq]
v_des = X_des[:,nv:]
p_mea = utils.get_p(q_mea, robot, id_endeff)
p_des = utils.get_p(q_des, robot, id_endeff) 
# Compute with Pinocchio from measured q,v in order to compare with PyBullet solution
f_mea = utils.get_f(q_mea, v_mea, U_mea, robot, id_endeff, dt=1e-3) 
# COnSim
q_mea_consim = X_mea_consim[:,:nq]
v_mea_consim = X_mea_consim[:,nv:]
p_mea_consim = utils.get_p(q_mea_consim, robot, id_endeff)
# Create time spans for X and U + Create figs and subplots
tspan_x = np.linspace(0, T_tot, N_tot+1)
tspan_u = np.linspace(0, T_tot-dt_ctrl, N_tot)
fig_x, ax_x = plt.subplots(nq, 2)
fig_u, ax_u = plt.subplots(nq, 1)
fig_p, ax_p = plt.subplots(3,1)
# For each joint
for i in range(nq):
    # Extract state predictions of i^th joint
    q_pred_i = q_pred[:,:,i]
    v_pred_i = v_pred[:,:,i]
    u_pred_i = U_pred[:,:,i]
    # print(u_pred_i[0,0])
    if(with_predictions):
        # For each planning step in the trajectory
        for j in range(N_p):
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
    ax_x[i,0].plot(tspan_x, q_des[:,i], 'b-', label='Desired')
    # Measured joint position (PyBullet)
    ax_x[i,0].plot(tspan_x, q_mea[:,i], 'r-', label='Measured')
    ax_x[i,0].plot(tspan_x, q_mea_consim[:,i], 'g-.', label='Measured (EI)')
    ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{i}$ (rad)')
    ax_x[i,0].grid()

    # Desired joint velocity (interpolated from prediction)
    ax_x[i,1].plot(tspan_x, v_des[:,i], 'b-', label='Desired')
    # Measured joint velocity (PyBullet)
    ax_x[i,1].plot(tspan_x, v_mea[:,i], 'r-', label='Measured')
    ax_x[i,1].plot(tspan_x, v_mea_consim[:,i], 'g-.', label='Measured (EI)')
    ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{i}$ (rad/s)')
    ax_x[i,1].grid()

    # Desired joint torque (interpolated feedforward)
    ax_u[i].plot(tspan_u, U_des[:,i], 'b-', label='Desired (ff)')
    # Total
    ax_u[i].plot(tspan_u, U_mea[:,i], 'r-', label='Measured (ff+fb)') 
    ax_u[i].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
    ax_u[i].grid()

    # Legend
    handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
    fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

    handles_u, labels_u = ax_u[i].get_legend_handles_labels()
    fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

# Get desired and measured contact forces
# f_ref = desiredFrameForce.vector
fig_f, ax_f = plt.subplots(6,1)
# Plot contact force
for i in range(6):
    ax_f[i].plot(tspan_u, F_des[:,i], 'b-', label='Desired (Crocoddyl)')
    # ax_f[i].plot(tspan_u, f_mea[:,i], 'g-.', label='Measured (Pinocchio)', alpha=0.3)
    ax_f[i].plot(tspan_u, F_mea_pyb[:,i], 'r-', label='Measured (PyBullet)', alpha=0.5)
    # ax_f[i].plot(tspan_u, F_mea_consim[:,i], 'g-.', label='Measured (EI)', alpha=0.3)
    ax_f[i].plot(tspan_u, F_ref[:,i], 'k--', label='reference', alpha=0.5)
    ax_f[i].set(xlabel='t (s)', ylabel='$f_{i}$ (N)')
    ax_f[i].grid()
    # Legend
    handles_f, labels_f = ax_f[i].get_legend_handles_labels()
    fig_f.legend(handles_f, labels_f, loc='upper right', prop={'size': 16})

# Compute predicted force using predicted trajs
if(with_predictions):
    # For dim
    for i in range(6):
        # Extract state predictions of i^th dim
        f_pred_i = F_pred[:, :, i]
        # For each planning step in the trajectory
        for j in range(N_p):
            # Receding horizon = [j,j+N_h]
            # Receding horizon = [j,j+N_h]
            t0_horizon = j*dt_plan
            tspan_f_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h) #np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
            # Set up lists of (x,y) points for predicted positions and velocities
            points_f = np.array([tspan_f_pred, f_pred_i[j,:]]).transpose().reshape(-1,1,2)
            # Set up lists of segments
            segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
            # Make collections segments
            cm = plt.get_cmap('Greys_r') 
            lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
            lc_f.set_array(tspan_f_pred)
            # Customize
            lc_f.set_linestyle('-')
            lc_f.set_linewidth(1)
            # Plot collections
            ax_f[i].add_collection(lc_f)
            # Scatter to highlight points
            colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
            my_colors = cm(colors)
            # ax_f[i].scatter(tspan_f_pred, f_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black' 
            ax_f[i].scatter(tspan_f_pred, f_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
    

# Plot endeff
# x
ax_p[0].plot(tspan_x, p_des[:,0], 'b-', label='x_des')
ax_p[0].plot(tspan_x, p_mea[:,0], 'r-.', label='x_mea')
ax_p[0].set_title('x-position')
ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
ax_p[0].grid()
# y
ax_p[1].plot(tspan_x, p_des[:,1], 'b-', label='y_des')
ax_p[1].plot(tspan_x, p_mea[:,1], 'r-.', label='y_mea')
ax_p[1].set_title('y-position')
ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
ax_p[1].grid()
# z
ax_p[2].plot(tspan_x, p_des[:,2], 'b-', label='z_des')
ax_p[2].plot(tspan_x, p_mea[:,2], 'r-.', label='z_mea')
ax_p[2].set_title('z-position')
ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
ax_p[2].grid()
# Add frame ref if any
p_ref = desiredFramePlacement.translation
ax_p[0].plot(tspan_x, [p_ref[0]]*(N_tot+1), 'ko', label='ref_contact', alpha=0.5)
ax_p[1].plot(tspan_x, [p_ref[1]]*(N_tot+1), 'ko', label='ref_contact', alpha=0.5)
ax_p[2].plot(tspan_x, [p_ref[2]]*(N_tot+1), 'ko', label='ref_contact', alpha=0.5)
handles_p, labels_p = ax_p[0].get_legend_handles_labels()
fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

# Titles
fig_x.suptitle('Joint trajectories: des. vs sim. (DDP-based MPC)', size=16)
fig_u.suptitle('Joint torques: des. vs sim. (DDP-based MPC)', size=16)
fig_p.suptitle('End-effector: ref. vs des. vs sim. (DDP-based MPC)', size=16)
fig_f.suptitle('End-effector force', size=16)

plt.show() 


# #################################
# ### EXTRACT SOLUTION AND PLOT ###
# #################################
# print("Extracting solution...")
# # Extract solution trajectories
# q = np.empty((N_h+1, nq))
# v = np.empty((N_h+1, nv))
# p_ee = np.empty((N_h+1, 3))
# for i in range(N_h+1):
#     q[i,:] = xs[i][:nq].T
#     v[i,:] = xs[i][nv:].T
#     pin.forwardKinematics(robot.model, robot.data, q[i])
#     pin.updateFramePlacements(robot.model, robot.data)
#     p_ee[i,:] = robot.data.oMf[id_endeff].translation.T
# u = np.empty((N_h, actuation.nu))
# for i in range(N_h):
#     u[i,:] = us[i].T
# # Estimate joint accelerations from desired joint velocities using finite differences
# a = np.zeros((N_h+1, nq))
# for i in range(N_h+1):
#     if i>0:
#         a[i,:] = (v[i,:] - v[i-1,:])/dt
# # Calculate contact force from desired trajectories
# f = np.empty((N_h, 6))
# for i in range(N_h):
#     # Much simpler
#     f[i,:] = ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector
# print("Plot results...")
# import matplotlib.pyplot as plt #; plt.ion()
# # Create time spans for X and U + figs and subplots
# tspan_x = np.linspace(0, N_h*dt, N_h+1)
# tspan_u = np.linspace(0, N_h*dt, N_h)
# fig_x, ax_x = plt.subplots(nq, 3)
# fig_u, ax_u = plt.subplots(nq, 1)
# fig_p, ax_p = plt.subplots(3,1)
# fig_f, ax_f = plt.subplots(6, 1)
# # Plot joints pos, vel , acc, torques
# for i in range(nq):
#     # Positions
#     ax_x[i,0].plot(tspan_x, q[:,i], 'b.', label='desired')
#     ax_x[i,0].plot(tspan_x[-1], q[-1,i], 'ro')
#     ax_x[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
#     ax_x[i,0].grid()
#     # Velocities
#     ax_x[i,1].plot(tspan_x, v[:,i], 'b.', label='desired')
#     ax_x[i,1].plot(tspan_x[-1], v[-1,i], 'ro')
#     ax_x[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
#     ax_x[i,1].grid()
#     # Accelerations
#     ax_x[i,2].plot(tspan_x, a[:,i], 'b.', label='desired')
#     ax_x[i,2].plot(tspan_x[-1], a[-1,i], 'ro')
#     ax_x[i,2].set_ylabel('$a_%s$'%i, fontsize=16)
#     ax_x[i,2].grid()
#     # Torques
#     ax_u[i].plot(tspan_u, u[:,i], 'b.', label='desired') # feedforward term
#     ax_u[i].set_ylabel(ylabel='$u_%d$'%i, fontsize=16)
#     ax_u[i].grid()
#     # Remove xticks labels for clarity 
#     if(i != nq-1):
#         for j in range(3):
#             ax_x[i,j].set_xticklabels([])
#         ax_u[i].set_xticklabels([])
#     # Set xlabel on bottom plot
#     if(i == nq-1):
#         for j in range(3):
#             ax_x[i,j].set_xlabel('t (s)', fontsize=16)
#         ax_u[i].set_xlabel('t (s)', fontsize=16)
#     # Legend
#     handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
#     fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
#     handles_u, labels_u = ax_u[i].get_legend_handles_labels()
#     fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
# # Plot contact force
# f_ref = desiredFrameForce.vector
# ylabels_f = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
# for i in range(6):
#     ax_f[i].plot(tspan_u, [f_ref[i]]*N_h, 'ro', label='REF', alpha=0.5)
#     ax_f[i].plot(tspan_u, f[:,i], 'b.', label='desired')
#     ax_f[i].set_ylabel(ylabel=ylabels_f[i], fontsize=16)
#     ax_f[i].grid()
#     # Legend
#     handles_f, labels_f = ax_f[i].get_legend_handles_labels()
#     fig_f.legend(handles_f, labels_f, loc='upper right', prop={'size': 16})
# ax_f[-1].set_xlabel('t (s)', fontsize=16)
# # Plot EE
# ylabels_p = ['Px', 'Py', 'Pz']
# p_contact = np.array([M_ct.translation]*(N_h+1))
# for i in range(3):
#     ax_p[i].plot(tspan_x, p_ee[:,i], 'b.', label='desired')
#     ax_p[i].plot(tspan_x, p_contact[:,i], 'ro', label='REF', alpha=0.5)
#     ax_p[i].set_ylabel(ylabel=ylabels_p[i], fontsize=16)
#     ax_p[i].grid()
#     handles_p, labels_p = ax_p[i].get_legend_handles_labels()
#     fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
# ax_p[-1].set_xlabel('t (s)', fontsize=16)
# # Align labels + set titles
# fig_x.align_ylabels()
# fig_u.align_ylabels()
# fig_f.align_ylabels()
# fig_p.align_ylabels()
# fig_x.suptitle('Joint trajectories', size=16)
# fig_u.suptitle('Joint torques', size=16)
# fig_f.suptitle('End-effector force', size=16)
# fig_p.suptitle('End-effector trajectory', size=16)
# plt.show()
