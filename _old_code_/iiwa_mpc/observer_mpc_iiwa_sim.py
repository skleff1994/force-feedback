# Title : test_IAMiiwaLPF.py
# Author: Sebastien Kleff
# Date : 03.03.2020 
# Copyright LAAS-CNRS, NYU

# DDP-based MPC with force feedback using 'actuation dynamics' approach for KUKA Arm 

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import pinocchio as pin
import crocoddyl
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot, IiwaConfig
from core.kalman_filter import ExtendedKalmanFilter
import core_mpc

import pybullet as p
import time 

############################################
### ROBOT MODEL & SIMULATION ENVIRONMENT ###
############################################
  # ROBOT 
    # Create a Pybullet simulation environment + create robot instance
env = BulletEnvWithGround()
robot = env.add_robot(IiwaRobot)
id_endeff = robot.pin_robot.model.getFrameId('contact')
nq = robot.pin_robot.model.nq 
nv = robot.pin_robot.model.nv
    # Reset robot to initial state in PyBullet and update pinocchio data accordingly 
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
dq0 = pin.utils.zero(nv)
robot.reset_state(q0, dq0)
robot.forward_robot(q0, dq0)
    # Get initial frame placement
M_ee = robot.pin_robot.data.oMf[id_endeff]
print("[PyBullet] Created robot (id = "+str(robot.robotId)+")")
print("Initial placement in WORLD frame : ")
print(M_ee)
  # CONTACT
    # Set contact placement = M_ee with offset (cf. below)
M_ct = pin.SE3.Identity()
M_ct.rotation = M_ee.rotation 
offset = 0.1 + 0.003499998807875214 
M_ct.translation = M_ee.act(np.array([0., 0., offset])) 
print("Contact placement in WORLD frame : ")
print(M_ct)

# Measure distance EE to contact surface using p.getContactPoints() 
# in order to avoid PyB repulsion due to penetration 
# Result = 0.03 + 0.003499998807875214. Problem : smaller than ball radius (changed urdf?) . 
contactId = core_mpc.display_contact_surface(M_ct, robot.robotId, with_collision=True)
print("[PyBullet] Created contact plane (id = "+str(contactId)+")")
print("[PyBullet]   >> Detect contact points : ")
p.stepSimulation()
contact_points = p.getContactPoints(1, 2)
for k,i in enumerate(contact_points):
  print("      Contact point n°"+str(k)+" : distance = "+str(i[8])+" (m) | force = "+str(i[9])+" (N)")
# time.sleep(100)


#################
### OCP SETUP ###
#################
  # OCP parameters 
dt = 2e-2                       # OCP integration step (s)               
N_h = 50                        # Number of knots in the horizon 
x0 = np.concatenate([q0, dq0])  # Initial state
print("Initial state : ", x0.T)
  # Construct cost function terms
   # State and actuation models
state = crocoddyl.StateMultibody(robot.pin_robot.model)
actuation = crocoddyl.ActuationModelFull(state)
   # State regularization
stateRegWeights = np.array([1.]*nq + [2.]*nv)  
x_reg_ref = x0
xRegCost = crocoddyl.CostModelState(state, 
                                    crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                    x_reg_ref, 
                                    actuation.nu)
print("Created state reg cost.")
   # Control regularization
ctrlRegWeights = np.ones(nq)
u_reg_ref =  pin.rnea(robot.pin_robot.model, robot.pin_robot.data, q0, np.zeros((nv,1)), np.zeros((nq,1)))
uRegCost = crocoddyl.CostModelControl(state, 
                                      crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                      u_reg_ref)
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
u_lim_ref = pin.rnea(robot.pin_robot.model, robot.pin_robot.data, q0, np.zeros((nv,1)), np.zeros((nq,1)))
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
desiredFramePlacement = robot.pin_robot.data.oMf[id_endeff] #M_ct
framePlacementWeights = np.ones(6)
framePlacementCost = crocoddyl.CostModelFramePlacement(state, 
                                                       crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                       crocoddyl.FramePlacement(id_endeff, desiredFramePlacement), 
                                                       actuation.nu) 
print("Created frame placement cost.")
# Contact model
ref_placement = crocoddyl.FramePlacement(id_endeff, robot.pin_robot.data.oMf[id_endeff]) #pin.SE3.Identity()) #pin_robot.data.oMf[id_endeff])
contact6d = crocoddyl.ContactModel6D(state, ref_placement, gains=np.array([50.,10.]))

# Create IAMs
runningModels = []
for i in range(N_h):
  # Create IAM 
  runningModels.append(crocoddyl.IntegratedActionModelEuler( 
      crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                          actuation, 
                                                          crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                          crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                          inv_damping=0., 
                                                          enable_force=True), dt) )
  # Add cost models
  runningModels[i].differential.costs.addCost("placement", framePlacementCost, 10.) 
  runningModels[i].differential.costs.addCost("force", frameForceCost, 1., active=True) 
  runningModels[i].differential.costs.addCost("stateReg", xRegCost, 1e-3) 
  runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, 1e-2)
  runningModels[i].differential.costs.addCost("stateLim", xLimitCost, 10) 
  runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, 1e-1) 
  # Add armature
  runningModels[i].differential.armature = np.array([.1]*7)
  # Add contact models
  runningModels[i].differential.contacts.addContact("contact", contact6d, active=True)
# Terminal IAM + set armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                        actuation, 
                                                        crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                        inv_damping=0., 
                                                        enable_force=True) )
# Add cost models
terminalModel.differential.costs.addCost("placement", framePlacementCost, 1e6) 
terminalModel.differential.costs.addCost("force", frameForceCost, 1., active=True)
terminalModel.differential.costs.addCost("stateReg", xRegCost, 1e-3) 
terminalModel.differential.costs.addCost("stateLim", xLimitCost, 10) 
# Add armature
terminalModel.differential.armature = np.array([.1]*7)
# Add contact model
terminalModel.differential.contacts.addContact("contact", contact6d, active=True)

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
#     pin.forwardKinematics(robot.pin_robot.model, robot.pin_robot.data, q[i])
#     pin.updateFramePlacements(robot.pin_robot.model, robot.pin_robot.data)
#     p_ee[i,:] = robot.pin_robot.data.oMf[id_endeff].translation.T
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



# TEST FILTERING
# Create the filter 
Q_cov = 0.01*np.eye(nq+nv)   # Process noise cov
R_cov = np.eye(nq+3)         # Measurement noise cov
kalman = ExtendedKalmanFilter(Q_cov, R_cov)
# Observation model (spring-damper )
K = 1000. 
B = 2*np.sqrt(K)
p0 = 0.
# Add noise on DDP trajectory and filter it to test Kalman filter
nx = nq+nv
ny = 3+nq
Y_mea = np.zeros((N_h, ny))      # measurements
X_hat = np.zeros((N_h+1, nx))      # state estimates
P_cov = np.zeros((N_h+1, nx, nx))  # covariance estimates
K_gain = np.zeros((N_h+1, nx, nx)) # optimal Kalman gains
Y_err = np.zeros((N_h+1, ny))      
X_real = np.reshape(xs, X_hat.shape) # Ground truth state trajectory
# print(X_real)
p_real = core_mpc.get_p(X_real[:,:nq], robot.pin_robot, id_endeff)
v_real = core_mpc.get_v(X_real[:,:nq], X_real[:,:nv], robot.pin_robot, id_endeff)
# Measurement noise model
mean = np.zeros(2)
std_p = .05 #np.array([0.005, N_h])
std_f = 50   #np.array([0.005, N_h])


# ESTIMATION LOOP (offline)
for i in range(N_h):
    print("Step "+str(i)+"/"+str(N_h))
    # Generate noisy force measurement 
      # Ideal visco-elastic force and real position + Noise them out
    wp,_ = np.random.normal(mean, std_p) 
    wf,_ = np.random.normal(mean, std_f)

    # Get p(q) from pinocchio data
    Y_mea[i,:] = (np.array([X_real[i,:nq], -K*(p_real[i] - M_ct.translation) - B*(v_real[i])]) + np.array([wp, wf]) )
    # Get partials
    data = ddp.problem.runningModels[i].createData()
    ddp.problem.runningModels[i].calcDiff(data, X_real[i,:], )
    # Filter and record
    X_hat[i+1,:], P_cov[i+1,:,:], K_gain[i+1,:,:], Y_err[i+1,:] = kalman.step(X_hat[i,:], P_cov[i,:,:], U[i,:], Y_mea[i,:])

# Display Kalman gains magnitude
dP_dP = np.vstack(( np.array([[K_gain[i,0,0] for i in range(N_h)]]).transpose())) 
dP_dF = np.vstack(( np.array([[K_gain[i,0,1] for i in range(N_h)]]).transpose())) 
dV_dP = np.vstack(( np.array([[K_gain[i,1,0] for i in range(N_h)]]).transpose())) 
dV_dF = np.vstack(( np.array([[K_gain[i,1,1] for i in range(N_h)]]).transpose())) 
# Norms
print("dP_dP Kalman gain norm : ", np.linalg.norm(dP_dP))
print("dP_dF Kalman gain norm : ", np.linalg.norm(dP_dF))
print("dV_dP Kalman gain norm : ", np.linalg.norm(dV_dP))
print("dV_dF Kalman gain norm : ", np.linalg.norm(dV_dF))


# ##################
# # MPC SIMULATION #
# ##################
# # MPC & simulation parameters
# maxit = 1
# T_tot = 2.
# plan_freq = 1000                      # MPC re-planning frequency (Hz)
# ctrl_freq = 1000                      # Control - simulation - frequency (Hz)
# N_tot = int(T_tot*ctrl_freq)          # Total number of control steps in the simulation (s)
# N_p = int(T_tot*plan_freq)            # Total number of OCPs (replan) solved during the simulation
# T_h = N_h*dt                          # Duration of the MPC horizon (s)
# # Initialize data
# nx = nq+nv
# nu = nq
# X_mea = np.zeros((N_tot+1, nx))       # Measured states 
# X_des = np.zeros((N_tot+1, nx))       # Desired states
# U_des = np.zeros((N_tot, nu))         # Desired controls 
# X_pred = np.zeros((N_p, N_h+1, nx))   # MPC predictions (state)
# U_pred = np.zeros((N_p, N_h, nu))     # MPC predictions (control)
# U_des = np.zeros((N_tot, nq))         # Feedforward torques planned by MPC (DDP) 
# U_mea = np.zeros((N_tot, nq))         # Torques sent to PyBullet
# contact_des = [False]*X_des.shape[0]                # Contact record for contact force
# contact_mea = [False]*X_mea.shape[0]                # Contact record for contact force
# contact_pred = np.zeros((N_p, N_h+1), dtype=bool)   # Contact record for contact force
# F_des = np.zeros((N_tot, 6))        # Desired contact force
# F_pin = np.zeros((N_tot, 6))        # Contact force computed with pinocchio (should be the same as desired)
# F_pred = np.zeros((N_p, N_h, 6))    # MPC prediction of contact force (same as pin on desired trajs)
# F_mea = np.zeros((N_tot, 6))        # PyBullet measurement of contact force (? at which contact point ?)
# # Logs
# print('                  ************************')
# print('                  * MPC controller ready *') 
# print('                  ************************')        
# print('---------------------------------------------------------')
# print('- Total simulation duration            : T_tot  = '+str(T_tot)+' s')
# print('- Control frequency                    : f_ctrl = '+str(ctrl_freq)+' Hz')
# print('- Replanning frequency                 : f_plan = '+str(plan_freq)+' Hz')
# print('- Total # of control steps             : N_tot  = '+str(N_tot))
# print('- Duration of MPC horizon              : T_ocp  = '+str(T_h)+' s')
# print('- Total # of replanning knots          : N_p    = '+str(N_p))
# print('- OCP integration step                 : dt     = '+str(dt)+' s')
# print('---------------------------------------------------------')
# print("Simulation will start...")
# time.sleep(1)

# # Measure initial state from simulation environment &init data
# q_mea, v_mea = robot.get_state()
# robot.forward_robot(q_mea, v_mea)
# x0 = np.concatenate([q_mea, v_mea]).T
# print("Initial state ", str(x0))
# X_mea[0, :] = x0
# X_des[0, :] = x0
# # F_mea[0, :] = ddp.problem.runningDatas[0].differential.costs.costs["force"].contact.f.vector
# # F_des[0, :] = F_mea[0, :]
# # Replan counter
# nb_replan = 0
# # SIMULATION LOOP
# switch=False
# for i in range(N_tot): 
#     print("  ")
#     print("Sim step "+str(i)+"/"+str(N_tot))
#     # Solve OCP if we are in a planning cycle
#     if(i%int(ctrl_freq/plan_freq) == 0):
#         print("  Replan step "+str(nb_replan)+"/"+str(N_p))
#         # Reset x0 to measured state + warm-start solution
#         ddp.problem.x0 = X_mea[i, :]
#         xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
#         xs_init[0] = X_mea[i, :]
#         us_init = list(ddp.us[1:]) + [ddp.us[-1]] 

#         ### HERE UPDATE OCP AS NEEDED ####
#         # STATE-based switch
#         if(len(p.getContactPoints(1, 2))>0 and switch==False):
#             switch=True
#             ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.Mref.placement =  robot.pin_robot.data.oMf[id_endeff]
#             ddp.problem.terminalModel.differential.contacts.changeContactStatus("contact", True)
#             ddp.problem.terminalModel.differential.costs.changeCostStatus("force", True)
#             for k,m in enumerate(ddp.problem.runningModels[:]):
#                 # Activate contact and force cost
#                 m.differential.contacts.contacts["contact"].contact.Mref.placement =  robot.pin_robot.data.oMf[id_endeff]
#                 m.differential.contacts.changeContactStatus("contact", True)
#                 m.differential.costs.changeCostStatus("force", True)
#                 m.differential.costs.costs["force"].weight = 10.
#                 # m.differential.costs.costs["placement"].weight = 1e-1
#                 # Update state reg cost
#                 m.differential.costs.costs["stateReg"].reference = xs_init[0]
#                 m.differential.costs.costs["stateReg"].weight = 0
#                 # Update control reg cost
#                 m.differential.costs.costs["ctrlReg"].reference = us_init[0]
#                 m.differential.costs.costs["ctrlReg"].weight = 1.

#         # Solve OCP & record MPC predictions
#         ddp.solve(xs_init, us_init, maxiter=maxit, isFeasible=False)
#         X_pred[nb_replan, :, :] = np.array(ddp.xs)
#         U_pred[nb_replan, :, :] = np.array(ddp.us)
#         for j in range(N_h):
#             F_pred[nb_replan, j, :] = ddp.problem.runningDatas[j].differential.multibody.contacts.contacts['contact'].f.vector
#         # F_pred[nb_replan, -1, :] = ddp.problem.terminalData.differential.multibody.contacts.contacts['contact'].f.vector
#         # Extract 1st control and 2nd state
#         u_des = U_pred[nb_replan, 0, :] 
#         x_des = X_pred[nb_replan, 1, :]
#         f_des = F_pred[nb_replan, 0, :]
#         # Increment replan counter
#         nb_replan += 1

#     # Record and apply the 1st control
#     U_des[i, :] = u_des
#     U_mea[i, :] = u_des
#     # Send control to simulation & step simulator
#     # robot.send_joint_command(u_des + ddp.K[0].dot(X_mea[i, :] - x_des)) # with Ricatti gain
#     robot.send_joint_command(u_des)
#     p.stepSimulation()
#     # Measure new state from simulation and record data
#     q_mea, v_mea = robot.get_state()
#     robot.forward_robot(q_mea, v_mea)
#     x_mea = np.concatenate([q_mea, v_mea]).T 
#     X_mea[i+1, :] = x_mea                    # Measured state
#     X_des[i+1, :] = x_des                    # Desired state
#     F_des[i, :] = f_des                      # Desired force
#     F_pin[i, :] = ddp.problem.runningDatas[0].differential.costs.costs["force"].contact.f.vector

# # GENERATE NICE PLOT OF SIMULATION
# with_predictions = False
# from matplotlib.collections import LineCollection
# import matplotlib.pyplot as plt
# import matplotlib
# # Time step duration of the control loop
# dt_ctrl = float(1./ctrl_freq)
# # Time step duration of planning loop
# dt_plan = float(1./plan_freq)
# # Reshape trajs if necessary 
# q_pred = X_pred[:,:,:nq]
# v_pred = X_pred[:,:,nv:]
# q_mea = X_mea[:,:nq]
# v_mea = X_mea[:,nv:]
# q_des = X_des[:,:nq]
# v_des = X_des[:,nv:]
# p_mea = utils.get_p(q_mea, robot.pin_robot, id_endeff)
# p_des = utils.get_p(q_des, robot.pin_robot, id_endeff) 
# # Create time spans for X and U + Create figs and subplots
# tspan_x = np.linspace(0, T_tot, N_tot+1)
# tspan_u = np.linspace(0, T_tot-dt_ctrl, N_tot)
# fig_x, ax_x = plt.subplots(nq, 2)
# fig_u, ax_u = plt.subplots(nq, 1)
# fig_p, ax_p = plt.subplots(3,1)
# # For each joint
# for i in range(nq):
#     # Extract state predictions of i^th joint
#     q_pred_i = q_pred[:,:,i]
#     v_pred_i = v_pred[:,:,i]
#     u_pred_i = U_pred[:,:,i]
#     # print(u_pred_i[0,0])
#     if(with_predictions):
#         # For each planning step in the trajectory
#         for j in range(N_p):
#             # Receding horizon = [j,j+N_h]
#             t0_horizon = j*dt_plan
#             tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
#             tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
#             # Set up lists of (x,y) points for predicted positions and velocities
#             points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
#             points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
#             points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
#             # Set up lists of segments
#             segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
#             segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
#             segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
#             # Make collections segments
#             cm = plt.get_cmap('Greys_r') 
#             lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
#             lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
#             lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
#             lc_q.set_array(tspan_x_pred)
#             lc_v.set_array(tspan_x_pred) 
#             lc_u.set_array(tspan_u_pred)
#             # Customize
#             lc_q.set_linestyle('-')
#             lc_v.set_linestyle('-')
#             lc_u.set_linestyle('-')
#             lc_q.set_linewidth(1)
#             lc_v.set_linewidth(1)
#             lc_u.set_linewidth(1)
#             # Plot collections
#             ax_x[i,0].add_collection(lc_q)
#             ax_x[i,1].add_collection(lc_v)
#             ax_u[i].add_collection(lc_u)
#             # Scatter to highlight points
#             colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
#             my_colors = cm(colors)
#             ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
#             ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
#             ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
    

#     # Desired joint position (interpolated from prediction)
#     ax_x[i,0].plot(tspan_x, q_des[:,i], 'b-', label='Desired')
#     # Measured joint position (PyBullet)
#     ax_x[i,0].plot(tspan_x, q_mea[:,i], 'r-', label='Measured')
#     ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{i}$ (rad)')
#     ax_x[i,0].grid()

#     # Desired joint velocity (interpolated from prediction)
#     ax_x[i,1].plot(tspan_x, v_des[:,i], 'b-', label='Desired')
#     # Measured joint velocity (PyBullet)
#     ax_x[i,1].plot(tspan_x, v_mea[:,i], 'r-', label='Measured')
#     ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{i}$ (rad/s)')
#     ax_x[i,1].grid()

#     # Desired joint torque (interpolated feedforward)
#     ax_u[i].plot(tspan_u, U_des[:,i], 'b-', label='Desired (ff)')
#     # Total
#     ax_u[i].plot(tspan_u, U_mea[:,i], 'r-', label='Measured (ff+fb)') 
#     # ax_u[i].plot(tspan_u[0], u_mea[0,i], 'co', label='Initial')
#     # print(" U0 mea plotted = "+str(u_mea[0,i]))
#     # ax_u[i].plot(tspan_u, u_mea[:,i]-u_des[:,i], 'g-', label='Riccati (fb)')
#     # Total torque applied
#     ax_u[i].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
#     ax_u[i].grid()

#     # Legend
#     handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
#     fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

#     handles_u, labels_u = ax_u[i].get_legend_handles_labels()
#     fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

# # Get desired and measured contact forces
# f_ref = desiredFrameForce.vector
# fig_f, ax_f = plt.subplots(6,1)
# # Plot contact force
# for i in range(6):
#     ax_f[i].plot(tspan_u, F_des[:,i], 'bo', label='f_des_NEW', alpha=0.3)
#     ax_f[i].plot(tspan_u, F_mea[:,i], 'ro', label='f_mea_NEW', alpha=0.3)
#     ax_f[i].plot(tspan_u, [f_ref[i]]*N_tot, 'k.', label='ref_contact', alpha=0.5)
#     ax_f[i].set(xlabel='t (s)', ylabel='$f_{i}$ (N)')
#     ax_f[i].grid()
#     # Legend
#     handles_f, labels_f = ax_f[i].get_legend_handles_labels()
#     fig_f.legend(handles_f, labels_f, loc='upper right', prop={'size': 16})

# # Compute predicted force using predicted trajs
# if(with_predictions):
#     # For dim
#     for i in range(6):
#         # Extract state predictions of i^th dim
#         f_pred_i = F_pred[:, :, i]
#         # For each planning step in the trajectory
#         for j in range(N_p):
#             # Receding horizon = [j,j+N_h]
#             # Receding horizon = [j,j+N_h]
#             t0_horizon = j*dt_plan
#             tspan_f_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h) #np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
#             # Set up lists of (x,y) points for predicted positions and velocities
#             points_f = np.array([tspan_f_pred, f_pred_i[j,:]]).transpose().reshape(-1,1,2)
#             # Set up lists of segments
#             segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
#             # Make collections segments
#             cm = plt.get_cmap('Greys_r') 
#             lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
#             lc_f.set_array(tspan_f_pred)
#             # Customize
#             lc_f.set_linestyle('-')
#             lc_f.set_linewidth(1)
#             # Plot collections
#             ax_f[i].add_collection(lc_f)
#             # Scatter to highlight points
#             colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
#             my_colors = cm(colors)
#             # ax_f[i].scatter(tspan_f_pred, f_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black' 
#             ax_f[i].scatter(tspan_f_pred, f_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
    

# # Plot endeff
# # x
# ax_p[0].plot(tspan_x, p_des[:,0], 'b-', label='x_des')
# ax_p[0].plot(tspan_x, p_mea[:,0], 'r-.', label='x_mea')
# ax_p[0].set_title('x-position')
# ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
# ax_p[0].grid()
# # y
# ax_p[1].plot(tspan_x, p_des[:,1], 'b-', label='y_des')
# ax_p[1].plot(tspan_x, p_mea[:,1], 'r-.', label='y_mea')
# ax_p[1].set_title('y-position')
# ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
# ax_p[1].grid()
# # z
# ax_p[2].plot(tspan_x, p_des[:,2], 'b-', label='z_des')
# ax_p[2].plot(tspan_x, p_mea[:,2], 'r-.', label='z_mea')
# ax_p[2].set_title('z-position')
# ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
# ax_p[2].grid()
# # Add frame ref if any
# p_ref = desiredFramePlacement.translation
# ax_p[0].plot(tspan_x, [p_ref[0]]*(N_tot+1), 'ko', label='ref_contact', alpha=0.5)
# ax_p[1].plot(tspan_x, [p_ref[1]]*(N_tot+1), 'ko', label='ref_contact', alpha=0.5)
# ax_p[2].plot(tspan_x, [p_ref[2]]*(N_tot+1), 'ko', label='ref_contact', alpha=0.5)
# handles_p, labels_p = ax_p[0].get_legend_handles_labels()
# fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

# # Titles
# fig_x.suptitle('Joint trajectories: des. vs sim. (DDP-based MPC)', size=16)
# fig_u.suptitle('Joint torques: des. vs sim. (DDP-based MPC)', size=16)
# fig_p.suptitle('End-effector: ref. vs des. vs sim. (DDP-based MPC)', size=16)
# fig_f.suptitle('End-effector force', size=16)

# plt.show() 

