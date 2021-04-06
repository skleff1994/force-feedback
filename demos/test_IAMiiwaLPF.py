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

from models.croco_IAMs import IntegratedActionModelLPF
import utils

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
nu = robot.pin_robot.model.nq
    # Reset robot to initial state in PyBullet and update pinocchio data accordingly 
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
dq0 = pin.utils.zero(nv)
robot.reset_state(q0, dq0)
robot.forward_robot(q0, dq0)
    # Get gravity torque for convenience
u_grav = pin.rnea(robot.pin_robot.model, robot.pin_robot.data, q0, np.zeros((nv,1)), np.zeros((nq,1)))
    # Get initial frame placement
M_ee = robot.pin_robot.data.oMf[id_endeff]
print("[PyBullet] Created robot (id = "+str(robot.robotId)+")")
print("Initial placement in WORLD frame : ")
print(M_ee)
  # CONTACT
    # Set contact placement = M_ee with offset (cf. below)
M_ct = pin.SE3.Identity()
M_ct.rotation = M_ee.rotation 
offset = 0.03 + 0.003499998807875214 # 0.1 + 0.003499998807875214 
M_ct.translation = M_ee.act(np.array([0., 0., offset])) 
print("Contact placement in WORLD frame : ")
print(M_ct)

# Measure distance EE to contact surface using p.getContactPoints() 
# in order to avoid PyB repulsion due to penetration 
# Result = 0.03 + 0.003499998807875214. Problem : smaller than ball radius (changed urdf?) . 
contactId = utils.display_contact_surface(M_ct, robot.robotId, with_collision=True)
print("[PyBullet] Created contact plane (id = "+str(contactId)+")")
print("[PyBullet]   >> Detect contact points : ")
p.stepSimulation()
contact_points = p.getContactPoints(1, 2)
for k,i in enumerate(contact_points):
  print("      Contact point n°"+str(k)+" : distance = "+str(i[8])+" (m) | force = "+str(i[9])+" (N)")
# time.sleep(100)

# # Get measured torques w.r.t. sent torques
# for jointId in range(p.getNumJoints(robot.robotId)):
#   print("Joint n°"+str(jointId)+" : ")
#   p.enableJointForceTorqueSensor(robot.robotId, jointId)
#   print(p.getJointState(robot.robotId, jointId))

# robot.send_joint_command(np.ones(nu))
# p.stepSimulation()


# # Get measured torques w.r.t. sent torques
# for jointId in range(p.getNumJoints(robot.robotId)):
#   print("Joint n°"+str(jointId)+" : ")
#   # p.enableJointForceTorqueSensor(robot.robotId, jointId, True)
#   print(p.getJointState(robot.robotId, jointId))

# robot.send_joint_command(np.ones(nu))
# p.stepSimulation()

# # Get measured torques w.r.t. sent torques
# for jointId in range(p.getNumJoints(robot.robotId)):
#   print("Joint n°"+str(jointId)+" : ")
#   # p.enableJointForceTorqueSensor(robot.robotId, jointId, True)
#   print(p.getJointState(robot.robotId, jointId))

#################
### OCP SETUP ###
#################
  # OCP parameters 
dt = 2e-2                              # OCP integration step (s)               
N_h = 50                                # Number of knots in the horizon 
x0 = np.concatenate([q0, dq0, u_grav])  # Initial state
print("Initial state : ", x0.T)
  # Construct cost function terms
   # State and actuation models
state = crocoddyl.StateMultibody(robot.pin_robot.model)
actuation = crocoddyl.ActuationModelFull(state)
   # State regularization
stateRegWeights = np.array([1.]*nq + [1.]*nv)  
x_reg_ref = x0[:nq+nv]
xRegCost = crocoddyl.CostModelState(state, 
                                    crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                    x_reg_ref, 
                                    actuation.nu)
print("Created state reg cost.")
   # Control regularization
ctrlRegWeights = np.ones(nu)
ctrlRegWeights[-1] = 100
u_reg_ref = u_grav 
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
u_lim_ref = np.zeros(nu)
uLimitCost = crocoddyl.CostModelControl(state, 
                                        crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                        u_lim_ref)
print("Created ctrl lim cost.")
   # End-effector contact force
desiredFrameForce = pin.Force(np.array([0., 0., -20., 0., 0., 0.]))
# desiredFrameForce = pin.Force( M_ee.act( np.array([0., 0., offset]) ) np.array([0., 0., 50., 0., 0., 0.]))
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
desiredFramePlacement = M_ct #robot.pin_robot.data.oMf[id_endeff] #M_ct
framePlacementWeights = np.ones(6)
framePlacementCost = crocoddyl.CostModelFramePlacement(state, 
                                                       crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                       crocoddyl.FramePlacement(id_endeff, desiredFramePlacement), 
                                                       actuation.nu) 
print("Created frame placement cost.")
# Contact model
ref_placement = crocoddyl.FramePlacement(id_endeff, M_ct) #robot.pin_robot.data.oMf[id_endeff]) #pin.SE3.Identity()) #pin_robot.data.oMf[id_endeff])
contact6d = crocoddyl.ContactModel6D(state, ref_placement, gains=np.array([50.,10.]))
# LPF (CT) param
# k_LPF = 0.001 /dt
# alpha = .01 #1 - k_LPF*dt                        
f_c = 5000 #( 1-alpha)/alpha ) * ( 1/(2*np.pi*dt) ) 
alpha =  1 / (1 + 2*np.pi*dt*f_c) # Smoothing factor : close to 1 means f_c decrease, close to 0 means f_c very large 
print("LOW-PASS FILTER : ")
print("f_c   = ", f_c)
print("alpha = ", alpha)
# Create IAMs
runningModels = []
for i in range(N_h):
  # Create IAM 
  runningModels.append(IntegratedActionModelLPF( 
      crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                          actuation, 
                                                          crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                          crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                          inv_damping=0., 
                                                          enable_force=True), dt=dt, f_c=f_c) )
  # Add cost models
  # runningModels[i].differential.costs.addCost("placement", framePlacementCost, 10) 
  # runningModels[i].differential.costs.addCost("velocity", frameVelocityCost,  10) #, active=False) 
  runningModels[i].differential.costs.addCost("force", frameForceCost, 1e-3, active=True) 
  runningModels[i].differential.costs.addCost("stateReg", xRegCost, 1e-4) 
  runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, 1e-4)
  # runningModels[i].differential.costs.addCost("stateLim", xLimitCost, 1e-3) 
  # runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, 1e-2) 
  # Set up cost on unfiltered control input (same as unfiltered?)
  runningModels[i].set_w_reg_lim_costs(1e-2, u_reg_ref, 1e3, u_lim_ref)
  # Add armature
  runningModels[i].differential.armature = np.array([.1]*7)
  # Add contact models
  runningModels[i].differential.contacts.addContact("contact", contact6d, active=True)

# Terminal IAM + set armature
terminalModel = IntegratedActionModelLPF(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                        actuation, 
                                                        crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                        inv_damping=0., 
                                                        enable_force=True), dt=0, f_c=f_c )
# Add cost models
# terminalModel.differential.costs.addCost("placement", framePlacementCost, 1e3) 
# terminalModel.differential.costs.addCost("force", frameForceCost, 1) #, active=False)
# terminalModel.differential.costs.addCost("stateReg", xRegCost, 1e-3) 
# terminalModel.differential.costs.addCost("stateLim", xLimitCost, 10) 
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
xs = np.array(ddp.xs) # optimal (q,v,u) traj
us = np.array(ddp.us) # optimal   (w)   traj

# #################################
# ### EXTRACT SOLUTION AND PLOT ###
# #################################
# print("Extracting solution...")
# # Extract solution trajectories
# q = np.empty((N_h+1, nq))
# v = np.empty((N_h+1, nv))
# u = np.empty((N_h+1, nq))
# p_ee = np.empty((N_h+1, 3))
# for i in range(N_h+1):
#     q[i,:] = xs[i][:nq].T
#     v[i,:] = xs[i][nv:nv+nq].T
#     u[i,:] = xs[i][nv+nq:].T
#     pin.forwardKinematics(robot.pin_robot.model, robot.pin_robot.data, q[i])
#     pin.updateFramePlacements(robot.pin_robot.model, robot.pin_robot.data)
#     p_ee[i,:] = robot.pin_robot.data.oMf[id_endeff].translation.T
# w = np.empty((N_h, actuation.nu))
# for i in range(N_h):
#     w[i,:] = us[i].T
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
# tspan_w = np.linspace(0, N_h*dt, N_h)
# fig_x, ax_x = plt.subplots(nq, 4)
# fig_w, ax_w = plt.subplots(nq, 1)
# fig_p, ax_p = plt.subplots(3, 1)
# fig_f, ax_f = plt.subplots(6, 1)
# # Plot joints pos, vel , acc, torques
# for i in range(nq):
#     # Positions
#     ax_x[i,0].plot(tspan_x, q[:,i], 'b.', label='pos_des')
#     ax_x[i,0].plot(tspan_x[-1], q[-1,i], 'ro')
#     ax_x[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
#     ax_x[i,0].grid()
#     # Velocities
#     ax_x[i,1].plot(tspan_x, v[:,i], 'b.', label='vel_des')
#     ax_x[i,1].plot(tspan_x[-1], v[-1,i], 'ro')
#     ax_x[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
#     ax_x[i,1].grid()
#     # Accelerations
#     ax_x[i,2].plot(tspan_x, a[:,i], 'b.', label='acc_des')
#     ax_x[i,2].plot(tspan_x[-1], a[-1,i], 'ro')
#     ax_x[i,2].set_ylabel('$a_%s$'%i, fontsize=16)
#     ax_x[i,2].grid()
#     # Torques
#     ax_x[i,3].plot(tspan_x, u[:,i], 'b.', label='torque_des')
#     ax_x[i,3].plot(tspan_x[-1], u[-1,i], 'ro')
#     ax_x[i,3].set_ylabel('$u_%s$'%i, fontsize=16)
#     ax_x[i,3].grid()
#     # Input (w)
#     ax_w[i].plot(tspan_w, w[:,i], 'b.', label='input_des') 
#     ax_w[i].set_ylabel(ylabel='$w_%d$'%i, fontsize=16)
#     ax_w[i].grid()
#     # Remove xticks labels for clarity 
#     if(i != nq-1):
#         for j in range(3):
#             ax_x[i,j].set_xticklabels([])
#         ax_w[i].set_xticklabels([])
#     # Set xlabel on bottom plot
#     if(i == nq-1):
#         for j in range(4):
#             ax_x[i,j].set_xlabel('t (s)', fontsize=16)
#         ax_w[i].set_xlabel('t (s)', fontsize=16)
#     # Legend
#     handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
#     fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
#     handles_w, labels_w = ax_w[i].get_legend_handles_labels()
#     fig_w.legend(handles_w, labels_w, loc='upper right', prop={'size': 16})
# # Plot contact force
# f_ref = desiredFrameForce.vector
# ylabels_f = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
# for i in range(6):
#     ax_f[i].plot(tspan_w, [f_ref[i]]*N_h, 'ro', label='REF', alpha=0.5)
#     ax_f[i].plot(tspan_w, f[:,i], 'b.', label='desired')
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
# fig_w.align_ylabels()
# fig_f.align_ylabels()
# fig_p.align_ylabels()
# fig_x.suptitle('Joint trajectories', size=16)
# fig_w.suptitle('Joint input', size=16)
# fig_f.suptitle('End-effector force', size=16)
# fig_p.suptitle('End-effector trajectory', size=16)
# plt.show()


##################
# MPC SIMULATION #
##################
# MPC & simulation parameters
maxit = 1
T_tot = 1.
plan_freq = 1000                      # MPC re-planning frequency (Hz)
ctrl_freq = 1000                      # Control - simulation - frequency (Hz)
N_tot = int(T_tot*ctrl_freq)          # Total number of control steps in the simulation (s)
N_p = int(T_tot*plan_freq)            # Total number of OCPs (replan) solved during the simulation
T_h = N_h*dt                          # Duration of the MPC horizon (s)
# Initialize data : in simulation, x=(q,v) u=tau !!!
nx = nq+nv+nu
X_mea = np.zeros((N_tot+1, nx))       # Measured states x=(q,v,tau) 
X_des = np.zeros((N_tot+1, nx))       # Desired states x=(q,v,tau)
X_pred = np.zeros((N_p, N_h+1, nx))   # MPC predictions (state) (t,q,v,tau)
U_pred = np.zeros((N_p, N_h, nu))     # MPC predictions (control) (t,w)
U_des = np.zeros((N_tot, nq))         # Unfiltered torques planned by MPC u=w
contact_des = [False]*X_des.shape[0]                # Contact record for contact force
contact_mea = [False]*X_mea.shape[0]                # Contact record for contact force
contact_pred = np.zeros((N_p, N_h+1), dtype=bool)   # Contact record for contact force
F_des = np.zeros((N_tot, 6))        # Desired contact force
F_pin = np.zeros((N_tot, 6))        # Contact force computed with pinocchio (should be the same as desired)
F_pred = np.zeros((N_p, N_h, 6))    # MPC prediction of contact force (same as pin on desired trajs)
F_mea = np.zeros((N_tot, 6))        # PyBullet measurement of contact force (? at which contact point ?)
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
q_mea, v_mea = robot.get_state()
robot.forward_robot(q_mea, v_mea)
u_mea = pin.rnea(robot.pin_robot.model, robot.pin_robot.data, q_mea, v_mea, np.zeros((nq,1)))
x0 = np.concatenate([q_mea, v_mea, u_mea]).T
print("Initial state ", str(x0))
X_mea[0, :] = x0
X_des[0, :] = x0
# F_mea[0, :] = ddp.problem.runningDatas[0].differential.costs.costs["force"].contact.f.vector
# F_des[0, :] = F_mea[0, :]
# Replan counter
nb_replan = 0
# SIMULATION LOOP
switch=False
for i in range(N_tot): 
    print("  ")
    print("Sim step "+str(i)+"/"+str(N_tot))
    
    print("  COST = ", ddp.cost)
    # Solve OCP if we are in a planning cycle
    if(i%int(ctrl_freq/plan_freq) == 0):
        print("  Replan step "+str(nb_replan)+"/"+str(N_p))

        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = X_mea[i, :].T 
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = X_mea[i, :].T
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        
        # ### HERE UPDATE OCP AS NEEDED ####
        # # STATE-based switch
        # if(len(p.getContactPoints(1, 2))>0 and switch==False):
        #     print("      !!! CONTACT !!!")
        #     switch=True
        #     ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.Mref.placement = robot.pin_robot.data.oMf[id_endeff]
        #     ddp.problem.terminalModel.differential.contacts.changeContactStatus("contact", True)
        #     # ddp.problem.terminalModel.differential.costs.changeCostStatus("force", True)
        #     # ddp.problem.terminalModel.differential.costs.costs["placement"].reference = robot.pin_robot.data.oMf[id_endeff]
        #     for k,m in enumerate(ddp.problem.runningModels[:]):
        #         # Activate contact and force cost
        #         m.differential.contacts.contacts["contact"].contact.Mref.placement = robot.pin_robot.data.oMf[id_endeff]
        #         m.differential.contacts.changeContactStatus("contact", True)
        #         m.differential.costs.changeCostStatus("force", True)
        #         m.differential.costs.costs["placement"].reference = robot.pin_robot.data.oMf[id_endeff]
        #         # m.differential.costs.changeCostStatus("velocity", True)
        #         m.differential.costs.costs["force"].weight = 2e-2
        #         m.differential.costs.costs["placement"].weight = 1e-3
        #         # # # Update state reg cost
        #         m.differential.costs.costs["stateReg"].reference = xs_init[0][:nq+nv]
        #         # m.differential.costs.costs["stateReg"].weight = 0.
        #         m.differential.costs.costs["stateLim"].weight = 1e-3
        #         # # # Update control reg cost
        #         ureg = pin.rnea(robot.pin_robot.model, robot.pin_robot.data, xs_init[0][:nq], np.zeros((nv,1)), np.zeros((nq,1)))
        #         m.differential.costs.costs["ctrlReg"].reference = ureg
        #         m.differential.costs.costs["ctrlReg"].weight = 1.
        #         m.set_w_reg_lim_costs(1., ureg, 1e2, np.zeros(nu))
        #         # m.w_reg_ref = ureg
        #         # m.differential.costs.costs["ctrlLim"].weight = 0.
        #         # m.differential.costs.costs["stateLim"].weight = 1e3*

        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=maxit, isFeasible=False)
        # print("DDP.XS[1] = ", ddp.xs[1])
        X_pred[nb_replan, :, :] = np.array(ddp.xs)# [:,:-nu] # (t,q,v)
        U_pred[nb_replan, :, :] = np.array(ddp.us)# [1:,-nu:] # (t,u)
        for j in range(N_h):
            F_pred[nb_replan, j, :] = ddp.problem.runningDatas[j].differential.multibody.contacts.contacts['contact'].f.vector
        # F_pred[nb_replan, -1, :] = ddp.problem.terminalData.differential.multibody.contacts.contacts['contact'].f.vector
        # Extract 1st control and 2nd state
        u_des = U_pred[nb_replan, 0, :] 
        x_des = X_pred[nb_replan, 1, :]
        f_des = F_pred[nb_replan, 0, :]
        # Increment replan counter
        nb_replan += 1
    # print("X_DES = ", x_des)
    # Record the 1st control : desired torque = unfiltered torque output by DDP
    U_des[i, :] = u_des
    # u_full = u_des + ddp.K[0].dot(X_mea[i, :] - x_des)
    # Select filtered torque = integration of LPF(u_des) = x_des ? Or integration over a control step only ?
    tau_des = x_des[nq+nv:] #alpha*X_mea[i, :][-nu:] + (1-alpha)*u_des  #x_des[nq+nv:] # same as : alpha*x0[-nu:] + (1-alpha)*u_des 
    # Send control to simulation & step u
    # robot.send_joint_command(u_des + ddp.K[0].dot(X_mea[i, :] - x_des)) # with Ricatti gain
    robot.send_joint_command(tau_des)
    p.stepSimulation()
    # Measure new state from simulation and record data
    q_mea, v_mea = robot.get_state()
    robot.forward_robot(q_mea, v_mea)
      # Simulate torque measurement : here add LPF or elastic elements
      # temporarily : measured = same as commanded torque 
    # new_alpha =  np.sin(i*1e-3) / (1 + 2*np.pi*1e-3*20)
    tau_mea = tau_des #new_alpha*X_mea[i, -nu:] + (1-new_alpha)*u_des   # tau_des
    x_mea = np.concatenate([q_mea, v_mea, tau_mea]).T 
    X_mea[i+1, :] = x_mea                    # Measured state
    X_des[i+1, :] = x_des                    # Desired state
    F_des[i, :] = f_des                      # Desired force
    F_pin[i, :] = ddp.problem.runningDatas[0].differential.costs.costs["force"].contact.f.vector
    # F_mea[i, :] = ddp.problem.runningDatas[0].differential.costs.costs["force"].contact.f.vector

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
v_pred = X_pred[:,:,nq:nq+nv]
tau_pred = X_pred[:,:,nq+nv:]
q_mea = X_mea[:,:nq]
v_mea = X_mea[:,nq:nq+nv]
tau_mea  = X_mea[:,nq+nv:]
q_des = X_des[:,:nq]
v_des = X_des[:,nq:nq+nv]
tau_des = X_des[:,nq+nv:]
p_mea = utils.get_p(q_mea, robot.pin_robot, id_endeff)
p_des = utils.get_p(q_des, robot.pin_robot, id_endeff) 
# Create time spans for X and U + Create figs and subplots
tspan_x = np.linspace(0, T_tot, N_tot+1)
tspan_u = np.linspace(0, T_tot-dt_ctrl, N_tot)
fig_x, ax_x = plt.subplots(nq, 3)
fig_u, ax_u = plt.subplots(nq, 1)
fig_p, ax_p = plt.subplots(3,1)
# For each joint
for i in range(nq):
    # Extract state predictions of i^th joint
    q_pred_i = q_pred[:,:,i]
    v_pred_i = v_pred[:,:,i]
    tau_pred_i = tau_pred[:,:,i]
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
            points_tau = np.array([tspan_x_pred, tau_pred_i[j,:]]).transpose().reshape(-1,1,2)
            points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
            # Set up lists of segments
            segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
            segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
            segs_tau = np.concatenate([points_tau[:-1], points_tau[1:]], axis=1)
            segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
            # Make collections segments
            cm = plt.get_cmap('Greys_r') 
            lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
            lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
            lc_tau = LineCollection(segs_tau, cmap=cm, zorder=-1)
            lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
            lc_q.set_array(tspan_x_pred)
            lc_v.set_array(tspan_x_pred) 
            lc_tau.set_array(tspan_x_pred) 
            lc_u.set_array(tspan_u_pred)
            # Customize
            lc_q.set_linestyle('-')
            lc_v.set_linestyle('-')
            lc_tau.set_linestyle('-')
            lc_u.set_linestyle('-')
            lc_q.set_linewidth(1)
            lc_v.set_linewidth(1)
            lc_tau.set_linewidth(1)
            lc_u.set_linewidth(1)
            # Plot collections
            ax_x[i,0].add_collection(lc_q)
            ax_x[i,1].add_collection(lc_v)
            ax_x[i,2].add_collection(lc_tau)
            ax_u[i].add_collection(lc_u)
            # Scatter to highlight points
            colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
            my_colors = cm(colors)
            ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
            ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
            ax_x[i,2].scatter(tspan_x_pred, tau_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
            ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
    

    # Joint positions
    ax_x[i,0].plot(tspan_x, q_des[:,i], 'b-', label='Desired')
    # Measured joint position (PyBullet)
    ax_x[i,0].plot(tspan_x, q_mea[:,i], 'r-', label='Measured')
    ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{i}$ (rad)')
    ax_x[i,0].grid()

    # Joint velocities
    ax_x[i,1].plot(tspan_x, v_des[:,i], 'b-', label='Desired')
    # Measured joint velocity (PyBullet)
    ax_x[i,1].plot(tspan_x, v_mea[:,i], 'r-', label='Measured')
    ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{i}$ (rad/s)')
    ax_x[i,1].grid()

    # Joint torques (filtered) = part of the state
    ax_x[i,2].plot(tspan_x, tau_des[:,i], 'b-', label='Desired')
    # Measured joint velocity (PyBullet)
    ax_x[i,2].plot(tspan_x, tau_mea[:,i], 'r-', label='Measured')
    ax_x[i,2].set(xlabel='t (s)', ylabel='$tau_{i}$ (Nm)')
    ax_x[i,2].grid()

    # Joint torques (unfiltered) = control input
    ax_u[i].plot(tspan_u, U_des[:,i], 'b-', label='Desired')
    # Total
    # ax_u[i].plot(tspan_u[0], u_mea[0,i], 'co', label='Initial')
    # print(" U0 mea plotted = "+str(u_mea[0,i]))
    # ax_u[i].plot(tspan_u, u_mea[:,i]-u_des[:,i], 'g-', label='Riccati (fb)')
    # Total torque applied
    ax_u[i].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
    ax_u[i].grid()

    # Legend
    handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
    fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

    handles_u, labels_u = ax_u[i].get_legend_handles_labels()
    fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

# Get desired and measured contact forces
f_ref = desiredFrameForce.vector
fig_f, ax_f = plt.subplots(6,1)
# Plot contact force
for i in range(6):
    ax_f[i].plot(tspan_u, F_des[:,i], 'bo', label='f_des_NEW', alpha=0.3)
    ax_f[i].plot(tspan_u, F_mea[:,i], 'ro', label='f_mea_NEW', alpha=0.3)
    ax_f[i].plot(tspan_u, [f_ref[i]]*N_tot, 'k.', label='ref_contact', alpha=0.5)
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
fig_x.suptitle('Joint positions, velocities and (filtered) torques ', size=16)
fig_u.suptitle('Joint command (unfiltered) torques ', size=16)
fig_p.suptitle('End-effector position ', size=16)
fig_f.suptitle('End-effector force', size=16)

plt.show() 

