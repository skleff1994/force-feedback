# Title : augmented_state_sim.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Test of the "augmented state" approach for force feedback on the point mass system 

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.dyn_models import PointMassContact
from models.cost_models import *
from models.croco_IAMs import ActionModel

from utils import animatePointMass, plotPointMass

#########
# MODEL #
#########
# Create dynamics model
dt = 1e-3
K = 10.
B = 2
model = PointMassContact(K=K, B=B, dt=dt, p0=0., integrator='euler')
p0 = model.p0 
# Running and terminal cost models
running_cost = CostSum(model)
terminal_cost = CostSum(model)
  # Setup cost terms
p_ref = 0.
v_ref = 0.
lmb_ref = -K*(p_ref - p0) - B*v_ref
x_ref = np.array([p_ref, v_ref, lmb_ref])
print("REF. ORIGIN = "+str(x_ref))
# running_cost.add_cost(QuadTrackingCost(model, np.zeros(3), 1e-4*np.eye(model.nx)))
# running_cost.add_cost(QuadTrackingCost(model, x_ref, .1*np.eye(model.nx)))    
running_cost.add_cost(QuadCtrlRegCost(model, 1e-2*np.eye(model.nu)))
terminal_cost.add_cost(QuadTrackingCost(model, x_ref, 10.*np.eye(model.nx)))
  # IAMs for Crocoddyl
running_IAM = ActionModel(model, running_cost) 
terminal_IAM = ActionModel(model, terminal_cost) 
# Define shooting problem
# Initial conditions
p = 1.                # initial position
v = 0.                # initial velocity 
lmb = -K*(p-p0) - B*v # initial contact force
x0 = np.matrix([p, v, lmb]).T
u0 = np.matrix([0.])
print("Initial state = "+str(x0.T))
N_h = 100
problem = crocoddyl.ShootingProblem(x0, [running_IAM]*N_h, terminal_IAM)
print(" Initial guess = x0 and u0 = quasiStatic(x0) ")
xs0 = [x0]*(N_h+1)
us0 = problem.quasiStatic(xs0[:-1])
# # print(xs0)
# # print(np.array(us0))
# problem.calcDiff(xs0, us0)
# # for k,d in enumerate(problem.runningDatas):
# #   print("Node "+str(k)+" : "+" cost = "+str(d.cost))
# print("Terminal node : cost = "+str(problem.terminalData.cost))
# print("Terminal node : Lx = "+str(problem.terminalData.Lx))
# print("Terminal node : r = "+str(problem.terminalData.r))
# # Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
# ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])
# Solve and retrieve X,U
done = ddp.solve([], [], 100)
X_real = np.array(ddp.xs)
U_real = np.array(ddp.us)
# # Compute integration residual on force
# res = model.get_residual(X_real, p0)
# import matplotlib.pyplot as plt
# plt.plot(res)
# from utils import plotPointMass
# plotPointMass(X_real, U_real, ref=x_ref)
# # Animate 
# from IPython.display import HTML
# anim = animatePointMass(ddp.xs, sleep=10)
# HTML(anim.to_html5_video())
# xs, us = simulate(running_IAM, ddp)
# plotPointMass(xs, us)
# # Display norm of partial derivatives 
# K1 = np.vstack(( np.array([[ddp.K[0][0] for i in range(T)]]).transpose())) # position gains
# K2 = np.vstack(( np.array([[ddp.K[0][1] for i in range(T)]]).transpose())) # velocity gains
# K3 = np.vstack(( np.array([[ddp.K[0][2] for i in range(T)]]).transpose())) # contact force gains
# # Norms
# print(np.linalg.norm(K1))
# print(np.linalg.norm(K2))
# print(np.linalg.norm(K3))

#######
# MPC #
#######
# Parameters
maxit=2
T_tot = 2.
plan_freq = 500                      # MPC re-planning frequency (Hz)
ctrl_freq = 1000                         # Control - simulation - frequency (Hz)
N_tot = int(T_tot*ctrl_freq)          # Total number of control steps in the simulation (s)
N_p = int(T_tot*plan_freq)            # Total number of OCPs (replan) solved during the simulation
T_h = N_h*dt                          # Duration of the MPC horizon (s)
# Init data
nx, nq, nv, nu = 3, 1, 1, 1
X_mea = np.zeros((N_tot+1, nx))       # Measured states 
X_des = np.zeros((N_tot+1, nx))       # Desired states
U_des = np.zeros((N_tot, nu))         # Desired controls 
X_pred = np.zeros((N_p, N_h+1, nx))   # MPC predictions (state)
U_pred = np.zeros((N_p, N_h, nu))     # MPC predictions (control)
# Replan counter
nb_replan = 0
# Measure initial state from simulation environment
X_mea[0, :] = list(x0)
X_des[0, :] = list(x0)
# SIMULATION LOOP
# Simulation loop (at control rate)
for i in range(N_tot): 
  print("  ")
  print("Sim step "+str(i)+"/"+str(N_tot))
  # Solve OCP if we are in a planning cycle
  if(i%int(ctrl_freq/plan_freq) == 0):
    print("  Replan step "+str(nb_replan)+"/"+str(N_p))
    # Reset problem to measured state 
    # print("    from state "+str(X_mea[0, :]))
    ddp.problem.x0 = X_mea[i, :]
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
    xs_init[0] = X_mea[i, :]
    us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
    ddp.solve(xs_init, us_init, maxit, False)
    # Record trajectories
    X_pred[nb_replan, :, :] = np.array(ddp.xs)
    U_pred[nb_replan, :, :] = np.array(ddp.us)
    # Extract 1st control and 2nd state
    u_des = U_pred[nb_replan, 0, :] 
    x_des = X_pred[nb_replan, 1, :]
    # Increment replan counter
    nb_replan += 1
  # Record and apply the 1st control
  U_des[i, :] = u_des
  # Measure new state from simulation 
  X_mea[i+1,:] = model.calc(X_mea[i,:] , u_des) 
  # Record desired state
  X_des[i+1, :] = x_des


# GENERATE NICE PLOT OF SIMULATION
with_predictions = True
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
# Time step duration of the control loop
dt_ctrl = float(1./ctrl_freq)
# Time step duration of planning loop
dt_plan = float(1./plan_freq)
# Joints & torques
    # State predictions (MPC)
q_pred = X_pred[:,:,:nq]
v_pred = X_pred[:,:,nv:]
u_pred = U_pred[:,:,:]
    # State measurements (PyBullet)
q_mea = X_mea[:,:nq]
v_mea = X_mea[:,nv:]
    # 'Desired' state = interpolated predictions
q_des = X_des[:,:nq]
v_des = X_des[:,nv:]
    # 'Desired' control = interpolation of DDP ff torques 
u_des = U_des
# Create time spans for X and U
tspan_x = np.linspace(0, T_tot, N_tot+1)
tspan_u = np.linspace(0, T_tot-dt_ctrl, N_tot)
# Create figs and subplots
fig_x, ax_x = plt.subplots(nq, 2)
fig_u, ax_u = plt.subplots(nq, 1)
# Extract state predictions of 0^th joint
q_pred_i = q_pred[:,:,0]
v_pred_i = v_pred[:,:,0]
u_pred_i = u_pred[:,:,0]
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
    ax_x[0].add_collection(lc_q)
    ax_x[1].add_collection(lc_v)
    ax_u.add_collection(lc_u)
    # Scatter to highlight points
    colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
    my_colors = cm(colors)
    ax_x[0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
    ax_x[1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
    ax_u.scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
# Desired joint position (interpolated from prediction)
ax_x[0].plot(tspan_x, q_des[:,0], 'b-', label='Desired')
# Measured joint position (PyBullet)
ax_x[0].plot(tspan_x, q_mea[:,0], 'r-', label='Measured')
ax_x[0].set(xlabel='t (s)', ylabel='$q_{0}$ (rad)')
ax_x[0].grid()
# Desired joint velocity (interpolated from prediction)
ax_x[1].plot(tspan_x, v_des[:,0], 'b-', label='Desired')
# Measured joint velocity (PyBullet)
ax_x[1].plot(tspan_x, v_mea[:,0], 'r-', label='Measured')
ax_x[1].set(xlabel='t (s)', ylabel='$v_{0}$ (rad/s)')
ax_x[1].grid()
# Desired joint torque (interpolated feedforward)
ax_u.plot(tspan_u, u_des, 'b-', label='Desired (ff)')
ax_u.set(xlabel='t (s)', ylabel='$u_{0}$ (Nm)')
ax_u.grid()
# Legend
handles_x, labels_x = ax_x[0].get_legend_handles_labels()
fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
handles_u, labels_u = ax_u.get_legend_handles_labels()
fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
fig_x.suptitle('Joint trajectories: des. vs sim. (DDP-based MPC)', size=16)
fig_u.suptitle('Joint torques: des. vs sim. (DDP-based MPC)', size=16)
plt.show() 


