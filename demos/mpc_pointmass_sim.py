# Title : test_IAMPM.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# DDP-based MPC on the point mass 

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.croco_IAMs import ActionModelPointMass

# Action model for point mass
dt = 1e-2
N_h = 50
# running_models = []
# for i in range(N_h):
#     md = ActionModelPointMass(dt=dt)
#     running_models.append(md)
integrator='euler'
running_model = ActionModelPointMass(dt=dt, integrator=integrator)
# running_model.w_x = 1e-6
# running_model.w_xreg = 1e-2
running_model.w_ureg = 1e-4

terminal_model = ActionModelPointMass(dt=0.)
terminal_model.w_x = 1.

# Problem + solver
x0 = np.array([1., 0.])
problem = crocoddyl.ShootingProblem(x0, [running_model]*N_h, terminal_model)
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# Solve and retrieve X,U
done = ddp.solve([], [], 100)
X = np.array(ddp.xs)
U = np.array(ddp.us)

# # PLOT
# import matplotlib.pyplot as plt
# p = X[:,0]
# v = X[:,1]
# u = U
# # Create time spans for X and U
# tspan_x = np.linspace(0, N_h*dt, N_h+1)
# tspan_u = np.linspace(0, N_h*dt, N_h)
# # Create figs and subplots
# fig_x, ax_x = plt.subplots(3, 1)
# # fig_u, ax_x[2] = plt.subplots(1, 1)
# # Plot joints
# ax_x[0].plot(tspan_x, p, 'b-', label='pos')
# ax_x[0].set_ylabel('p (m)', fontsize=16)
# ax_x[0].grid()
# ax_x[1].plot(tspan_x, v, 'g-', label='vel')
# ax_x[1].set_ylabel('v (m/s)', fontsize=16)
# ax_x[1].grid()
# ax_x[2].plot(tspan_u, u, 'r-', label='torque')
# ax_x[2].set_ylabel('tau (Nm)', fontsize=16)
# ax_x[2].grid()
# # Legend
# ax_x[-1].set_xlabel('time (s)', fontsize=16)
# handles_x, labels_x = ax_x[0].get_legend_handles_labels()
# fig_x.legend(loc='upper right', prop={'size': 16})
# # Titles
# fig_x.suptitle('State - Control trajectories', size=16)
# plt.show()

#######
# MPC #
#######
# Parameters
maxit= 1
T_tot = 1.
plan_freq = 1000                      # MPC re-planning frequency (Hz)
ctrl_freq = 1000                      # Control - simulation - frequency (Hz)
N_tot = int(T_tot*ctrl_freq)          # Total number of control steps in the simulation (s)
N_p = int(T_tot*plan_freq)            # Total number of OCPs (replan) solved during the simulation
T_h = N_h*dt                          # Duration of the MPC horizon (s)
# Init data
nx, nq, nv, nu = 2, 1, 1, 1
X_mea = np.zeros((N_tot+1, nx))       # Measured states 
X_des = np.zeros((N_tot+1, nx))       # Desired states
U_des = np.zeros((N_tot, nu))         # Desired controls 
X_pred = np.zeros((N_p, N_h+1, nx))   # MPC predictions (state)
U_pred = np.zeros((N_p, N_h, nu))     # MPC predictions (control)
# Replan counter
nb_replan = 0
# Measure initial state from simulation environment
X_mea[0, :] = x0
X_des[0, :] = x0
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
    # Set initial state to measured state
    ddp.problem.x0 = X_mea[i, :]
    # Warm-start solution
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
    xs_init[0] = X_mea[i, :]
    us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
    # Solve OCP
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
  # Measure new state from simulation : integrate current control over 1 control step (1 ms) 
  X_mea[i+1,:] = X_mea[i,:] + running_model.f(X_mea[i,:], u_des)*1e-3 #dt # (X_mea[i,:] , u_des) 
  # Record desired state
  X_des[i+1, :] = x_des


# GENERATE NICE PLOT OF SIMULATION
with_predictions = False
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
fig_x, ax_x = plt.subplots(3, 1)
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
    ax_x[2].add_collection(lc_u)
    # Scatter to highlight points
    colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
    my_colors = cm(colors)
    ax_x[0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
    ax_x[1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
    ax_x[2].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
# Desired joint position (interpolated from prediction)
ax_x[0].plot(tspan_x, q_des[:,0], 'b-', label='Desired')
# Measured joint position (PyBullet)
ax_x[0].plot(tspan_x, q_mea[:,0], 'r-', label='Measured')
ax_x[0].set_ylabel('p (m)', fontsize=16)
ax_x[0].grid()
# Desired joint velocity (interpolated from prediction)
ax_x[1].plot(tspan_x, v_des[:,0], 'b-', label='Desired')
# Measured joint velocity (PyBullet)
ax_x[1].plot(tspan_x, v_mea[:,0], 'r-', label='Measured')
ax_x[1].set_ylabel('v (m/s)', fontsize=16)
ax_x[1].grid()
# Desired joint torque (interpolated feedforward)
ax_x[2].plot(tspan_u, u_des, 'b-', label='Desired (ff)')
ax_x[2].set_ylabel('tau (Nm)', fontsize=16)
ax_x[2].grid()
# Legend
ax_x[-1].set_xlabel('time (s)', fontsize=16)
handles_x, labels_x = ax_x[0].get_legend_handles_labels()
fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
fig_x.suptitle('State and position trajectories', size=16)
plt.show() 