# Title : test_IAMPMContact.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# DDP-based MPC with force feedback using 'augmented state' approach for point mass 

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.croco_IAMs import ActionModelPointMassContact

# Action model for point mass
dt = 1e-2
N_h = 100

# Spring damper
K = 1000.
B = 2*np.sqrt(K)
integrator='euler'
running_model = ActionModelPointMassContact(dt=dt, K=K, B=B, p0=0., integrator=integrator)
# running_model.w_x = 1.
# running_model.w_xreg = 1e-2
running_model.w_ureg = 1e-4
# running_model.w_xlim = 1e-2 # to enforce positive force?

terminal_model = ActionModelPointMassContact(dt=0.)
terminal_model.w_x = 1

# Problem + solver
p0 = 1.
v0 = 1.
lmb0 = -K*(p0 - running_model.p0) - B*v0 #max(0.,-K*(p0 - running_model.p0) - B*v0)
# print(lmb0)
x0 = np.array([p0, v0, lmb0])
problem = crocoddyl.ShootingProblem(x0, [running_model]*N_h, terminal_model)
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# Solve and retrieve X,U
done = ddp.solve([], [], 100)
X = np.array(ddp.xs)
U = np.array(ddp.us)

# # PLOT
# nx = running_model.nx 
# nu = running_model.nu
# import matplotlib.pyplot as plt
# p = X[:,0]
# v = X[:,1]
# f = X[:,2]
# u = U
# # Create time spans for X and U
# tspan_x = np.linspace(0, N_h*dt, N_h+1)
# tspan_u = np.linspace(0, N_h*dt, N_h)
# # Create figs and subplots
# fig_x, ax_x = plt.subplots(4, 1)
# # Plot joints
# ax_x[0].plot(tspan_x, p, 'b-', label='pos')
# ax_x[0].set_ylabel('p (m)', fontsize=16)
# ax_x[0].grid()
# ax_x[1].plot(tspan_x, v, 'g-', label='vel')
# ax_x[1].set_ylabel('v (m/s)', fontsize=16)
# ax_x[1].grid()
# ax_x[2].plot(tspan_x, f, 'm-', label='force')
# ax_x[2].set_ylabel('f (N)', fontsize=16)
# ax_x[2].grid()
# ax_x[3].plot(tspan_u, u, 'r-', label='ctrl')
# ax_x[3].set_ylabel('u (Nm)', fontsize=16)
# ax_x[3].grid()
# # If ref specified
# ref = running_model.x_tar
# if(ref is not None):
#     ax_x[0].plot(tspan_x, [ref[0]]*(N_h+1), 'k-.', label='ref')
#     ax_x[1].plot(tspan_x, [ref[1]]*(N_h+1), 'k-.')
#     ax_x[2].plot(tspan_x, [ref[2]]*(N_h+1), 'k-.')
# # Legend
# ax_x[-1].set_xlabel('time (s)', fontsize=16)
# handles_x, labels_x = ax_x[0].get_legend_handles_labels()
# fig_x.legend(loc='upper right', prop={'size': 16})
# # Titles
# fig_x.suptitle('State - Control trajectories', size=16)
# plt.show()

# # Display magnitude of Riccati gains*
# K = ddp.K.tolist()
# Kp = np.vstack(( np.array([[K[i][0] for i in range(N_h)]]).transpose())) 
# Kv = np.vstack(( np.array([[K[i][1] for i in range(N_h)]]).transpose())) 
# Kf = np.vstack(( np.array([[K[i][2] for i in range(N_h)]]).transpose())) 
# # Norms
# print("Kp gain norm : ", np.linalg.norm(Kp))
# print("Kv gain norm : ", np.linalg.norm(Kv))
# print("Kf gain norm : ", np.linalg.norm(Kf))

#######
# MPC #
#######
# Parameters
maxit= 1
T_tot = .5
plan_freq = 1000                      # MPC re-planning frequency (Hz)
ctrl_freq = 1000                      # Control - simulation - frequency (Hz)
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
K_gains = np.zeros((N_p, N_h, nx))    # Riccati feedback gains
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
    # Record riccati feedback gains
    K_gains[nb_replan, :, :] = np.array(ddp.K.tolist())
    # print(ddp.K.tolist()[-3][0])
    # Increment replan counter
    nb_replan += 1
  # Record and apply the 1st control
  U_des[i, :] = u_des
  # Measure new state from simulation (integrate)
  X_mea[i+1,:] = X_mea[i,:] + running_model.f(X_mea[i,:], u_des)*1e-3#+ 0.01*np.random.rand(3)# (X_mea[i,:] , u_des) 
  # Record desired state
  X_des[i+1, :] = x_des


# # Display magnitude of Riccati gains
# Kp, Kv, Kf = [], [], []
# for i in range(N_p):
#   K = K_gains[i, 0, :]
#   Kp.append(np.sum(K_gains[i, :, 0], axis=0))
#   # Kv.append(K_gains[i, 0, 1])
#   # Kf.append(K_gains[i, 0, 2])
# print(Kp)
# # plt.plot(np.linspace(0,T_tot-0.001, N_p), Kp, 'r')
# # plt.plot(np.linspace(0,T_tot-0.001, N_p), Kv, 'b')
# # plt.plot(np.linspace(0,T_tot-0.001, N_p), Kf, 'g')
# # plt.show()

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
p_pred = X_pred[:,:,0]
v_pred = X_pred[:,:,1]
f_pred = X_pred[:,:,2]
u_pred = U_pred[:,:,0]
    # State measurements (PyBullet)
p_mea = X_mea[:,0]
v_mea = X_mea[:,1]
f_mea = X_mea[:,2]
    # 'Desired' state = interpolated predictions
p_des = X_des[:,0]
v_des = X_des[:,1]
f_des = X_des[:,2]
    # 'Desired' control = interpolation of DDP ff torques 
u_des = U_des
# Create time spans for X and U
tspan_x = np.linspace(0, T_tot, N_tot+1)
tspan_u = np.linspace(0, T_tot-dt_ctrl, N_tot)
# Create figs and subplots
fig_x, ax_x = plt.subplots(nx+nu, 1)
if(with_predictions):
  # For each planning step in the trajectory
  for j in range(N_p):
    # Receding horizon = [j,j+N_h]
    t0_horizon = j*dt_plan
    tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
    tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
    # Set up lists of (x,y) points for predicted positions and velocities
    points_p = np.array([tspan_x_pred, p_pred[j,:]]).transpose().reshape(-1,1,2)
    points_v = np.array([tspan_x_pred, v_pred[j,:]]).transpose().reshape(-1,1,2)
    points_f = np.array([tspan_x_pred, f_pred[j,:]]).transpose().reshape(-1,1,2)
    points_u = np.array([tspan_u_pred, u_pred[j,:]]).transpose().reshape(-1,1,2)
    # Set up lists of segments
    segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
    segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
    segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
    segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap('Greys_r') 
    lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
    lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
    lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
    lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
    lc_p.set_array(tspan_x_pred)
    lc_v.set_array(tspan_x_pred) 
    lc_f.set_array(tspan_x_pred) 
    lc_u.set_array(tspan_u_pred)
    # Customize
    lc_p.set_linestyle('-')
    lc_v.set_linestyle('-')
    lc_f.set_linestyle('-')
    lc_u.set_linestyle('-')
    lc_p.set_linewidth(1)
    lc_v.set_linewidth(1)
    lc_f.set_linewidth(1)
    lc_u.set_linewidth(1)
    # Plot collections
    ax_x[0].add_collection(lc_p)
    ax_x[1].add_collection(lc_v)
    ax_x[2].add_collection(lc_f)
    ax_x[3].add_collection(lc_u)
    # Scatter to highlight points
    colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
    my_colors = cm(colors)
    ax_x[0].scatter(tspan_x_pred, p_pred[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
    ax_x[1].scatter(tspan_x_pred, v_pred[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
    ax_x[2].scatter(tspan_x_pred, f_pred[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
    ax_x[3].scatter(tspan_u_pred, u_pred[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
# Positions
ax_x[0].plot(tspan_x, p_des, 'b-', label='Desired')
ax_x[0].plot(tspan_x, p_mea, 'r-', label='Measured')
ax_x[0].set_ylabel('p (m)', fontsize=16)
ax_x[0].grid()
# Velocities
ax_x[1].plot(tspan_x, v_des, 'b-', label='Desired')
ax_x[1].plot(tspan_x, v_mea, 'r-', label='Measured')
ax_x[1].set_ylabel('v (m/s)', fontsize=16)
ax_x[1].grid()
# Forces
ax_x[2].plot(tspan_x, f_des, 'b-', label='Desired')
ax_x[2].plot(tspan_x, f_mea, 'r-', label='Measured')
ax_x[2].set_ylabel('f (N)', fontsize=16)
ax_x[2].grid()
# Controls
ax_x[3].plot(tspan_u, u_des, 'r-', label='Desired')
ax_x[3].set_ylabel('u (Nm)', fontsize=16)
ax_x[3].grid()
# Legend
ax_x[-1].set_xlabel('time (s)', fontsize=16)
handles_x, labels_x = ax_x[0].get_legend_handles_labels()
fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
fig_x.suptitle('State and control trajectories', size=16)
plt.show() 