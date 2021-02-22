# Title : test_IAMPM.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Test of the "augmented state" approach for force feedback on the point mass system 

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.dyn_models import PointMassLPF
from models.cost_models import *
from models.croco_IAMs import ActionModelPointMassObserver
from core.kalman_filter import KalmanFilter
from utils import animatePointMass, plotPointMass


# Action model for point mass
dt = 1e-2 #5e-3
N_h = 20
integrator='euler'
running_model = ActionModelPointMassObserver(dt=dt, integrator=integrator)
# running_model.w_x = 1e-1
# running_model.w_xreg = 1e-2
running_model.w_ureg = 1e-4

terminal_model = ActionModelPointMassObserver(dt=0.)
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

# PLOT
import matplotlib.pyplot as plt
p = X[:,0]
v = X[:,1]
u = U
# Create time spans for X and U
tspan_x = np.linspace(0, N_h*dt, N_h+1)
tspan_u = np.linspace(0, N_h*dt, N_h)
# Create figs and subplots
fig_x, ax_x = plt.subplots(3, 1)
# fig_u, ax_u = plt.subplots(1, 1)
# Plot joints
ax_x[0].plot(tspan_x, p, 'b-', label='pos')
ax_x[0].set(xlabel='t (s)', ylabel='p (m)')
ax_x[0].grid()
ax_x[1].plot(tspan_x, v, 'g-', label='vel')
ax_x[1].set(xlabel='t (s)', ylabel='v (m/s)')
ax_x[1].grid()
ax_x[2].plot(tspan_u, u, 'r-', label='torque')
ax_x[2].set(xlabel='t (s)', ylabel='tau (Nm)')
ax_x[2].grid()
# Legend
handles_x, labels_x = ax_x[0].get_legend_handles_labels()
fig_x.legend(loc='upper right', prop={'size': 16})
# Titles
fig_x.suptitle('State - Control trajectories', size=16)
plt.show()

# TEST FILTERING
# Create the filter 
Q_cov = np.eye(2)   # Process noise cov
R_cov = .0001*np.eye(2)        # Measurement noise cov
kalman = KalmanFilter(running_model, Q_cov, R_cov)
# Observation model (spring-damper )
K = 1000. 
B = 2*np.sqrt(K)
p0 = 0.
# Add noise on DDP trajectory and filter it to test Kalman filter
nx = running_model.nx
ny = running_model.ny
Y_mea = np.zeros((N_h, ny))      # measurements
X_hat = np.zeros((N_h+1, nx))      # state estimates
P_cov = np.zeros((N_h+1, nx, nx))  # covariance estimates
K_gain = np.zeros((N_h+1, nx, nx)) # optimal Kalman gains
Y_err = np.zeros((N_h+1, ny))      
X_real = np.reshape(X[:N_h], Y_mea.shape) # Ground truth state trajectory
# Measurement noise model
mean = np.zeros(2)
std_p = 0.01  #np.array([0.005, N_h])
std_f = 10    #np.array([0.005, N_h])

# ESTIMATION LOOP (offline)
for i in range(N_h):
    print("Step "+str(i)+"/"+str(N_h))
    # Generate noisy force measurement 
      # Ideal visco-elastic force and real position + Noise them out
    wp,_ = np.random.normal(mean, std_p) 
    wf,_ = np.random.normal(mean, std_f)
    Y_mea[i,:] = (np.array([X_real[i,0], -K*(X_real[i,0]- 0.) - B*X_real[i,1]]) + np.array([wp, wf]) )
    # Filter and record
    X_hat[i+1,:], P_cov[i+1,:,:], K_gain[i+1,:,:], Y_err[i+1,:] = kalman.step(X_hat[i,:], P_cov[i,:,:], U[i,:], Y_mea[i,:])

# # Display Kalman gains magnitude
# dP_dP = np.vstack(( np.array([[K_gain[i][0,0] for i in range(N_h)]]).transpose())) 
# dP_dF = np.vstack(( np.array([[K_gain[i][0,1] for i in range(N_h)]]).transpose())) 
# dV_dP = np.vstack(( np.array([[K_gain[i][1,0] for i in range(N_h)]]).transpose())) 
# dV_dF = np.vstack(( np.array([[K_gain[i][1,1] for i in range(N_h)]]).transpose())) 
# # Norms
# print("dP_dP Kalman gain norm : ", np.linalg.norm(dP_dP))
# print("dP_dF Kalman gain norm : ", np.linalg.norm(dP_dF))
# print("dV_dP Kalman gain norm : ", np.linalg.norm(dV_dP))
# print("dV_dF Kalman gain norm : ", np.linalg.norm(dV_dF))

# # Plot results 
# import matplotlib.pyplot as plt
# # Extract trajectories and reshape
# tspan = np.linspace(0, N_h*dt - dt, N_h+1)
# # Create fig
# fig, ax = plt.subplots(3,1)
# # Plot position
# ax[0].plot(tspan[:N_h], Y_mea[:,0], 'b-', linewidth=2, alpha=.5, label='Measured')
# ax[0].plot(tspan, X_hat[:,0], 'r-', linewidth=3, alpha=.8, label='Estimated')
# ax[0].plot(tspan[:N_h], X_real[:,0], 'k-.', linewidth=2, label='Ground truth')
# # ax[0].set_title('Position p', size=16)
# ax[0].set_ylabel('p (m)', fontsize=16)
# ax[0].grid()
# # Plot velocities
# ax[1].plot(tspan, X_hat[:,1], 'r-', linewidth=3, alpha=.8, label='Estimated')
# ax[1].plot(tspan[:N_h], X_real[:,1], 'k-.', linewidth=2, label='Ground truth')
# # ax[1].set_title('Velocity p', size=16)
# ax[1].set_ylabel('v (m/s)', fontsize=16)
# ax[1].grid()
# # Plot force
# ax[2].plot(tspan[:N_h], Y_mea[:,1], 'b-', linewidth=2, alpha=.5, label='Measured')
# # ax[2].set_title('Force lmb', size=16)
# ax[2].set_ylabel('lmb (N)', fontsize=16)
# ax[2].grid()
# # Legend
# ax[-1].set_xlabel('time (s)',fontsize=16)
# handles, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', prop={'size': 16})
# fig.suptitle('Kalman-filtered point mass trajectory', size=16)
# plt.show()

#######
# MPC #
#######
# Parameters
maxit= 1
T_tot = .1
plan_freq = 1000                      # MPC re-planning frequency (Hz)
ctrl_freq = 1000                      # Control - simulation - frequency (Hz)
N_tot = int(T_tot*ctrl_freq)          # Total number of control steps in the simulation (s)
N_p = int(T_tot*plan_freq)            # Total number of OCPs (replan) solved during the simulation
T_h = N_h*dt                          # Duration of the MPC horizon (s)
# Init data
nx, nq, nv, nu = 2, 1, 1, 1
# Control
X_mea = np.zeros((N_tot+1, nx))       # Measured states 
X_des = np.zeros((N_tot+1, nx))       # Desired states
U_des = np.zeros((N_tot, nu))         # Desired controls 
X_pred = np.zeros((N_p, N_h+1, nx))   # MPC predictions (state)
U_pred = np.zeros((N_p, N_h, nu))     # MPC predictions (control)
# Estimation
Y_mea = np.zeros((N_tot, ny))        # output measurements
X_hat = np.zeros((N_tot+1, nx))      # state estimates
X_real = np.zeros((N_tot+1, nx))     # real (unknown) state (for simulation purpose)
P_cov = np.zeros((N_tot+1, nx, nx))  # covariance estimates
K_gain = np.zeros((N_tot+1, nx, nx)) # optimal Kalman gains
Y_err = np.zeros((N_tot+1, ny))      # error in predicted measurements

# Initialize 
X_real[0, :] = x0 # Initial true state 
X_des[0, :] = x0  # Initial desired state 
X_hat[0, :] = x0  # Initial estimate = ground truth

# Replan counter
nb_replan = 0

# SIMULATION LOOP
# Simulation loop (at control rate)
for i in range(N_tot): 
  print("  ")
  print("Sim step "+str(i)+"/"+str(N_tot))
  
  # Measurement 
    # Ideal measurement of visco-elastic force and real position (measuring at ctrl frequency)
    # Noise it out (for simulation) NO NOISE right now so measurement = real (hidden) state 
  wp,_ = np.random.normal(mean, std_p) 
  wf,_ = np.random.normal(mean, std_f)
  Y_mea[i,:] = np.array([X_real[i,0], -K*(X_real[i,0]- 0) - B*X_real[i,1]]) + np.array([wp, wf])

  # ESTIMATE state and SOLVE OCP if we are in a planning cycle
  if(i%int(ctrl_freq/plan_freq) == 0):

    print("  Replan step "+str(nb_replan)+"/"+str(N_p))

    # ESTIMATION 
    # Filter measurement to reconstruct state
    x_hat, p_cov, k_gain, y_err = kalman.step(X_hat[i,:], P_cov[i,:,:], U_des[i,:], Y_mea[i,:])

    # CONTROL
    # Set initial state to measured state
    ddp.problem.x0 = X_hat[i,:]
    # Warm-start solution
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
    xs_init[0] = X_hat[i,:]
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
  # Record next real (unknown) state = integrate dynamics 
  X_real[i+1,:] = X_real[i,:] + running_model.f(X_real[i,:], u_des)*dt 
  # Record next desired state
  X_des[i+1, :] = x_des
  # Record filter output
  X_hat[i+1,:] = x_hat 
  P_cov[i+1,:,:] = p_cov 
  K_gain[i+1,:,:] = k_gain 
  Y_err[i+1,:] = y_err

# Final estimation step ? or remove last element of Kalman vars

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
p_pred = X_pred[:,:,0]
v_pred = X_pred[:,:,1]
u_pred = U_pred[:,:,0]
# Create time spans for X and U
tspan_x = np.linspace(0, T_tot, N_tot+1)
tspan_u = np.linspace(0, T_tot-dt_ctrl, N_tot)
# Create figs and subplots
fig_x, ax_x = plt.subplots(4, 1)
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
    points_u = np.array([tspan_u_pred, u_pred[j,:]]).transpose().reshape(-1,1,2)
    # Set up lists of segments
    segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
    segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
    segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap('Greys_r') 
    lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
    lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
    lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
    lc_p.set_array(tspan_x_pred)
    lc_v.set_array(tspan_x_pred) 
    lc_u.set_array(tspan_u_pred)
    # Customize
    lc_p.set_linestyle('-')
    lc_v.set_linestyle('-')
    lc_u.set_linestyle('-')
    lc_p.set_linewidth(1)
    lc_v.set_linewidth(1)
    lc_u.set_linewidth(1)
    # Plot collections
    ax_x[0].add_collection(lc_p)
    ax_x[1].add_collection(lc_v)
    ax_x[3].add_collection(lc_u)
    # Scatter to highlight points
    colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
    my_colors = cm(colors)
    ax_x[0].scatter(tspan_x_pred, p_pred[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
    ax_x[1].scatter(tspan_x_pred, v_pred[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
    ax_x[3].scatter(tspan_u_pred, u_pred[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
# Positions
ax_x[0].plot(tspan_u, Y_mea[:,0], 'm-', linewidth=2, alpha=.5, label='Measured')
ax_x[0].plot(tspan_x, X_hat[:,0], 'b-', linewidth=3, alpha=.8, label='Estimated')
ax_x[0].plot(tspan_x, X_real[:,0], 'k-.', linewidth=1, label='Ground truth')
ax_x[0].plot(tspan_x, X_des[:,0], 'y--', alpha=0.8, label='Desired')
ax_x[0].set_ylabel('p (m)', fontsize=16)
ax_x[0].grid()
# Velocities
ax_x[1].plot(tspan_x, X_hat[:,1], 'b-', linewidth=3, alpha=.8, label='Estimated')
ax_x[1].plot(tspan_x, X_real[:,1], 'k-.', linewidth=1, label='Ground truth')
ax_x[1].plot(tspan_x, X_des[:,1], 'y--', alpha=0.8, label='Desired')
ax_x[1].set_ylabel('v (m/s)', fontsize=16)
ax_x[1].grid()
# Forces
ax_x[2].plot(tspan_u, Y_mea[:,1], 'm-', linewidth=2, alpha=.5, label='Measured')
ax_x[2].set_ylabel('f (N)', fontsize=16)
ax_x[2].grid()
# Controls
ax_x[3].plot(tspan_u, U_des, 'y--', alpha=0.8, label='Desired')
ax_x[3].set_ylabel('u (N)', fontsize=16)
ax_x[3].grid()
# Legend
ax_x[-1].set_xlabel('time (s)', fontsize=16)
handles_x, labels_x = ax_x[0].get_legend_handles_labels()
fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
fig_x.suptitle('State and control trajectories', size=16)
plt.show() 
