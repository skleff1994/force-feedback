# Title : kalman_filter_test.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Testing script for Kalman filter (using point mass)

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.dyn_models import PointMassPartialObs
from models.cost_models import *

from models.croco_IAMs import ActionModel
from core_mpc import animatePointMass, plotPointMass, plotFiltered
from core.kalman_filter import KalmanFilter

# Create dynamics model
dt = 1e-2
K = 1000
B = 10.
model = PointMassPartialObs(dt=dt, K=K, B=B, integrator='euler')
# Running and terminal cost models
running_cost = CostSum(model)
terminal_cost = CostSum(model)
  # Setup cost terms
x_ref = np.array([0., 0.])
# running_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*np.eye(model.nx)))  
running_cost.add_cost(QuadCtrlRegCost(model, 1e-4*np.eye(model.nu)))
terminal_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*np.eye(model.nx)))
  # IAMs for Crocoddyl
running_IAM = ActionModel(model, running_cost) 
terminal_IAM = ActionModel(model, terminal_cost) 
# Define shooting problem
# Initial conditions
p0 = 0.               # reference position (contact point)
p = 1.                # initial position
v = 1.                # initial velocity 
lmb = -K*(p-p0) - B*v # initial contact force
x0 = np.matrix([p, v]).T
T = 1000
problem = crocoddyl.ShootingProblem(x0, [running_IAM]*T, terminal_IAM)
# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])
# Solve and retrieve X,U
done = ddp.solve([], [], 10)
X_real = np.array(ddp.xs)
U_real = np.array(ddp.us)
# # Plot solution
# from utils import plotPointMass
# plotPointMass(X_real, U_real)

# Create the filter 
Q_cov = .01*np.eye(2) # Process noise cov
R_cov = 0.01*np.eye(2)  # Measurement noise cov
kalman = KalmanFilter(model, Q_cov, R_cov)

# Add noise on DDP trajectory and filter it to test Kalman filter
N = U_real.shape[0]
Y_mea = []   # measurements
X_hat = []   # state estimates
P_cov = []   # covariance estimates
K_gain = []  # optimal Kalman gains
Y_err = []
# Initialize 
P_cov.append(np.eye(2))
X_hat.append(X_real[0])
# Noise params
mean = np.zeros(2)
std = np.array([0.05, 100])
for i in range(N):
    # Generate noisy force measurement 
      # Ideal visco-elastic force and real position
    lmb = -model.K*(X_real[i][0] - p0) - model.B*X_real[i][1]
    pos = X_real[i][0]
      # Noise them out
    Y_mea.append(np.array([pos, lmb]) + np.random.normal(mean, std) )
    # Filter
    x, P, K, y = kalman.step(X_hat[i], P_cov[i], U_real[i], Y_mea[i])
    # Record estimates + gais
    X_hat.append(x)
    P_cov.append(P)
    K_gain.append(K)
    Y_err.append(y)

# Display Kalman gains magnitude
dP_dP = np.vstack(( np.array([[K_gain[i][0,0] for i in range(N)]]).transpose())) 
dP_dF = np.vstack(( np.array([[K_gain[i][0,1] for i in range(N)]]).transpose())) 
dV_dP = np.vstack(( np.array([[K_gain[i][1,0] for i in range(N)]]).transpose())) 
dV_dF = np.vstack(( np.array([[K_gain[i][1,1] for i in range(N)]]).transpose())) 

# Norms
print("dP_dP Kalman gain norm : ", np.linalg.norm(dP_dP))
print("dP_dF Kalman gain norm : ", np.linalg.norm(dP_dF))
print("dV_dP Kalman gain norm : ", np.linalg.norm(dV_dP))
print("dV_dF Kalman gain norm : ", np.linalg.norm(dV_dF))


# Plot results 
import matplotlib.pyplot as plt
# Extract trajectories and reshape
T = len(Y_mea)
ny = len(Y_mea[0])
nx = len(X_real[0])
tspan = np.linspace(0, T*dt, T+1)
Y_mea = np.array(Y_mea).reshape((T, ny))
X_hat = np.array(X_hat).reshape((T+1, nx))
X_real = np.array(X_real).reshape((T+1, nx))
# Create fig
fig, ax = plt.subplots(3,1)
# Plot position
ax[0].plot(tspan[:T], Y_mea[:,0], 'b-', linewidth=2, alpha=.5, label='Measured')
ax[0].plot(tspan, X_hat[:,0], 'r-', linewidth=3, alpha=.8, label='Estimated')
ax[0].plot(tspan, X_real[:,0], 'k-.', linewidth=2, label='Ground truth')
ax[0].set_title('Position p', size=16)
ax[0].set_ylabel('p (m)', fontsize=16)
ax[0].grid()
# Plot velocities
ax[1].plot(tspan, X_hat[:,1], 'r-', linewidth=3, alpha=.8, label='Estimated')
ax[1].plot(tspan, X_real[:,1], 'k-.', linewidth=2, label='Ground truth')
ax[1].set_title('Velocity p', size=16)
ax[1].set_ylabel('v (m/s)', fontsize=16)
ax[1].grid()
# Plot force
ax[2].plot(tspan[:T], Y_mea[:,1], 'b-', linewidth=2, alpha=.5, label='Measured')
ax[2].set_title('Force lmb', size=16)
ax[2].set_ylabel('lmb (N)', fontsize=16)
ax[2].grid()
# Legend
ax[-1].set_xlabel('time (s)',fontsize=16)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 16})
fig.suptitle('Kalman-filtered point mass trajectory', size=16)
plt.show()
    