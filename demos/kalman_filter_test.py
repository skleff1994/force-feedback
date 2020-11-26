# Title : kalman_filter_test.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Test custom Kalman filter on point mass : generate noisy traj with Crocoddyl + filter and plot

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.dyn_models import *
from models.cost_models import *
from models.croco_IAMs import ActionModelPM

from utils import animatePointMass, plotPointMass, plotFiltered
from core.kalman_filter import KalmanFilter

### GENERATE TRAJECTORY ###
# Create dynamics model
dt = 1e-2
model = PointMass(dt)
# Running and terminal cost models
running_cost = CostSum(model)
terminal_cost = CostSum(model)
  # Setup cost terms
x_ref = np.array([0., 0.])
running_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*np.eye(model.nx)))  
running_cost.add_cost(QuadCtrlRegCost(model, 1e-3*np.eye(model.nu)))
terminal_cost.add_cost(QuadTrackingCost(model, x_ref, 10.*np.eye(model.nx)))
  # IAMs for Crocoddyl
running_IAM = ActionModelPM(model, running_cost, dt) 
terminal_IAM = ActionModelPM(model, terminal_cost, 0.) 
# Define shooting problem
x0 = np.array([1., 1.]).T
u0 = np.array([0.])
T = 500
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

### SETUP FILTER ###
# Process and measurement noise covariances+ observation matrix
Q_cov = .001*np.eye(2)
R_cov = 0.1*np.eye(2)
# Filter
kalman = KalmanFilter(model, Q_cov, R_cov)

### GENERATE NOISE AND TEST FILTER ###
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
std = np.array([.01, .05]) #*np.ones(2)
for i in range(N):
    # Gaussian noise on state (measurement)
    Y_mea.append(X_real[i] + np.random.normal(mean, std))
    # Filter
    x, P, K, y = kalman.step(X_hat[i], P_cov[i], U_real[i], Y_mea[i])
    # Record estimates + gais
    X_hat.append(x)
    P_cov.append(P)
    K_gain.append(K)
    Y_err.append(y)
    
# Plot results
plotFiltered(Y_mea, X_hat, X_real)