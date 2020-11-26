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
from models.croco_IAMs import ActionModelPointMass
from utils import animatePointMass, plotPointMass, plotFiltered
from core.kalman_filter import KalmanFilter

### GENERATE TRAJECTORY ###
# Create IAM 
dt = 1e-2
running_IAM = ActionModelPointMass(dt)
terminal_IAM = ActionModelPointMass(dt)
# Initial conditions
p = 1.
v = 0.
x = np.matrix([p, v]).T
u = np.matrix([0.])
# Define shooting problem
T = 1000
problem = crocoddyl.ShootingProblem(x, [running_IAM]*T, terminal_IAM)
# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])
# Solve and retrieve X,U
done = ddp.solve([], [], 10)
X_real = np.array(ddp.xs)
U_real = np.array(ddp.us)

### SETUP FILTER ###
# Process and measurement noise covariances+ observation matrix
Q_cov = .001*np.eye(2)
R_cov = 0.1*np.eye(2)
H = running_IAM.Ad
# Filter
kalman = KalmanFilter(running_IAM, Q_cov, H, R_cov)
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
std = .05*np.ones(2)
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