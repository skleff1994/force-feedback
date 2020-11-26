# Title : observer_sim.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Test of the "observer" approach for force feedback on the point mass system 
import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.dyn_models import *
from models.cost_models import *

from models.croco_IAMs_new import ActionModelPM
from utils import animatePointMass, plotPointMass, plotFiltered
from core.kalman_filter import KalmanFilter

# Create model
dt = 1e-2
model = PointMass(dt)

# # Add cost models
x_ref = np.array([[1],[0]])
# # Running and terminal cost sums
running_cost = CostSum(model)
running_cost.add_cost(QuadTrackingRunningCost(model, x_ref, 1.*np.eye(model.nx)))  
running_cost.add_cost(QuadCtrlRegCost(model, 1e-3*np.eye(model.nu)))
# terminal_cost = CostSum(model)
# terminal_cost.add_cost(QuadTrackingTerminalCost(model, x_ref, 10.*np.eye(model.nx)))
# Define Croco IAM from running and terminal cost models
running_IAM = ActionModelPM(model, running_cost, dt) #model, running_cost)
# terminal_IAM = ActionModelPM(model, running_cost, dt) #model, terminal_cost)

# Initial conditions
p = 1.
v = 0.
x = np.matrix([p, v]).T
u = np.matrix([0.])

# Define shooting problem
T = 1000
problem = crocoddyl.ShootingProblem(x, [running_IAM]*T, running_IAM)

# Integrate (rollout)
us = [ u ]*T
xs = problem.rollout(us)

# # Create the DDP solver and setup callbacks
# ddp = crocoddyl.SolverDDP(problem)
# ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# # Solve and retrieve X,U
# done = ddp.solve([], [], 10)
# X_real = np.array(ddp.xs)
# U_real = np.array(ddp.us)

# # Create the filter 
# Q_cov = .001*np.eye(2)
# H = running_IAM.Ad
# R_cov = 0.1*np.eye(2)

# kalman = KalmanFilter(running_IAM, Q_cov, H, R_cov)

# # Add noise on DDP trajectory and filter it to test Kalman filter
# N = U_real.shape[0]
# Y_mea = []   # measurements
# X_hat = []   # state estimates
# P_cov = []   # covariance estimates
# K_gain = []  # optimal Kalman gains
# Y_err = []
# # Initialize 
# P_cov.append(np.eye(2))
# X_hat.append(X_real[0])
# # Noise params
# mean = np.zeros(2)
# std = .05*np.ones(2)
# for i in range(N):
#     # Gaussian noise on state (measurement)
#     Y_mea.append(X_real[i] + np.random.normal(mean, std))
#     # Filter
#     # print(" In x = ", X_hat[i].T)
#     # print(" In u = ", U_real[i])
#     x, P, K, y = kalman.step(X_hat[i], P_cov[i], U_real[i], Y_mea[i])
#     # Record estimates + gais
#     X_hat.append(x)
#     P_cov.append(P)
#     K_gain.append(K)
#     Y_err.append(y)

# plotFiltered(Y_mea, X_hat, X_real)
# # # animatePointMass(ddp.xs)

# # Display norm of partial derivatives ???