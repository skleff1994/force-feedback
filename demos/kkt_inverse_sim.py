# Title : kkt_inverse_sim.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Test of the "kkt inverse" approach for force feedback on the point mass system 
# i.e. under rigit contact assumption

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.dyn_models import PointMass
from models.cost_models import *
from models.croco_IAMs import ActionModelPM

from utils import animatePointMass, plotPointMass

# Create dynamics model
dt = 1e-3
K = 1000
B = 10.
model = PointMass(dt=dt, integrator='rk4')

# Running and terminal cost models
running_cost = CostSum(model)
terminal_cost = CostSum(model)
  # Setup cost terms
x_ref = np.array([0., 0.])
running_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*np.eye(model.nx)))  
running_cost.add_cost(QuadCtrlRegCost(model, .1*np.eye(model.nu)))
terminal_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*np.eye(model.nx)))
  # IAMs for Crocoddyl
running_IAM = ActionModelPM(model, running_cost, dt) 
terminal_IAM = ActionModelPM(model, terminal_cost, 0.) 
# Define shooting problem
# Initial conditions
p0 = 0.               # reference position (contact point)
p = 1.                # initial position
v = 1.                # initial velocity 
# lmb = -K*(p-p0) - B*v # initial contact force
x0 = np.matrix([p, v]).T
u0 = np.matrix([0.])
T = 1000
problem = crocoddyl.ShootingProblem(x0, [running_IAM]*T, terminal_IAM)
# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])
# Solve and retrieve X,U
done = ddp.solve([], [], 10)
X_plan = np.array(ddp.xs)
U_plan = np.array(ddp.us)
U_plan_force = np.array(ddp.us) # adding the force feedback term ?

# How to get the force ? Use pinocchio integrator
# # Add noise on DDP trajectory and filter it to test Kalman filter
# N = U_plan.shape[0]
# X_mea = []   # measurements (noise on plan)
# X_mea = []
# # Noise params
# mean = np.zeros(2)
# std = np.array([0.05, 0.05])
# for i in range(N):
#     # Generate noisy state  
#     X_mea.append(X_plan[i] + np.random.normal(mean, std) )
#     # Apply Riccati gains to correct
#     X_mea


# from utils import plotPointMass
# plotPointMass(X_real, U_real)

# # from IPython.display import HTML
# anim = animatePointMass(ddp.xs, sleep=10)
# # HTML(anim.to_html5_video())
# # xs, us = simulate(running_IAM, ddp)
# # plotPointMass(xs, us)

# # Display norm of partial derivatives 
# K1 = np.vstack(( np.array([[ddp.K[i][0] for i in range(T)]]).transpose())) # position gains
# K2 = np.vstack(( np.array([[ddp.K[i][1] for i in range(T)]]).transpose())) # velocity gains
# K3 = np.vstack(( np.array([[ddp.K[i][2] for i in range(T)]]).transpose())) # contact force gains
# # Norms
# print(np.linalg.norm(K1))
# print(np.linalg.norm(K2))
# print(np.linalg.norm(K3))