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
from models.croco_IAMs import ActionModelPM

from utils import animatePointMass, plotPointMass

# Create dynamics model
dt = 1e-2
K = 10
B = 1.
model = PointMassContact(K=K, B=B, dt=dt, p0=0., integrator='euler')
p0 = model.p0 

# Running and terminal cost models
running_cost = CostSum(model)
terminal_cost = CostSum(model)
  # Setup cost terms
p_ref = 10.
v_ref = 2.
lmb_ref = -K*(p_ref - p0) - B*v_ref
x_ref = np.array([p_ref, v_ref, lmb_ref])
print("x_ref = ", x_ref)
Q = np.eye(model.nx)
# Q[-1,-1] = 1
print("Q = ", Q)
# running_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*Q))  
# running_cost.add_cost(QuadCtrlRegCost(model, 1.*np.eye(model.nu)))
terminal_cost.add_cost(QuadTrackingCost(model, x_ref, 1000.*Q))
  # IAMs for Crocoddyl
running_IAM = ActionModelPM(model, running_cost, dt) 
terminal_IAM = ActionModelPM(model, terminal_cost, 0.) 
# Define shooting problem
# Initial conditions
p = 1.                # initial position
v = 1.                # initial velocity 
lmb = -K*(p-p0) - B*v # initial contact force
x0 = np.matrix([p, v, lmb]).T
u0 = np.matrix([0.])
T = 1000
problem = crocoddyl.ShootingProblem(x0, [running_IAM]*T, terminal_IAM)
# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])
# Solve and retrieve X,U
done = ddp.solve([], [], 10)
X_real = np.array(ddp.xs)
U_real = np.array(ddp.us)

# print("Ac = ", model.Ac)
# print("Bc = ", model.Bc)
# print("Ad_exact = ", np.eye(model.nx) + model.dt*model.Ac)
# print("Bd_exact = ", model.dt*model.Bc + .5*model.dt**2*model.Ac.dot(model.Bc))
# print("Ad_euler = ", np.eye(model.nx) + model.dt*model.Ac)
# print("Bd_euler = ", model.dt*model.Bc)

# # Compute integration residual on force
# res = np.zeros(T)
# for i in range(T):
#   res[i] = X_real[i,2] + K*(X_real[i,0]-p0) + B*X_real[i,1]
#   # print("Step "+str(i)+" : "+str(res[i]))
# import matplotlib.pyplot as plt
# plt.plot(res)

from utils import plotPointMass
plotPointMass(X_real, U_real, ref=x_ref)

# from IPython.display import HTML
# anim = animatePointMass(ddp.xs, sleep=10)
# HTML(anim.to_html5_video())
# xs, us = simulate(running_IAM, ddp)
# plotPointMass(xs, us)

# Display norm of partial derivatives 
K1 = np.vstack(( np.array([[ddp.K[i][0] for i in range(T)]]).transpose())) # position gains
K2 = np.vstack(( np.array([[ddp.K[i][1] for i in range(T)]]).transpose())) # velocity gains
K3 = np.vstack(( np.array([[ddp.K[i][2] for i in range(T)]]).transpose())) # contact force gains
# Norms
print(np.linalg.norm(K1))
print(np.linalg.norm(K2))
print(np.linalg.norm(K3))

