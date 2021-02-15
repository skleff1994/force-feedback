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

from models.dyn_models import PointMassLPF
from models.cost_models import *
from models.croco_IAMs import ActionModel

from utils import animatePointMass, plotPointMass

# Create dynamics model
dt = 5e-2
model = PointMassLPF(dt=dt, k=1)
# Test LPF
# X,U = model.rollout(np.array([0,0,0]), np.ones(100))
# model.plot_traj(X,U)

# Running and terminal cost models
running_cost = CostSum(model)
terminal_cost = CostSum(model)
  # Setup cost terms
x_ref = np.array([0., 0., 0.])
print("REF. ORIGIN = ", x_ref)
Q = np.eye(model.nx)
running_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*Q))  
running_cost.add_cost(QuadCtrlRegCost(model, 0.001*np.eye(model.nu)))
terminal_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*Q))
  # IAMs for Crocoddyl
running_IAM = ActionModel(model, running_cost) 
terminal_IAM = ActionModel(model, terminal_cost) 
# Define shooting problem
# Initial conditions
x0 = np.matrix([1., 0., 0.]).T
u0 = np.matrix([0.])
print("Initial state = ", x0)
T = 200
problem = crocoddyl.ShootingProblem(x0, [running_IAM]*T, terminal_IAM)
print(" Initial guess = x0 and u0 = quasiStatic(x0) ")
xs0 = [x0]*(T+1)
us0 = problem.quasiStatic(xs0[:-1])
# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])
# Solve and retrieve X,U
done = ddp.solve([], [], 100)
X_real = np.array(ddp.xs)
U_real = np.array(ddp.us)
# Plot result
model.plot_traj(X_real, U_real, ref=x_ref)

# What about force ?

# contactPhase = IntegratedActionModelLPF(contactDifferentialModel, dt)
# contactPhase.set_alpha(f_c)
# contactPhase.u_lb = - robot_model.effortLimit
# contactPhase.u_ub = robot_model.effortLimit

# and then as usual you could create the problem with the IAM

# problem_with_contact = crocoddyl.ShootingProblem(x0,
#                                                 [contactPhase] * contactNodes + [runningModel] * flyingNodes,
#                                                 terminalModel)

# In the code the state in the IAM is the augmented one, so u is the unfiltered torque (which is only bounded inside the actuation limits either with a barrier cost or with the projection of the BoxFDDP), while the last part of the state is the torque with the filtering

# To work more confortably with this change of notation I was using another class to just select the "real" torque that goes to the system

# class extractDDPLPF():
#         def __init__(self, ddp, nu):
#                 self.xs = np.array(ddp.xs)[:,:-nu]
#                 self.us = np.array(ddp.xs)[1:,-nu:]
#                 self.w = ddp.us
#                 self.robot_model = ddp.robot_model
#                 self.problem = ddp.problem

# ddpLPF = extractDDPLPF(ddp, actuation.nu)
