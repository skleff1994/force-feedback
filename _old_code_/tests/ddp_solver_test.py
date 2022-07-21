# Title : ddp_solver_test.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Test custom DDP solver on point mass : offline trajectory generation + plots and animation

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

from models.dyn_models import PointMass
from models.cost_models import *
from core.ddp import DDPSolver

import numpy as np 
from matplotlib import pyplot as plt 

# Create point mass model
dt = 1e-2
model = PointMass(dt)
x_0 = np.array([[0],[-1]])

# Create DDP solver
T = 5.
ddp = DDPSolver(model, model.dt)
ddp.init_all(T)

# Add cost models
Q = np.eye(model.nx)
R = np.eye(model.nu)
x_ref = np.array([1,0])

# Running and terminal cost sums
running_cost = CostSum(model)
running_cost.add_cost(QuadTrackingCost(model, x_ref, 1.*Q))  
running_cost.add_cost(QuadCtrlRegCost(model, 1e-3*R))
terminal_cost = CostSum(model)
terminal_cost.add_cost(QuadTrackingCost(model, x_ref, 10.*Q))
# Add to OCP 
ddp.set_running_cost(running_cost)
ddp.set_terminal_cost(terminal_cost)

# Solve 
ddp.solve()
# Test a point mass rollout with zero input
us = ddp.us
xs = ddp.xs
# X, U = model.rollout(x_0, us)
# model.plot_traj(X,U)
from core_mpc import animatePointMass, plotPointMass
# animatePointMass(xs, sleep=10)
plotPointMass(xs, us)