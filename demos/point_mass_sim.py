
import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

from models.dyn_models import PointMass
from models.cost_models import QuadCtrlRegCost, QuadTrackingRunningCost, QuadTrackingTerminalCost
from core.ddp import DDPSolver

import numpy as np 
from matplotlib import pyplot as plt 

# Create point mass model
dt = 1e-2
model = PointMass(dt)
x_0 = np.array([[0],[0]])

# Create DDP solver
T = 1.
ddp = DDPSolver(model, model.dt)
ddp.init_all(T)

# Add cost models
Q = np.eye(model.nx)
R = np.eye(model.nu)
x_ref = np.array([[1],[0]])
ddp.add_running_cost(QuadTrackingRunningCost(model, x_ref, 1.*Q))
ddp.add_terminal_cost(QuadTrackingTerminalCost(model, x_ref, 10.*Q))
ddp.add_running_cost(QuadCtrlRegCost(model, 1e-6*R))

# Solve 
ddp.solve()
# Test a point mass rollout with zero input
us = ddp.us
xs = ddp.xs
# X, U = model.rollout(x_0, us)
# # model.plot_traj(X,U)
from utils import animatePointMass, plotPointMass
# animatePointMass(X)
plotPointMass(xs, us)
