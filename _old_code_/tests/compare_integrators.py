# Title : point_mass_sim.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# simple point mass simulation with custom DDP solver 

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

from models.dyn_models import PointMass
from core.ddp import DDPSolver

import numpy as np 
from matplotlib import pyplot as plt 

# Create point mass model
dt = 5e-1
T = 10000
model1 = PointMass(dt=dt, integrator='exact')
model2 = PointMass(dt=dt, integrator='euler')
model3 = PointMass(dt=dt, integrator='rk4')

x_0 = np.array([[0],[-1]])
us = [np.array([0.])]*T

X1, U1 = model1.rollout(x_0, us)
X2, U2 = model2.rollout(x_0, us)
X3, U3 = model3.rollout(x_0, us)

# # # model.plot_traj(X,U)
from core_mpc_utils import animatePointMass, plotPointMass
# animatePointMass(xs, sleep=10)
plotPointMass(X1, U1)
plotPointMass(X2, U2)
plotPointMass(X3, U3)