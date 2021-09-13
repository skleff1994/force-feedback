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
from models.croco_IAMs import ActionModelPointMass
from utils import animatePointMass, plotPointMass
from core.kalman_filter import KalmanFilter

# Create IAM (integrate DAM with Euler)
dt = 1e-2
running_IAM = ActionModelPointMass(dt, 'euler')
terminal_IAM = ActionModelPointMass(0., 'euler')

# Initial conditions
p = 1.
v = 0.
x = np.matrix([p, v]).T
u = np.matrix([0.])

# Define shooting problem
T = 1000
problem = crocoddyl.ShootingProblem(x, [running_IAM]*T, terminal_IAM)

# Integrate (rollout)
us = [ u ]*T
xs = problem.rollout(us)

# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# Solve and retrieve X,U
done = ddp.solve([], [], 10)

# Create the filter 
kalman = KalmanFilter(running_IAM.Ad, running_IAM.Bd, np.eye(3), np.eye(3), np.eye(3))

# Simulate the thing
def simulate(model, solver, filter):
    '''
    Simulate the DDP policy
        model  : IAM object
        solver : ddp solver object
        filter : Kalman filter
    '''
    N = len(solver.us)
    xs, us = [], []
    xs.append(solver.xs[0])
    for i in range(N):
        # Gaussian noise on state (measurement)
        w = .1*np.random.normal(np.zeros(2), np.ones(2))
        print("w = ", w)
        xs[i] += w
        # Filter state 
        
        # Apply Riccati gain to correct 
        us.append(solver.us[i] + solver.k[i] - solver.K[i].dot(xs[i] - solver.xs[i+1]))
        xs.append(model.calc(x=xs[i], u=us[i])) 
    return xs, us


xs, us = simulate(running_IAM, ddp, kalman)

plotPointMass(xs, us)
animatePointMass(ddp.xs)

# Display norm of partial derivatives ???