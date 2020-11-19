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
from models.croco_IAMs import ActionModelPointMassContact
from utils import animatePointMass, plotPointMass

# soft contact model params
K = 1e6     # stiffness
B = 0.      # damping
# Create IAM (integrate DAM with Euler)
dt = 1e-4 #5e-2
running_IAM = ActionModelPointMassContact(K=K, B=B, dt=dt, integrator='rk4')
terminal_IAM = ActionModelPointMassContact(K=K, B=B, dt=dt, integrator='rk4')

# Initial conditions
p0 = 0.               # reference position (contact point)
p = 1.                # initial position
v = 0.                # initial velocity 
lmb = -K*(p-p0) - B*v # initial contact force
x = np.matrix([p, v, lmb]).T
u = np.matrix([0.])
# Define shooting problem
T = 200
problem = crocoddyl.ShootingProblem(x, [running_IAM]*T, terminal_IAM)
# Integrate (rollout)
us = [ u ]*T
xs = problem.rollout(us)

# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# Solve and retrieve X,U
done = ddp.solve([], [], 10)
plotPointMass(ddp.xs, ddp.us)

# from IPython.display import HTML
anim = animatePointMass(ddp.xs, sleep=10)
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