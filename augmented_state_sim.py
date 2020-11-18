# Author: Sebastien
# NYU 2020
# Simulation of the "augmented state" approach for the point mass system 

import numpy as np
import crocoddyl
from point_mass_contact_model import ActionModelPointMassContact
from utils import animatePointMass, plotPointMass

# soft contact model params
K = 1e4  # stiffness
B = 1.      # damping
# Create IAM (integrate DAM with Euler)
dt = 1e-2 #5e-2
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

# Extract and plot trajectories
# plotPointMass(xs, us)
# Animate
# from IPython.display import HTML
# anim = animatePointMass(xs)
# HTML(anim.to_html5_video())

# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# Solve and retrieve X,U
done = ddp.solve([], [], 10)
plotPointMass(ddp.xs, ddp.us)

# from IPython.display import HTML
# anim = animatePointMass(ddp.xs)
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