# Author: Sebastien
# NYU 2020
# Simulation of the "observer" approach for the point mass system 

import numpy as np
import crocoddyl
from point_mass_model import ActionModelPointMass
from utils import animatePointMass, plotPointMass
from kalman_filter import KalmanFilter

# Create IAM (integrate DAM with Euler)
dt = 1e-2
running_IAM = ActionModelPointMass(dt, 'euler')
terminal_IAM = ActionModelPointMass(dt, 'euler')

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

# # Extract and plot trajectories
# plotPointMass(xs, us)

# # Animate
# from IPython.display import HTML
# anim = animatePointMass(xs)
# HTML(anim.to_html5_video())

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
        w = .1*np.random.normal(np.zeros(3), np.ones(3))
        print("w = ", w)
        xs[i] += w
        # Filter state 
        
        # Apply Riccati gain to correct 
        us.append(solver.us[i] + solver.k[i] - solver.K[i].dot(xs[i] - solver.xs[i+1]))
        xs.append(model.calc(x=xs[i], u=us[i])) 
    return xs, us


xs, us = simulate(running_IAM, ddp)
plotPointMass(xs, us)

plotPointMass(ddp.xs, ddp.us)
from IPython.display import HTML
anim = animatePointMass(ddp.xs)
# HTML(anim.to_html5_video())


# Display norm of partial derivatives ???