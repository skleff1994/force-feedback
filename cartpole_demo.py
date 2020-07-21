from ddp_problem import DDPProblem
from dynamics import CartPole
from cost_model import QuadCost
from ddp_solver import DDPSolver
import numpy as np 

# import autograd.numpy as np   # Thinly-wrapped version of Numpy
# from autograd import jacobian

# Create point mass
cartpole = CartPole(dt=5e-2, integrator='euler')
x0 = np.array([[0],[1],[.1],[-.2]])

# Create cost model
cost_weights = [1,1e-6,10]
N = 5
x_ref = np.array([[0],[np.pi],[0],[0]])
quad_cost = QuadCost(cartpole, x_ref, N, cost_weights)

# Create shooting problem
problem = DDPProblem(cartpole, quad_cost, x0)

# print(cartpole.numdiff(x0, np.array([0])))

# create DDP solver
ddp = DDPSolver(problem)

# Test a point mass rollout with zero input
us = ddp.us
xs = ddp.xs
X, U = cartpole.rollout(x0, us)
# cartpole.plot_traj(X,U)

# print(cartpole.calcDiff(xs,us))
# solve DDP 
xs, us = ddp.solve(1)

# Plot results  
ddp.plot()
