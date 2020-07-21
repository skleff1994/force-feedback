from ddp_problem import DDPProblem
from dynamics import PointMassContact
from cost_model import QuadCost
from ddp_solver import DDPSolver
import numpy as np 

# Create point mass
point_mass = PointMassContact(m=1, K=100., dt=1e-3)
x0 = np.array([[1.],[0.],[0.]])

# Create cost model
cost_weights = [10, 0, 100]
N = 1000
x_ref = np.array([[0.],[0.],[0.]])
quad_cost = QuadCost(point_mass, x_ref, N, cost_weights)

# Create shooting problem
problem = DDPProblem(point_mass, quad_cost, x0)

# create DDP solver
ddp = DDPSolver(problem)

# # Test a point mass rollout with zero input
# us = np.sin(np.linspace(0,2*np.pi/.5,N)) #ddp.us
# xs = ddp.xs
# X, U = point_mass.rollout(x0, us)
# point_mass.plot_traj(X,U)

# # print(point_mass.calcDiff(xs,us))
# solve DDP 
xs, us = ddp.solve(5)

# # Test gains
# k, K = ddp.k, ddp.K
# X = np.zeros((3,len(xs)))
# U = np.zeros(len(us))
# X[:,0] = xs[0].squeeze(axis=1)
# for i in range(len(us)):
#     U[i] = k[i] + K[i].dot(X[:,i])
#     X[:,i+1] = point_mass.calc(np.array([X[:,i]]).T, U[i]).squeeze(axis=1)
# point_mass.plot_traj(X.T,U)

# Plot result
ddp.plot()
