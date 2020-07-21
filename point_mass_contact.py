import crocoddyl
import pinocchio

import numpy as np

from utils import animatePointMass, plotPointMass

class ActionModelPointMassContact(crocoddyl.ActionModelAbstract):
    '''
    Discretized point mass model environment class 
    '''

    def __init__(self, dt=1e-3, integrator='euler'):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 1, 4) # nu = 1, nr = 3
        self.nx = 3
        self.unone = np.zeros(self.nu)
        self.dt = dt                        # integration step 
        self.integrator = integrator
        self.w_x, self.w_u = 10., 0.         # cost x, u
        self.m, self.K, self.B = 1., 20., 1.  # mass, stiffness, damping
        # CT dynamics
        self.Ac = np.array([[0, 1, 0],
                            [0, 0, 1/self.m],
                            [0, -self.K, -self.B/self.m]])
        self.Bc = np.array([[0],
                            [1/self.m],
                            [-self.B/self.m]])
        # DT model
        self.Ad = np.eye(self.nx) + self.dt*self.Ac
        self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)


    def f(self, x, u):
        '''
        CT dynamics 
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, data, x, u=None):
        '''
        Discretized dynamics + cost residuals
        '''
        if u is None: 
            u=self.unone

        if(self.integrator=='rk4'): # rk4 step
            k1 = self.f(x, u) * self.dt
            k2 = self.f(x + k1 / 2.0, u) * self.dt
            k3 = self.f(x + k2 / 2.0, u) * self.dt
            k4 = self.f(x + k3, u) * self.dt
            data.xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
        else: # default Euler
            data.xnext = x + self.f(x,u)*self.dt

        # Computing cost residual and value 
        data.r[:self.nx], data.r[:self.nu] = self.w_x*x, self.w_u*u
        data.cost = .5* sum(data.r**2)

    def calcDiff(self, data, x, u=None):
        '''
        Partial derivatives of IAM
        '''
        if u is None: 
            u=self.unone
        
        data.Fx = self.Ad
        data.Fu = self.Bd
        data.Lx = x*([self.w_x**2] * self.nx)
        data.Lx = u*([self.w_u**2] * self.nu)
        data.Lxx[range(self.nx), range(self.nx)] = [self.w_x**2]*self.nx
        data.Luu[range(self.nu), range(self.nu)] = [self.w_u**2]*self.nu


# Create IAM (integrate DAM with Euler)
dt = 1e-2
running_IAM = ActionModelPointMassContact(dt, 'rk4')
terminal_IAM = ActionModelPointMassContact(dt, 'rk4')

# Initial conditions
p0 = 0. # reference contact point 
p = 1.
v = 0.
lmb = -running_IAM.K*(p - p0) - running_IAM.B*v
x = np.matrix([p, v, lmb]).T
u = np.matrix([0.])

# Define shooting problem
T = 1000
problem = crocoddyl.ShootingProblem(x, [running_IAM]*T, terminal_IAM)

# Integrate (rollout)
us = [ u ]*T
xs = problem.rollout(us)

# # Extract and plot trajectories
# 

# # Animate
# from IPython.display import HTML
# anim = animatePointMass(xs)
# HTML(anim.to_html5_video())

# Create the DDP solver and setup callbacks
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# Solve and retrieve X,U
done = ddp.solve([], [], 10)

plotPointMass(ddp.xs, ddp.us)
from IPython.display import HTML
anim = animatePointMass(ddp.xs)
# HTML(anim.to_html5_video())


# Display norm of partial derivatives ???




# class DifferentialActionModelPointMass(crocoddyl.DifferentialActionModelAbstract):
#     '''
#     Point mass model environment class
#     '''

#     def __init__(self):
#         crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(2), 1, 3) # nu = 1, nr = 3
#         self.unone = np.zeros(self.nu)
#         self.costWeights = [1, 1, 1]

#     def calc(self, data, x, u=None):
#         if u is None: u=self.unone

#         # State and control variables
#         p, v = np.asscalar(x[0]), np.asscalar(x[1])
#         f = np.asscalar(u[0])

#         # Compute next state
#         data.xout = np.matrix([f]).T

#         # Computing cost residual and value
#         data.r = np.matrix(self.costWeights * np.array([p, v, f])).T
#         data.cost = .5*np.asscalar(sum(np.asarray(data.r)**2))


# # Create DAM
# pointmassDAM = DifferentialActionModelPointMass()
# pointmassData = pointmassDAM.createData()
# # Create DAM + derivatives
# pointmassND = crocoddyl.DifferentialActionModelNumDiff(pointmassDAM, True)
# pointmassDataND = pointmassND.createData()
# # Create IAM (integrate DAM) + data
# dt = 1e-2
# pointmassIAM = crocoddyl.IntegratedActionModelEuler(pointmassND, dt)
# pointmassDataIAM = pointmassIAM.createData()

# # Create terminal model 
# terminalPointmassDAM = DifferentialActionModelPointMass()
# terminalPointmassND = crocoddyl.DifferentialActionModelNumDiff(terminalPointmassDAM, True)
# terminalPointmassIAM = crocoddyl.IntegratedActionModelEuler(terminalPointmassND, dt)
# terminalPointmassDAM.costWeights[0] = 0 # p
# terminalPointmassDAM.costWeights[1] = 0 # v
# terminalPointmassDAM.costWeights[2] = 0 # f

# # Define shooting problem 
# x0 = np.matrix([0, 1.]).T
# T = 100
# problem = crocoddyl.ShootingProblem(x0, [pointmassIAM]*T, terminalPointmassIAM)

# # Integrate (rollout)
# us = [ pinocchio.utils.zero(pointmassIAM.differential.nu) ]*T
# xs = problem.rollout(us)