# Title : croco_IAMs.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Collection of Integrated Action Models (IAM) following Crocoddyl template
#   i.e. contains both the dynamics and the cost model 
#        and IAM is separated from data
# Directly usable with python interface of Crocoddyl 

import crocoddyl
import numpy as np


class ActionModelPM(crocoddyl.ActionModelAbstract):
    '''
    Discretized point mass model 
    '''
    def __init__(self, dyn_model, cost_model, dt=1e-3, integrator='euler'):
        # Initialize abstract model
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(dyn_model.nx), dyn_model.nu) 
        # Define dynamics model and cost function
        self.dyn_model = dyn_model
        self.cost_model = cost_model
        # Must be defined for Croco
        self.unone = np.zeros(self.dyn_model.nu)
        self.xnone = np.zeros(self.dyn_model.nx)

    def f(self, x, u):
        '''
        CT dynamics 
        '''
        return self.dyn_model.f(x,u)

    def calc(self, data=None, x=None, u=None):
        '''
        Discretized dynamics + cost residuals
        '''
        # If no args
        if(x is None):
            x = self.xnone
        if(u is None):
            u = self.unone
        # Integrate next state
        xnext = self.dyn_model.calc(x, u)   
        # Calculate cost value 
        value = self.cost_model.calc(x, u) 
        if(data is None):
            return xnext
        else:
            # Integrate next state
            data.xnext = xnext
            # Calculate cost value 
            data.cost = value

    def calcDiff(self, data=None, x=None, u=None):
        ''' 
        Partial derivatives of dynamics and cost (for crocoddyl)
        '''
        # If no args
        if(x is None):
            x = self.xnone
        if(u is None):
            u = self.unone
        # Calculate partials of dynamics and cost
        f_x, f_u = self.dyn_model.calcDiff(x, u)
        l_x, l_u, l_xx, l_uu, l_ux = self.cost_model.calcDiff(x, u)
        # Fill data
        data.Fx = f_x
        data.Fu = f_u
        data.Lx = l_x
        data.Lx = l_u
        data.Lxx = l_xx
        data.Luu = l_uu
        data.Lux = l_ux 

    def rollout(self, x0, us):
        '''
        Rollout from x0 using us 
        '''
        self.dyn_model.rollout(x0, us)

    def plot_traj(self, X, U):
        '''
        Plot trajectories X, U
        '''
        self.dyn_model.plot_traj(X, U)


class ActionModelPMContact(crocoddyl.ActionModelAbstract):
    '''
    Discrete point mass model with soft contact
    '''

    def __init__(self, dyn_model, cost_model, dt=1e-3, integrator='euler'):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(dyn_model.nx), dyn_model.nu)
        self.nx = 3
        self.unone = np.zeros(self.nu)
        self.xnone = np.zeros(self.nx)
        self.dt = dt                        # integration step 
        self.integrator = integrator
        self.w_x, self.w_u = 1., .1        # cost x, u
        self.m, self.K, self.B = 1., K, B   # mass, stiffness, damping
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

    def calc(self, data=None, x=None, u=None):
        '''
        Discretized dynamics + cost residuals
        '''
        if u is None: 
            u = self.unone
        if x is None:
            x = self.xnone

        # Runge-Kutta 4
        if(self.integrator=='rk4'):
            k1 = self.f(x, u) * self.dt
            k2 = self.f(x + k1 / 2.0, u) * self.dt
            k3 = self.f(x + k2 / 2.0, u) * self.dt
            k4 = self.f(x + k3, u) * self.dt
            xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
        # Euler (default)
        if(self.integrator=='euler'):
            xnext = x + self.f(x,u)*self.dt
        # Exact (default)
        else:
            xnext = self.Ad.dot(x) + self.Bd.dot(u)

        # If not used in croco solver
        if(data is None):
            return xnext
        else:
            data.xnext = xnext
            # Computing cost residual and value 
            # data.r[:self.nx], data.r[:self.nu] = self.w_x*x, self.w_u*u
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

    def rollout(self, x0, us):
        '''
        Rollout from x0 using us 
        '''
        N = len(us)
        X = np.zeros((N+1, self.nx))
        U = np.zeros((N, self.nu))
        X[0,:] = x0.T
        for i in range(N):
            U[i,:] = us[i].T
            X[i+1,:] = self.calc(x=np.array([X[i,:]]).T, u=us[i].T).T
        return X, U


# class ActionModelPointMassContact(crocoddyl.ActionModelAbstract):
#     '''
#     Discrete point mass model with soft contact
#     '''

#     def __init__(self, K=1., B=.1, dt=1e-3, integrator='euler'):
#         crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 1, 4) # nu = 1, nr = 3
#         self.nx = 3
#         self.unone = np.zeros(self.nu)
#         self.xnone = np.zeros(self.nx)
#         self.dt = dt                        # integration step 
#         self.integrator = integrator
#         self.w_x, self.w_u = 1., .1        # cost x, u
#         self.m, self.K, self.B = 1., K, B   # mass, stiffness, damping
#         # CT dynamics
#         self.Ac = np.array([[0, 1, 0],
#                             [0, 0, 1/self.m],
#                             [0, -self.K, -self.B/self.m]])
#         self.Bc = np.array([[0],
#                             [1/self.m],
#                             [-self.B/self.m]])
#         # DT model
#         self.Ad = np.eye(self.nx) + self.dt*self.Ac
#         self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)

#     def f(self, x, u):
#         '''
#         CT dynamics 
#         '''
#         return self.Ac.dot(x) + self.Bc.dot(u)

#     def calc(self, data=None, x=None, u=None):
#         '''
#         Discretized dynamics + cost residuals
#         '''
#         if u is None: 
#             u = self.unone
#         if x is None:
#             x = self.xnone

#         # Runge-Kutta 4
#         if(self.integrator=='rk4'):
#             k1 = self.f(x, u) * self.dt
#             k2 = self.f(x + k1 / 2.0, u) * self.dt
#             k3 = self.f(x + k2 / 2.0, u) * self.dt
#             k4 = self.f(x + k3, u) * self.dt
#             xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
#         # Euler (default)
#         if(self.integrator=='euler'):
#             xnext = x + self.f(x,u)*self.dt
#         # Exact (default)
#         else:
#             xnext = self.Ad.dot(x) + self.Bd.dot(u)

#         # If not used in croco solver
#         if(data is None):
#             return xnext
#         else:
#             data.xnext = xnext
#             # Computing cost residual and value 
#             # data.r[:self.nx], data.r[:self.nu] = self.w_x*x, self.w_u*u
#             data.cost = .5* sum(data.r**2)

#     def calcDiff(self, data, x, u=None):
#         '''
#         Partial derivatives of IAM
#         '''
#         if u is None: 
#             u=self.unone
        
#         data.Fx = self.Ad
#         data.Fu = self.Bd
#         data.Lx = x*([self.w_x**2] * self.nx)
#         data.Lx = u*([self.w_u**2] * self.nu)
#         data.Lxx[range(self.nx), range(self.nx)] = [self.w_x**2]*self.nx
#         data.Luu[range(self.nu), range(self.nu)] = [self.w_u**2]*self.nu

#     def rollout(self, x0, us):
#         '''
#         Rollout from x0 using us 
#         '''
#         N = len(us)
#         X = np.zeros((N+1, self.nx))
#         U = np.zeros((N, self.nu))
#         X[0,:] = x0.T
#         for i in range(N):
#             U[i,:] = us[i].T
#             X[i+1,:] = self.calc(x=np.array([X[i,:]]).T, u=us[i].T).T
#         return X, U