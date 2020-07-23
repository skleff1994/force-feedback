# Author: Sebastien
# NYU 2020
# Point mass model

import crocoddyl
import numpy as np
from utils import animatePointMass, plotPointMass

class ActionModelPointMass(crocoddyl.ActionModelAbstract):
    '''
    Discretized point mass model 
    '''

    def __init__(self, dt=1e-3, integrator='euler'):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(2), 1, 3) # nu = 1, nr = 3
        self.nx = 2
        self.unone = np.zeros(self.nu)
        self.dt = dt                        # integration step 
        self.integrator = integrator
        self.w_x, self.w_u = 1., 1.         # cost x, u
        self.m = 1.                         # mass, stiffness, damping

        # CT dynamics
        self.Ac = np.array([[0, 1],
                            [0, 0]])
        self.Bc = np.array([[0],
                            [1/self.m]])
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

        # Runge-Kutta 4
        if(self.integrator=='rk4'):
            k1 = self.f(x, u) * self.dt
            k2 = self.f(x + k1 / 2.0, u) * self.dt
            k3 = self.f(x + k2 / 2.0, u) * self.dt
            k4 = self.f(x + k3, u) * self.dt
            data.xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
        # Euler (default)
        else: 
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
