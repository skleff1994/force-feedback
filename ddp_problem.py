# Author: Sebastien
# My python implementation of DDP solver

import numpy as np


class DDPProblem:
    '''
    DDP problem formulation
    '''
    
    def __init__(self, dynamics, cost, x0):
        # dynamics model (disccrete or discretized)
        self.dynamics = dynamics
        # cost model
        self.cost = cost
        # horizon and time step
        self.N = self.cost.N
        self.dt = self.dynamics.dt 
        # initial state 
        self.x0 = x0

    def calc(self, xs, us):
        '''
        Compute values of cost 
        '''
        return self.cost.calc(xs, us)

    def calcDiff(self, xs, us):
        '''
        Compute derivatives of cost and dynamics
        '''
        f_x, f_u = self.dynamics.calcDiff(xs, us)
        l_x, l_u, l_xx, l_uu, l_ux = self.cost.calcDiff(xs, us)
        return f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux 
