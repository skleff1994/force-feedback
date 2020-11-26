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
    IAM compatible with Crocoddyl python interface
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