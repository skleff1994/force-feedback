# Title : cost_models.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Collection cost models compatible with custom DDP solver in ../core/ddp.py 
# Can be used to initialize custom IAMs compatible with Crocoddyl, see croco_IAMs.py

import numpy as np

class QuadTrackingCost:
    '''
    Quadratic cost term to track a reference state
      i.e. (x - x_ref)^T . Q . (x - x_ref)
    '''
    def __init__(self, model, x_ref, Q):
        # Discrete dynamics
        self.model = model
        # Reference state and cost weights matrix
        self.x_ref = x_ref
        self.Q = Q

    def calc(self, x, u):
        '''
        Calculate cost at (x,u)
        '''
        return .5*(x - self.x_ref).T.dot(self.Q).dot(x - self.x_ref)

    def calcDiff(self, x, u):
        '''
        Calculate partial derivatives of the cost at (x,u)
        '''
        l_x = self.Q.dot((x - self.x_ref))
        l_u = np.zeros(self.model.nu)
        l_xx = self.Q
        l_uu = np.zeros((self.model.nu, self.model.nu))
        l_ux = np.zeros((self.model.nu, self.model.nx))
        # print("cost_model.l_x = ", l_x)
        return l_x, l_u, l_xx, l_uu, l_ux 


class QuadCtrlRegCost:
    '''
    Quadratic regularization cost term on control
      i.e. u^T . R . u
    '''
    def __init__(self, model, R):
        # Discrete dynamics
        self.model = model
        # Cost weight matrix
        self.R = R

    def calc(self, x, u):
        '''
        Calculate cost at (x,u)
        '''
        return .5*u.T.dot(self.R).dot(u)

    def calcDiff(self, x, u):
        '''
        Calculate partial derivatives of the cost at (x,u)
        '''
        l_x = np.zeros(self.model.nx)
        l_u = self.R.dot(u)
        l_xx = np.zeros((self.model.nx, self.model.nx)) 
        l_uu = self.R
        l_ux = np.zeros((self.model.nu, self.model.nx))    
        return l_x, l_u, l_xx, l_uu, l_ux 


class CostSum:
    '''
    Sum of cost models, i.e. describes the cost function at one node (x,u) of the OCP 
    '''
    def __init__(self, model):
        # Cost models
        self.costs = []
        # Dynamics model
        self.model = model

    def add_cost(self, cost):
        '''
        Add cost term (i.e. one of the above classes) to the cost function 
        '''
        self.costs.append(cost)

    def calc(self, x, u=None):
        '''
        Evaluate the cost sum at node (x,u)
        '''
        value = 0.
        for cost in self.costs:
            value += cost.calc(x, u)
        return value 

    def calcDiff(self, x, u=None):
        '''
        Calculate partial derivatives of the cost sum at (x,u)
        '''
        l_x = np.zeros(self.model.nx)
        l_xx = np.zeros((self.model.nx, self.model.nx))
        l_u = np.zeros(self.model.nu)
        l_ux = np.zeros((self.model.nu, self.model.nx))
        l_uu = np.zeros((self.model.nu, self.model.nu))
        for cost in self.costs:
            c_x, c_u, c_xx, c_uu, c_ux = cost.calcDiff(x, u)
            # print("C_X = ",c_x)
            l_x += c_x 
            # print("L_X = ", l_x)
            l_u += c_u
            l_xx += c_xx
            l_uu += c_uu
            l_ux += c_ux
        # print("cost_model_sum.l_x = ", l_x)
        return l_x, l_u, l_xx, l_uu, l_ux