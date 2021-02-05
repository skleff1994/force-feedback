# Title : croco_IAMs.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Collection of Integrated Action Models (IAM) following Crocoddyl template
#   i.e. contains both the dynamics and the cost model 
#        and IAM is separated from data
# Directly usable with python interface of Crocoddyl 

import crocoddyl
crocoddyl.switchToNumpyArray()
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
            data.xnext = xnext.copy()
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
        if(data is None):
            return f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux
        else:
            data.Fx = f_x.copy()
            data.Fu = f_u.copy()
            data.Lx = l_x.copy()
            data.Lx = l_u.copy()
            data.Lxx = l_xx.copy()
            data.Luu = l_uu.copy()
            data.Lux = l_ux.copy()

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

# From Gabriele
class IntegratedActionModelLPF(crocoddyl.ActionModelAbstract):
    '''
        Add a low pass effect on the torque dynamics
            tau+ = alpha * tau + (1 - alpha) * w
        where alpha is a parameter depending of the memory of the system
        tau is the filtered torque included in the state and w the unfiltered control
        The state is augmented so that it includes the filtered torque
            y = [x, tau].T
    '''
    def __init__(self, diffModel, dt=1e-3, withCostResiduals=True, f_c = np.NaN):
            '''
                If f_c is undefined or NaN, it is assumed to be infinite, unfiltered case
            '''
            crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(diffModel.state.nx + diffModel.nu), diffModel.nu)
            self.differential = diffModel
            self.dt = dt
            self.withCostResiduals = withCostResiduals
            self.set_alpha(f_c)
            self.nx = diffModel.state.nx
            # self.nw = diffModel.nu # augmented control dimension
            self.ny = self.nu + self.nx
            if self.dt == 0:
                self.enable_integration_ = False
            else:
                self.enable_integration_ = True

            # weight of the unfiltered torque cost
            self.w_bound = 1e3
            self.w_reg = 1e-2
            w_lb = - diffModel.state.pinocchio.effortLimit
            w_ub = diffModel.state.pinocchio.effortLimit
            bounds = crocoddyl.ActivationBounds(w_lb, w_ub)
            self.activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

    def createData(self):
        '''
            The data is created with a custom data class that contains the filtered torque tau_plus and the activation
        '''
        data = IntegratedActionDataLPF(self)
        return data

    def set_alpha(self, f_c = None):
        '''
            Sets the parameter alpha according to the cut-off frequency f_c
            alpha = 1 / (1 + 2pi dt f_c)
        '''
        if f_c > 0:
            omega = 1/(2 * np.pi * self.dt * f_c)
            self.alpha = omega/(omega + 1)
        else:
            self.alpha = 0

    def calc(self, data, y, w = None):
        x = y[:self.differential.state.nx]
        # filtering the torque with the previous state
        data.tau_plus[:] = self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w
        # dynamics
        self.differential.calc(data.differential, x, data.tau_plus)
        if self.withCostResiduals:
            data.r = data.differential.r
        if self.enable_integration_:
            data.cost = self.dt * data.differential.cost
            # adding the cost on the unfiltered torque
            self.activation.calc(data.activation, w)
            data.cost += self.dt * self.w_bound * data.activation.a_value + self.dt * w @ w / 2 * self.w_reg
            data.dx = np.concatenate([x[self.differential.state.nq:] * self.dt + data.differential.xout * self.dt**2, data.differential.xout * self.dt])
            data.xnext[:self.nx] = self.differential.state.integrate(x, data.dx)
            data.xnext[self.nx:] = data.tau_plus
        else:
            data.dx = np.zeros(len(y))
            data.xnext[:] = y
            data.cost = data.differential.cost
            # adding the cost on the unfiltered torque
            self.activation.calc(data.activation, w)
            data.cost += self.w_bound * data.activation.a_value + w @ w / 2 * self.w_reg

        return data.xnext, data.cost

    def calcDiff(self, data, y, w=None):
        self.calc(data, y, w)

        x = y[:-self.differential.nu]
        # data.tau_plus[:] = np.array([self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w])
        self.differential.calcDiff(data.differential, x, data.tau_plus)
        dxnext_dx, dxnext_ddx = self.differential.state.Jintegrate(x, data.dx)
        da_dx, da_du = data.differential.Fx, np.resize(data.differential.Fu, (self.differential.state.nv, self.differential.nu))
        ddx_dx = np.vstack([da_dx * self.dt, da_dx])
        ddx_dx[range(self.differential.state.nv), range(self.differential.state.nv, 2 * self.differential.state.nv)] += 1
        ddx_du = np.vstack([da_du * self.dt, da_du])

        # In this scope the data.* are in the augmented state coordinates
        # while all the differential dd are in the canonical x coordinates
        # we must set correctly the quantities where needed
        Fx = dxnext_dx + self.dt * np.dot(dxnext_ddx, ddx_dx)
        Fu = self.dt * np.dot(dxnext_ddx, ddx_du) # wrong according to NUM DIFF, no timestep

        # TODO why is this not multiplied by timestep?
        data.Fx[:self.nx, :self.nx] = Fx
        data.Fx[:self.nx, self.nx:self.ny] = self.alpha * Fu
        data.Fx[self.nx:, self.nx:] = self.alpha * np.eye(self.nu)
        # print('Fy : ', data.Fx)
        # TODO CHECKING WITH NUMDIFF, NO TIMESTEP HERE
        if self.nu == 1:
            data.Fu.flat[:self.nx] = (1 - self.alpha) * Fu
            data.Fu.flat[self.nx:] = (1 - self.alpha) * np.eye(self.nu)
        else:
            data.Fu[:self.nx, :self.nu] = (1 - self.alpha) * Fu
            data.Fu[self.nx:, :self.nu] = (1 - self.alpha) * np.eye(self.nu)

        if self.enable_integration_:

            data.Lx[:self.nx] = self.dt * data.differential.Lx
            data.Lx[self.nx:] = self.dt * self.alpha * data.differential.Lu

            data.Lu[:] = self.dt * (1 - self.alpha) * data.differential.Lu

            data.Lxx[:self.nx,:self.nx] = self.dt * data.differential.Lxx
            # TODO reshape is not the best, see better how to cast this
            data.Lxx[:self.nx,self.nx:] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.dt * self.alpha**2 * data.differential.Luu

            data.Lxu[:self.nx] = self.dt * (1 - self.alpha) * data.differential.Lxu
            data.Lxu[self.nx:] = self.dt * (1 - self.alpha) * self.alpha * data.differential.Luu

            data.Luu[:, :] = self.dt * (1 - self.alpha)**2 * data.differential.Luu

            # adding the unfiltered torque cost
            self.activation.calcDiff(data.activation, w)
            data.Lu[:] += self.dt * self.w_bound * data.activation.Ar + w * self.dt * self.w_reg
            data.Luu[:, :] += self.dt * self.w_bound * data.activation.Arr + np.diag(np.ones(self.nu)) * self.dt * self.w_reg

        else:

            data.Lx[:self.nx] = data.differential.Lx
            data.Lx[self.nx:] = self.alpha * data.differential.Lu

            data.Lu[:] = (1 - self.alpha) * data.differential.Lu

            data.Lxx[:self.nx,:self.nx] = data.differential.Lxx
            data.Lxx[:self.nx,self.nx:] = self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.alpha**2 * data.differential.Luu

            data.Lxu[:self.nx] = (1 - self.alpha) * data.differential.Lxu
            data.Lxu[self.nx:] = (1 - self.alpha) * self.alpha * data.differential.Luu

            data.Luu[:, :] = (1 - self.alpha)**2 * data.differential.Luu

            # adding the unfiltered torque cost
            self.activation.calcDiff(data.activation, w)
            data.Lu[:] += self.w_bound * data.activation.Ar + w * self.w_reg
            data.Luu[:, :] += self.w_bound * data.activation.Arr + np.diag(np.ones(self.nu)) * self.w_reg

class IntegratedActionDataLPF(crocoddyl.ActionDataAbstract):
    '''
    Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
    '''
    def __init__(self, am):
        crocoddyl.ActionDataAbstract.__init__(self, am)
        self.differential = am.differential.createData()
        self.activation = am.activation.createData()
        self.tau_plus = np.zeros(am.nu)
        self.Fx = np.zeros((am.ny, am.ny))
        self.Fu = np.zeros((am.ny, am.nu))
        self.Lx = np.zeros(am.ny)
        self.Lu = np.zeros(am.nu)
        self.Lxx = np.zeros((am.ny, am.ny))
        self.Lxu = np.zeros((am.ny, am.nu))
        self.Luu = np.zeros((am.nu,am.nu))