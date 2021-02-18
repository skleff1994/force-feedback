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

# Action model for the point mass 
class ActionModelPointMass(crocoddyl.ActionModelAbstract):
    '''
    IAM for point mass using Euler, RK4 or exact integration
    Cost is hard-coded in this class
    '''
    def __init__(self, dt=0.01, integrator='euler'):
        # Initialize abstract model
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(2), 1, 5) 
        self.nx = 2
        # Must be defined for Croco
        self.unone = np.zeros(self.nx)
        self.xnone = np.zeros(self.nu)
        # dt 
        self.dt = dt
        self.integrator = integrator
        # Cost ref 
        self.x_tar = np.zeros(self.nx)
        self.x_ref = np.zeros(self.nx)
        self.u_ref = 0.
        # Cost weights
        self.w_x = 0.
        self.w_xreg = 0. 
        self.w_ureg = 0.
        # CT dynamics
        self.Ac = np.array([[0,1],[0,0]])
        self.Bc = np.array([[0],[1]])
        # DT model
        self.Ad = np.eye(self.nx) + self.dt*self.Ac
        self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)

    def f(self, x, u):
        '''
        CT dynamics
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, data, x, u):
        '''
        Discretized dynamics (Euler) + cost residuals
        '''
        # Integrate next state
            # Euler step
        if(self.integrator=='euler'):
            xnext = x + self.f(x,u)*self.dt
            # RK4 step 
        if(self.integrator=='rk4'):
            k1 = self.f(x, u) * self.dt
            k2 = self.f(x + k1 / 2.0, u) * self.dt
            k3 = self.f(x + k2 / 2.0, u) * self.dt
            k4 = self.f(x + k3, u) * self.dt
            xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
            # Exact (default)
        else:
            xnext = self.Ad.dot(x) + self.Bd.dot(u)
        data.xnext = xnext #x + self.f(x,u)*self.dt

        # Euler integration
        # data.xnext = x + self.f(x,u)*self.dt

        data.r[:self.nx] = self.w_x * ( x - self.x_tar ) 
        data.r[self.nx:2*self.nx] = self.w_xreg * ( x - self.x_ref )
        data.r[:-1] = self.w_ureg * ( u - self.u_ref )
        # Cost value
        data.cost = .5 * sum(data.r**2)

    def calcDiff(self, data, x, u):
        ''' 
        Partial derivatives of dynamics and cost (for crocoddyl)
        '''
        data.Fx = np.eye(self.nx) + self.dt*self.Ac
        data.Fu = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)
        data.Lx = ( x - self.x_tar ) * ( [self.w_x**2] * self.nx ) + ( x - self.x_ref ) * ( [self.w_xreg**2] * self.nx ) 
        data.Lu = ( u - self.u_ref ) * ( [self.w_ureg**2] * self.nx)
        data.Lxx = self.w_x**2 * np.eye(self.nx)
        data.Luu = np.array([self.w_ureg**2])


# Action model for the "augmented state" point mass (spring-damper)
class ActionModelPointMassContact(crocoddyl.ActionModelAbstract):
    '''
    IAM for point mass using Euler, rk4 or exactS integration
    Cost is hard-coded in this class
    '''
    def __init__(self, dt=0.01, K=0., B=0., p0=0., integrator='euler'):
        # Initialize abstract model
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 1, 7) 
        self.nx = 3
        # Must be defined for Croco
        self.unone = np.zeros(self.nx)
        self.xnone = np.zeros(self.nu)
        # dt (Euler)
        self.dt = dt
        self.integrator = integrator
        # Stiffness, damping and anchor point 
        self.K = K
        self.B = B
        self.p0 = p0
        # Reference state for state reg (origin)
        # Cost ref 
        self.x_tar = np.zeros(self.nx)
        self.x_ref = np.zeros(self.nx)
        self.u_ref = 0.
        # Cost weights
        self.w_x = 0.
        self.w_xreg = 0. 
        self.w_ureg = 0.
        # CT dynamics
        self.Ac = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [0, -self.K, -self.B]])
        self.Bc = np.array([[0],
                            [1],
                            [-self.B]])
        # DT model
        self.Ad = np.eye(self.nx) + self.dt*self.Ac
        self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)

    def f(self, x, u):
        '''
        CT dynamics
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, data, x, u):
        '''
        Discretized dynamics (Euler) + cost residuals
        '''
        # Integrate next state
            # Euler step
        if(self.integrator=='euler'):
            xnext = x + self.f(x,u)*self.dt
            # RK4 step 
        if(self.integrator=='rk4'):
            k1 = self.f(x, u) * self.dt
            k2 = self.f(x + k1 / 2.0, u) * self.dt
            k3 = self.f(x + k2 / 2.0, u) * self.dt
            k4 = self.f(x + k3, u) * self.dt
            xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
            # Exact (default)
        else:
            xnext = self.Ad.dot(x) + self.Bd.dot(u)
        data.xnext = xnext #x + self.f(x,u)*self.dt
        data.r[:self.nx] = self.w_x * ( x - self.x_tar ) 
        data.r[self.nx:2*self.nx] = self.w_xreg * ( x - self.x_ref )
        data.r[:-1] = self.w_ureg * ( u - self.u_ref )
        # Cost value
        data.cost = .5 * sum(data.r**2)

    def calcDiff(self, data, x, u):
        ''' 
        Partial derivatives of dynamics and cost (for crocoddyl)
        '''
        data.Fx = np.eye(self.nx) + self.dt*self.Ac
        data.Fu = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)
        data.Lx = ( x - self.x_tar ) * ( [self.w_x**2] * self.nx ) + ( x - self.x_ref ) * ( [self.w_xreg**2] * self.nx ) 
        data.Lu = ( u - self.u_ref ) * ( [self.w_ureg**2] * self.nx)
        data.Lxx = self.w_x**2 * np.eye(self.nx)
        data.Luu = np.array([self.w_ureg**2])


# Action model for the "observer" point mass (spring-damper) 
class ActionModelPointMassObserver(crocoddyl.ActionModelAbstract):
    '''
    IAM for point mass using Euler integration
    Cost is hard-coded in this class
    '''
    def __init__(self, dt=0.01, K=0., B=0., integrator='euler'):
        # Initialize abstract model
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(2), 1, 5) 
        self.nx = 2
        # Must be defined for Croco
        self.unone = np.zeros(self.nx)
        self.xnone = np.zeros(self.nu)
        # dt 
        self.dt = dt
        self.integrator = integrator
        # Cost ref 
        self.x_tar = np.zeros(self.nx)
        self.x_ref = np.zeros(self.nx)
        self.u_ref = 0.
        # Cost weights
        self.w_x = 0.
        self.w_xreg = 0. 
        self.w_ureg = 0.
        # CT dynamics
        self.Ac = np.array([[0,1],[0,0]])
        self.Bc = np.array([[0],[1]])
        # Measurement model
        self.K = K
        self.B = B
        self.Hc = np.array([[1, 0],
                            [-K, -B]])
        # DT model
        self.Ad = np.eye(self.nx) + self.dt*self.Ac
        self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)
        self.Hd = self.Hc
        
    def f(self, x, u):
        '''
        CT dynamics
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, data, x, u):
        '''
        Discretized dynamics (Euler) + cost residuals
        '''
        # Integrate next state
            # Euler step
        if(self.integrator=='euler'):
            xnext = x + self.f(x,u)*self.dt
            # RK4 step 
        if(self.integrator=='rk4'):
            k1 = self.f(x, u) * self.dt
            k2 = self.f(x + k1 / 2.0, u) * self.dt
            k3 = self.f(x + k2 / 2.0, u) * self.dt
            k4 = self.f(x + k3, u) * self.dt
            xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
            # Exact (default)
        else:
            xnext = self.Ad.dot(x) + self.Bd.dot(u)
        data.xnext = xnext #x + self.f(x,u)*self.dt

        data.r[:self.nx] = self.w_x * ( x - self.x_tar ) 
        data.r[self.nx:2*self.nx] = self.w_xreg * ( x - self.x_ref )
        data.r[:-1] = self.w_ureg * ( u - self.u_ref )
        # Cost value
        data.cost = .5 * sum(data.r**2)

    def calcDiff(self, data, x, u):
        ''' 
        Partial derivatives of dynamics and cost (for crocoddyl)
        '''
        data.Fx = np.eye(self.nx) + self.dt*self.Ac
        data.Fu = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)
        data.Lx = ( x - self.x_tar ) * ( [self.w_x**2] * self.nx ) + ( x - self.x_ref ) * ( [self.w_xreg**2] * self.nx ) 
        data.Lu = ( u - self.u_ref ) * ( [self.w_ureg**2] * self.nx)
        data.Lxx = self.w_x**2 * np.eye(self.nx)
        data.Luu = np.array([self.w_ureg**2])


# Action model for the "actuation dynamics" point mass (low-pass filter)
class ActionModelPointMassActuation(crocoddyl.ActionModelAbstract):
    '''
    IAM for point mass using Euler, rk4 or exacts integration
    Cost is hard-coded in this class
    '''
    def __init__(self, dt=0.01, k=0., integrator='euler'):
        # Initialize abstract model
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 1, 7) 
        self.nx = 3
        # Must be defined for Croco
        self.unone = np.zeros(self.nx)
        self.xnone = np.zeros(self.nu)
        # dt (Euler)
        self.dt = dt
        self.integrator = integrator
        # Stiffness, damping and anchor point 
        self.K = K
        self.B = B
        self.p0 = p0
        # Reference state for state reg (origin)
        # Cost ref 
        self.x_tar = np.zeros(self.nx)
        self.x_ref = np.zeros(self.nx)
        self.u_ref = 0.
        # Cost weights
        self.w_x = 0.
        self.w_xreg = 0. 
        self.w_ureg = 0.
        # coef. of 1st order actuation dynamics
        self.k = k 
        self.Ac = np.array([[0, 1, 0],
                            [0, 0, 1/self.m],
                            [0, 0, -self.k]])
        self.Bc = np.array([[0],
                            [0],
                            [self.k]])
        # DT model
        self.Ad = np.eye(self.nx) + self.dt*self.Ac
        self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)

    def f(self, x, u):
        '''
        CT dynamics
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, data, x, u):
        '''
        Discretized dynamics (Euler) + cost residuals
        '''
        # Integrate next state
            # Euler step
        if(self.integrator=='euler'):
            xnext = x + self.f(x,u)*self.dt
            # RK4 step 
        if(self.integrator=='rk4'):
            k1 = self.f(x, u) * self.dt
            k2 = self.f(x + k1 / 2.0, u) * self.dt
            k3 = self.f(x + k2 / 2.0, u) * self.dt
            k4 = self.f(x + k3, u) * self.dt
            xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
            # Exact (default)
        else:
            xnext = self.Ad.dot(x) + self.Bd.dot(u)
        data.xnext = xnext #x + self.f(x,u)*self.dt
        data.r[:self.nx] = self.w_x * ( x - self.x_tar ) 
        data.r[self.nx:2*self.nx] = self.w_xreg * ( x - self.x_ref )
        data.r[:-1] = self.w_ureg * ( u - self.u_ref )
        # Cost value
        data.cost = .5 * sum(data.r**2)

    def calcDiff(self, data, x, u):
        ''' 
        Partial derivatives of dynamics and cost (for crocoddyl)
        '''
        data.Fx = np.eye(self.nx) + self.dt*self.Ac
        data.Fu = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)
        data.Lx = ( x - self.x_tar ) * ( [self.w_x**2] * self.nx ) + ( x - self.x_ref ) * ( [self.w_xreg**2] * self.nx ) 
        data.Lu = ( u - self.u_ref ) * ( [self.w_ureg**2] * self.nx)
        data.Lxx = self.w_x**2 * np.eye(self.nx)
        data.Luu = np.array([self.w_ureg**2])


# Could be replace by simple IAMEuler?
class ActionModel(crocoddyl.ActionModelAbstract):
    '''
    IAM compatible with Crocoddyl python interface
    dyn_model  : CT model + discretization
    cost_model : cost model
    '''
    def __init__(self, dyn_model, cost_model):
        # Initialize abstract model
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(dyn_model.nx), dyn_model.nu) 
        # Define dynamics model and cost function
        self.dyn_model = dyn_model
        self.cost_model = cost_model
        # Must be defined for Croco
        self.unone = np.zeros(self.dyn_model.nu)
        self.xnone = np.zeros(self.dyn_model.nx)
        # dt (Euler)
        self.dt = self.dyn_model.dt 

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
            data.Fx = self.dt * f_x.copy()
            data.Fu = self.dt * f_u.copy()
            data.Lx = self.dt * l_x.copy()
            data.Lx = self.dt * l_u.copy()
            data.Lxx = self.dt * l_xx.copy()
            data.Luu = self.dt * l_uu.copy()
            data.Lux = self.dt * l_ux.copy()

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
        Initialized from DAM
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

            # Why hard-code this part ? Not necessary for the generic IAM
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
        '''
        Euler integration (or no integration depending on dt)
        '''
        # what if w is none?
        x = y[:self.differential.state.nx]
        # filtering the torque with the previous state : get tau_q+ from w 
        data.tau_plus[:] = self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w
        # dynamics : get a_q = DAM(q, vq, tau_q+)
        self.differential.calc(data.differential, x, data.tau_plus)
        if self.withCostResiduals:
            data.r = data.differential.r
        # Euler integration step of dt : get v_q+, q+
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
        '''
        Compute derivatives 
        '''
        # First call calc
        self.calc(data, y, w)
        x = y[:-self.differential.nu]
        # Get derivatives of DAM under LP-Filtered input 
        self.differential.calcDiff(data.differential, x, data.tau_plus)
        # Get d(IAM)/dx =  [d(q+)/dx, d(v_q+)/dx] 
        dxnext_dx, dxnext_ddx = self.differential.state.Jintegrate(x, data.dx)
        # Get d(DAM)/dx , d(DAM)/du (why resize?)
        da_dx, da_du = data.differential.Fx, np.resize(data.differential.Fu, (self.differential.state.nv, self.differential.nu))
        ddx_dx = np.vstack([da_dx * self.dt, da_dx])
        # ??? ugly way of coding identity matrix ?
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


# # From Gabriele
# class IntegratedActionModelLPF(crocoddyl.ActionModelAbstract):
#     '''
#         Add a low pass effect on the torque dynamics
#             tau+ = alpha * tau + (1 - alpha) * w
#         where alpha is a parameter depending of the memory of the system
#         tau is the filtered torque included in the state and w the unfiltered control
#         The state is augmented so that it includes the filtered torque
#             y = [x, tau].T
#     '''
#     def __init__(self, diffModel, dt=1e-3, withCostResiduals=True, f_c = np.NaN):
#             '''
#                 If f_c is undefined or NaN, it is assumed to be infinite, unfiltered case
#             '''
#             crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(diffModel.state.nx + diffModel.nu), diffModel.nu)
#             self.differential = diffModel
#             self.dt = dt
#             self.withCostResiduals = withCostResiduals
#             self.set_alpha(f_c)
#             self.nx = diffModel.state.nx
#             # self.nw = diffModel.nu # augmented control dimension
#             self.ny = self.nu + self.nx
#             if self.dt == 0:
#                 self.enable_integration_ = False
#             else:
#                 self.enable_integration_ = True

#             # weight of the unfiltered torque cost
#             self.w_bound = 1e3
#             self.w_reg = 1e-2
#             w_lb = - diffModel.state.pinocchio.effortLimit
#             w_ub = diffModel.state.pinocchio.effortLimit
#             bounds = crocoddyl.ActivationBounds(w_lb, w_ub)
#             self.activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

#     def createData(self):
#         '''
#             The data is created with a custom data class that contains the filtered torque tau_plus and the activation
#         '''
#         data = IntegratedActionDataLPF(self)
#         return data

#     def set_alpha(self, f_c = None):
#         '''
#             Sets the parameter alpha according to the cut-off frequency f_c
#             alpha = 1 / (1 + 2pi dt f_c)
#         '''
#         if f_c > 0:
#             omega = 1/(2 * np.pi * self.dt * f_c)
#             self.alpha = omega/(omega + 1)
#         else:
#             self.alpha = 0

#     def calc(self, data, y, w = None):
#         x = y[:self.differential.state.nx]
#         # filtering the torque with the previous state
#         data.tau_plus[:] = self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w
#         # dynamics
#         self.differential.calc(data.differential, x, data.tau_plus)
#         if self.withCostResiduals:
#             data.r = data.differential.r
#         if self.enable_integration_:
#             data.cost = self.dt * data.differential.cost
#             # adding the cost on the unfiltered torque
#             self.activation.calc(data.activation, w)
#             data.cost += self.dt * self.w_bound * data.activation.a_value + self.dt * w @ w / 2 * self.w_reg
#             data.dx = np.concatenate([x[self.differential.state.nq:] * self.dt + data.differential.xout * self.dt**2, data.differential.xout * self.dt])
#             data.xnext[:self.nx] = self.differential.state.integrate(x, data.dx)
#             data.xnext[self.nx:] = data.tau_plus
#         else:
#             data.dx = np.zeros(len(y))
#             data.xnext[:] = y
#             data.cost = data.differential.cost
#             # adding the cost on the unfiltered torque
#             self.activation.calc(data.activation, w)
#             data.cost += self.w_bound * data.activation.a_value + w @ w / 2 * self.w_reg

#         return data.xnext, data.cost

#     def calcDiff(self, data, y, w=None):
#         self.calc(data, y, w)

#         x = y[:-self.differential.nu]
#         # data.tau_plus[:] = np.array([self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w])
#         self.differential.calcDiff(data.differential, x, data.tau_plus)
#         dxnext_dx, dxnext_ddx = self.differential.state.Jintegrate(x, data.dx)
#         da_dx, da_du = data.differential.Fx, np.resize(data.differential.Fu, (self.differential.state.nv, self.differential.nu))
#         ddx_dx = np.vstack([da_dx * self.dt, da_dx])
#         ddx_dx[range(self.differential.state.nv), range(self.differential.state.nv, 2 * self.differential.state.nv)] += 1
#         ddx_du = np.vstack([da_du * self.dt, da_du])

#         # In this scope the data.* are in the augmented state coordinates
#         # while all the differential dd are in the canonical x coordinates
#         # we must set correctly the quantities where needed
#         Fx = dxnext_dx + self.dt * np.dot(dxnext_ddx, ddx_dx)
#         Fu = self.dt * np.dot(dxnext_ddx, ddx_du) # wrong according to NUM DIFF, no timestep

#         # TODO why is this not multiplied by timestep?
#         data.Fx[:self.nx, :self.nx] = Fx
#         data.Fx[:self.nx, self.nx:self.ny] = self.alpha * Fu
#         data.Fx[self.nx:, self.nx:] = self.alpha * np.eye(self.nu)
#         # print('Fy : ', data.Fx)
#         # TODO CHECKING WITH NUMDIFF, NO TIMESTEP HERE
#         if self.nu == 1:
#             data.Fu.flat[:self.nx] = (1 - self.alpha) * Fu
#             data.Fu.flat[self.nx:] = (1 - self.alpha) * np.eye(self.nu)
#         else:
#             data.Fu[:self.nx, :self.nu] = (1 - self.alpha) * Fu
#             data.Fu[self.nx:, :self.nu] = (1 - self.alpha) * np.eye(self.nu)

#         if self.enable_integration_:

#             data.Lx[:self.nx] = self.dt * data.differential.Lx
#             data.Lx[self.nx:] = self.dt * self.alpha * data.differential.Lu

#             data.Lu[:] = self.dt * (1 - self.alpha) * data.differential.Lu

#             data.Lxx[:self.nx,:self.nx] = self.dt * data.differential.Lxx
#             # TODO reshape is not the best, see better how to cast this
#             data.Lxx[:self.nx,self.nx:] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
#             data.Lxx[self.nx:,:self.nx] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
#             data.Lxx[self.nx:,self.nx:] = self.dt * self.alpha**2 * data.differential.Luu

#             data.Lxu[:self.nx] = self.dt * (1 - self.alpha) * data.differential.Lxu
#             data.Lxu[self.nx:] = self.dt * (1 - self.alpha) * self.alpha * data.differential.Luu

#             data.Luu[:, :] = self.dt * (1 - self.alpha)**2 * data.differential.Luu

#             # adding the unfiltered torque cost
#             self.activation.calcDiff(data.activation, w)
#             data.Lu[:] += self.dt * self.w_bound * data.activation.Ar + w * self.dt * self.w_reg
#             data.Luu[:, :] += self.dt * self.w_bound * data.activation.Arr + np.diag(np.ones(self.nu)) * self.dt * self.w_reg

#         else:

#             data.Lx[:self.nx] = data.differential.Lx
#             data.Lx[self.nx:] = self.alpha * data.differential.Lu

#             data.Lu[:] = (1 - self.alpha) * data.differential.Lu

#             data.Lxx[:self.nx,:self.nx] = data.differential.Lxx
#             data.Lxx[:self.nx,self.nx:] = self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
#             data.Lxx[self.nx:,:self.nx] = self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
#             data.Lxx[self.nx:,self.nx:] = self.alpha**2 * data.differential.Luu

#             data.Lxu[:self.nx] = (1 - self.alpha) * data.differential.Lxu
#             data.Lxu[self.nx:] = (1 - self.alpha) * self.alpha * data.differential.Luu

#             data.Luu[:, :] = (1 - self.alpha)**2 * data.differential.Luu

#             # adding the unfiltered torque cost
#             self.activation.calcDiff(data.activation, w)
#             data.Lu[:] += self.w_bound * data.activation.Ar + w * self.w_reg
#             data.Luu[:, :] += self.w_bound * data.activation.Arr + np.diag(np.ones(self.nu)) * self.w_reg


class DAMPointMass(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(2), 1, 6)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        self.m1 = 1.
        self.m2 = .1
        self.l  = .5
        self.g  = 9.81
        self.costWeights = [1., 1., 0.1, 0.001, 0.001, 1.]  # sin, 1-cos, x, xdot, thdot, f
        
    def calc(self, data, x, u=None):
        if u is None: 
            u = model.unone
        # Getting the state and control variables
        y, th, ydot, thdot = x[0].item(), x[1].item(), x[2].item(), x[3].item()
        f = u[0].item()

        # Shortname for system parameters
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)

        ###########################################################################
        ############ TODO: Write the dynamics equation of your system #############
        ###########################################################################
        # Hint:
        # You don't need to implement integration rules for your dynamic system.
        # Remember that DAM implemented action models in continuous-time.
        m = m1 + m2
        mu = m1 + m2 * s ** 2
        xddot, thddot = cartpole_dynamics(self, data, x, u)  # Write the cartpole dynamics here
        data.xout = np.matrix([ xddot, thddot ]).T
        
        # Computing the cost residual and value : using cost model?
        data.r = np.matrix(self.costWeights * np.array([ s, 1 - c, y, ydot, thdot, f ])).T
        data.cost = .5 * sum(np.asarray(data.r) ** 2).item()

    def calcDiff(model,data,x,u=None):
        # Advance user might implement the derivatives in cartpole_analytical_derivatives
        pass
        # cartpole_analytical_derivatives(model, data, x, u)