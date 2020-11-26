# Title : dyn_models.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Collection dynamics model classes compatible with custom DDP solver in ../core/ddp.py 
# Can also be used to initialize custom IAMs compatible with Crocoddyl , see croco_IAMs.py

import numpy as np
import matplotlib.pyplot as plt

class PointMass:
    '''
    Discretized dynamics model of the point mass (1D double integrator)
    Variables: 
      State   : x = position, velocity
      Control : u = input_force
      Output  : y = x = position, velocity 
    CT model:
      state transition : x'(t) = A x(t) + B u(t)
      output equation  : y(t)  = x(t)
    DT model:  
      state transition : x(n+1) = Ad x(n) + Bd u(n)
      output equation  : y(n)   = x(n) 
    '''
    def __init__(self, dt=0.01, integrator='exact'):
        # Dimensins
        self.nx = 2
        self.nu = 1
        # Default u 
        self.u_none = np.zeros(self.nu)
        # Sampling time
        self.dt = dt
        # CT model
        self.Ac = np.array([[0,1],[0,0]])
        self.Bc = np.array([[0],[1]])
        self.Hc = np.eye(2)
        # DT model
        self.Ad = np.eye(self.nx) + self.dt*self.Ac
        self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)
        self.Hd = self.Hc
        # Integration type
        self.integrator = integrator

    def f(self, x, u):
        '''
        CT dynamics [mandatory function]
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, x, u):
        '''
        DT dynamics [mandatory function]
        '''
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
        return xnext 
    
    def calcDiff(self, x, u):
        '''
        Get partial derivatives f_x, f_u at (x,u)
        '''
        return self.Ad, self.Bd

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
            X[i+1,:] = self.calc(X[i,:], U[i,:])
        return X, U

    def plot_traj(self, X, U):
        '''
        Plot trajectories X, U
        '''
        N = np.shape(U)[0]
        p = X[:,:self.nu]
        v = X[:,self.nu:]
        u = U
        # Create time spans for X and U
        tspan_x = np.linspace(0, N*self.dt, N+1)
        tspan_u = np.linspace(0, N*self.dt, N)
        # Create figs and subplots
        fig_x, ax_x = plt.subplots(1, 2)
        fig_u, ax_u = plt.subplots(1, 1)
        # Plot joints
        ax_x[0].plot(tspan_x, p, 'b-', label='pos')
        ax_x[0].set(xlabel='t (s)', ylabel='p (m)')
        ax_x[1].plot(tspan_x, v, 'b-', label='vel')
        ax_x[1].set(xlabel='t (s)', ylabel='v (m/s)')
        ax_u.plot(tspan_u, u, 'b-', label='acc') 
        ax_u.set(xlabel='t (s)', ylabel='u (N)')
        # Legend
        handles_x, labels_x = ax_x[0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
        handles_u, labels_u = ax_u.get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
        # Titles
        fig_x.suptitle('State trajectories', size=16)
        fig_u.suptitle('Control trajectory', size=16)
        plt.show()


class PointMassPartialObs:
    '''
    Discretized dynamics model of the point mass (1D double integrator)
    with visco-elastic force-position measurement model
    Variables: 
      State   : x = position, velocity
      Control : u = input_force
      Output  : y = contact_force, position
    CT model:
      state transition : x'(t) = A x(t) + B u(t)
      output equation  : y(t)  = H x(t)
    DT model:  
      state transition : x(n+1) = Ad x(n) + Bd u(n)
      output equation  : y(n)   = Hd x(n) 
    '''
    def __init__(self, dt=0.01, K=1, B=1., integrator='exact'):
        # Dimensins
        self.nx = 2
        self.nu = 1
        # Default u 
        self.u_none = np.zeros(self.nu)
        # Sampling time
        self.dt = dt
        # CT model
            # State transition
        self.Ac = np.array([[0,1],[0,0]])
        self.Bc = np.array([[0],[1]])
          # Measurement model
        self.K = K
        self.B = B
        self.Hc = np.array([[1, 0],
                            [-K, -B]])
        # DT model
        self.Ad = np.eye(self.nx) + self.dt*self.Ac
        self.Bd = self.dt*self.Bc + .5*self.dt**2*self.Ac.dot(self.Bc)
        self.Hd = self.Hc
        # Integration type
        self.integrator = integrator

    def f(self, x, u):
        '''
        CT dynamics [mandatory function]
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, x, u):
        '''
        DT dynamics [mandatory function]
        '''
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
        return xnext 
    
    def calcDiff(self, x, u):
        '''
        Get partial derivatives f_x, f_u at (x,u)
        '''
        return self.Ad, self.Bd

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
            X[i+1,:] = self.calc(X[i,:], U[i,:])
        return X, U

    def plot_traj(self, X, U):
        '''
        Plot trajectories X, U
        '''
        N = np.shape(U)[0]
        p = X[:,:self.nu]
        v = X[:,self.nu:]
        u = U
        # Create time spans for X and U
        tspan_x = np.linspace(0, N*self.dt, N+1)
        tspan_u = np.linspace(0, N*self.dt, N)
        # Create figs and subplots
        fig_x, ax_x = plt.subplots(1, 2)
        fig_u, ax_u = plt.subplots(1, 1)
        # Plot joints
        ax_x[0].plot(tspan_x, p, 'b-', label='pos')
        ax_x[0].set(xlabel='t (s)', ylabel='p (m)')
        ax_x[1].plot(tspan_x, v, 'b-', label='vel')
        ax_x[1].set(xlabel='t (s)', ylabel='v (m/s)')
        ax_u.plot(tspan_u, u, 'b-', label='acc') 
        ax_u.set(xlabel='t (s)', ylabel='u (N)')
        # Legend
        handles_x, labels_x = ax_x[0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
        handles_u, labels_u = ax_u.get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
        # Titles
        fig_x.suptitle('State trajectories', size=16)
        fig_u.suptitle('Control trajectory', size=16)
        plt.show()


class PointMassContact:
    '''
    Dynamics model of point mass in visco-elastic contact
    Variables: 
      State   : x = position, velocity, contact_force 
      Control : u = input_force
    CT model:
      state transition : x'(t) = A x(t) + B u(t)
    DT model:  
      state transition : x(n+1) = Ad x(n) + Bd u(n)
    '''
    def __init__(self, m=1, K=1, B=1., dt=0.01, integrator='euler'):
        # Dimensins
        self.nx = 3
        self.nu = 1
        # Sampling time
        self.dt = dt
        # Mass and stiffness
        self.m = m
        self.K = K
        self.B = B
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
        # Integration type 
        self.integrator = integrator

    def f(self, x, u):
        '''
        CT dynamics [mandatory function]
        '''
        return self.Ac.dot(x) + self.Bc.dot(u)

    def calc(self, x, u):
        '''
        DT dynamics [mandatory function]
        '''
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
        return xnext 
    
    def calcDiff(self, x, u):
        '''
        Get partial derivatives f_x, f_u at (x,u)
        '''
        return self.Ad, self.Bd

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
            X[i+1,:] = self.calc(X[i,:], U[i,:])
        return X, U

    def plot_traj(self, X, U):
        '''
        Plot trajectories X, U
        '''
        N = np.shape(U)[0]
        p = X[:,:self.nu]
        v = X[:,self.nu:-1]
        f = X[:,-1:]
        u = U
        # Create time spans for X and U
        tspan_x = np.linspace(0, N*self.dt, N+1)
        tspan_u = np.linspace(0, N*self.dt, N)
        # Create figs and subplots
        fig_x, ax_x = plt.subplots(1, 3)
        fig_u, ax_u = plt.subplots(1, 1)
        # Plot joints
        ax_x[0].plot(tspan_x, p, 'b-', label='pos')
        ax_x[0].set(xlabel='t (s)', ylabel='p (m)')
        ax_x[1].plot(tspan_x, v, 'g-', label='vel')
        ax_x[1].set(xlabel='t (s)', ylabel='v (m/s)')
        ax_x[2].plot(tspan_x, f, 'r-', label='force')
        ax_x[2].set(xlabel='t (s)', ylabel='f (N)')
        ax_u.plot(tspan_u, u, 'k-', label='control') 
        ax_u.set(xlabel='t (s)', ylabel='u (N)')
        # Legend
        # handles_x, labels_x = ax_x[0]get_legend_handles_labels()
        # fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
        fig_x.legend(loc='upper right', prop={'size': 16})
        # handles_u, labels_u = ax_u.get_legend_handles_labels()
        # fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
        fig_u.legend(loc='upper right', prop={'size': 16})
        # Titles
        fig_x.suptitle('State trajectories', size=16)
        fig_u.suptitle('Control trajectory', size=16)
        plt.show()



# class CartPole:
#     '''
#     Dynamics model of cart pole
#     '''

#     def __init__(self, dt=0.01, integrator='euler'):
#         # Dimensions
#         self.nx = 4
#         self.nu = 1
#         # Sampling time
#         self.dt = dt
#         # integrator ('euler' or 'rk4')
#         self.integrator = integrator 
#         # Params
#         self.m1 = 1.
#         self.m2 = .1
#         self.l = .5
#         self.g = 9.81
#         # Finite differences steps for numdiff
#         self.hx = .005
#         self.hu = .005 

#     def f(self, x, u):
#         '''
#         CT dynamics [mandatory]
#         '''
#         # Getting the state and control variables
#         p, th, v, w = x[0], x[1], x[2], x[3]
#         f = u #float(u)
#         # Parameters
#         m1, m2, l, g = self.m1, self.m2, self.l, self.g
#         s, c = np.sin(th.astype(np.float64)), np.cos(th.astype(np.float64))
#         mu = m1 + m2 * s**2
#         m = m1 + m2
#         # dynamics equations
#         dxdt0 = v
#         dxdt1 = w
#         dxdt2 = (1/mu)*(f + m2*s*(l*w**2 + g*c))
#         dxdt3 = (1/mu)*(-f*c -m2*l*w**2*s*c - m*g*s)
#         print(np.array([[dxdt0],[dxdt1],[dxdt2],[dxdt3]]))
#         return np.array([[dxdt0],[dxdt1],[dxdt2],[dxdt3]])

#     def calc(self, x, u):
#         '''
#         DT dynamics (discretization) [mandatory]
#         '''
#         if self.integrator=='euler':
#             xnext = x + self.f(x,u)*self.dt
#         else:
#             # RK4 step (default)
#             k1 = self.f(x, u) * self.dt
#             k2 = self.f(x + k1 / 2.0, u) * self.dt
#             k3 = self.f(x + k2 / 2.0, u) * self.dt
#             k4 = self.f(x + k3, u) * self.dt
#             xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
#         return xnext
    
#     def calcDiff(self, xs, us):
#         '''
#         Get partial derivatives of f along (xs,us) [mandatory]
#         '''
#         # to store
#         f_x = []
#         f_u = []
#         # fill 
#         for i in range(len(us)):
#             f_xi, f_ui = self.numdiff(xs[i],us[i]) 
#             f_x.append(f_xi)
#             f_u.append(f_ui)
#         return f_x, f_u

#     def numdiff(self, x, u):
#         '''
#         Evaluate partial derivatives at x,u using (centered) finite differences 
#         '''
#         # Call jacobian
#         dfdx = jacobian(self.f,0)(x,u)
#         dfdu = jacobian(self.f,1)(x,u)
#         return dfdx, dfdu

#         # # finite difference step
#         # Ac = np.zeros((self.nx, self.nx))
#         # Bc = np.zeros((self.nx, self.nu))
#         # # dfdx
#         # for i in range(self.nx):
#         #     # Get step in direction i
#         #     dx = np.zeros((self.nx, 1))
#         #     dx[i] = self.hx
#         #     # Get variations of f in direction i
#         #     df_dxi = self.f(x + dx, u) - self.f(x - dx, u) / (2*self.hx)
#         #     Ac[:,i] = df_dxi[:,0]
#         # # dfdu
#         # for i in range(self.nu):
#         #     # Get step in direction i
#         #     du = np.zeros((self.nu, 1))
#         #     du[i] = self.hu
#         #     # Get variations of f in direction i
#         #     df_dui = self.f(x, u + du) - self.f(x, u - du) / (2*self.hu)
#         #     Bc[:,i] = df_dui[:,0]
#         # # Discretize
#         # Ad = np.eye(self.nx) + self.dt*Ac
#         # Bd = self.dt*Bc + .5*self.dt**2*Ac.dot(Bc)
#         # return Ad, Bd

#     def rollout(self, x0, us):
#         '''
#         Rollout from x0 using us 
#         '''
#         N = len(us)
#         X = np.zeros((N+1, self.nx))
#         U = np.zeros((N, self.nu))
#         X[0,:] = x0.T
#         for i in range(N):
#             print("u :",us[i].T)
#             print("dxdt :", self.calc(X[i,:], us[i]))
#             U[i,:] = us[i].T
#             X[i+1,:] = self.calc(X[i,:], us[i].T)
#             # X[i+1,:] = self.calc(np.array([X[i,:]]).T, us[i].T).T
#         return X, U


#     def plot_traj(self, X, U):
#         '''
#         Plot trajectories X, U
#         '''
#         N = np.shape(U)[0]
#         p = X[:,0]
#         th = X[:,1]
#         v = X[:,2]
#         w = X[:,3]

#         u = U
#         # Create time spans for X and U
#         tspan_x = np.linspace(0, N*self.dt, N+1)
#         tspan_u = np.linspace(0, N*self.dt, N)
#         # Create figs and subplots
#         fig_x, ax_x = plt.subplots(4,1)
#         fig_u, ax_u = plt.subplots(1, 1)
#         # Plot joints
#         ax_x[0].plot(tspan_x, p, 'b-', label='pos')
#         ax_x[0].set(xlabel='t (s)', ylabel='p (m)')
#         ax_x[1].plot(tspan_x, th, 'b-', label='ang vel')
#         ax_x[1].set(xlabel='t (s)', ylabel='v (rad)')
#         ax_x[2].plot(tspan_x, v, 'b-', label='vel')
#         ax_x[2].set(xlabel='t (s)', ylabel='v (m/s)')
#         ax_x[3].plot(tspan_x, w, 'b-', label='ang vel')
#         ax_x[3].set(xlabel='t (s)', ylabel='v (rad/s)')
#         ax_u.plot(tspan_u, u, 'b-', label='force') 
#         ax_u.set(xlabel='t (s)', ylabel='u (N)')
#         # Legend
#         handles_x, labels_x = ax_x[0].get_legend_handles_labels()
#         fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
#         handles_u, labels_u = ax_u.get_legend_handles_labels()
#         fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
#         # Titles
#         fig_x.suptitle('State trajectories', size=16)
#         fig_u.suptitle('Control trajectory', size=16)
#         plt.show()