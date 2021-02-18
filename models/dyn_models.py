# Title : dyn_models.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Collection dynamics model classes compatible with custom DDP solver in ../core/ddp.py 
# Can also be used to initialize custom IAMs compatible with Crocoddyl , see croco_IAMs.py

import numpy as np
import matplotlib.pyplot as plt

# Make everything proper DAMs

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
    def __init__(self, m=1, K=1, B=1., p0=0., dt=0.01, integrator='euler'):
        # Dimensins
        self.nx = 3
        self.nu = 1
        # Sampling time
        self.dt = dt
        # Mass and stiffness
        self.m = m
        self.K = K
        self.B = B
        # Contact anchor point
        self.p0 = p0
        # CT dynamics
        self.Ac = np.array([[0, 1, 0],
                            [0, 0, 1/self.m],
                            [0, -self.K, -self.B/self.m]])
        self.Bc = np.array([[0],
                            [1/self.m],
                            [-self.B/self.m]])
        # DT model # That's Euler
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

    def get_residual(self, X, p0):
        '''
        Compute integration residual (force)
        '''
        res = np.zeros(np.shape(X)[0])
        for i in range(np.shape(X)[0]):
            res[i] = X[i,2] + self.K*(X[i,0]-p0) + self.B*X[i,1]
        return res

class PointMassLPF:
    '''
    Dynamics model of point mass in visco-elastic contact 
    with LPF on input (actuation dynamics)
    Variables: 
      State   : x = position, velocity, torque 
      Control : u = torque derivative
      Output  : force
    CT model:
      state transition : x'(t) = A x(t) + B u(t)
    DT model:  
      state transition : x(n+1) = Ad x(n) + Bd u(n)
    '''
    def __init__(self, m=1, k=0., dt=1e-2):
        # Dimensions
        self.nx = 3
        self.nu = 1
        # Euler step 
        self.dt = dt
        # 3rd-order CT dynamics
        self.m = m # mass
        self.k = k # coef. of 1st order actuation dynamics
        self.Ac = np.array([[0, 1, 0],
                            [0, 0, 1/self.m],
                            [0, 0, -self.k]])
        self.Bc = np.array([[0],
                            [0],
                            [self.k]])
        
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
        return x + self.f(x,u)*self.dt
    
    def calcDiff(self, x, u):
        '''
        Get partial derivatives f_x, f_u at (x,u)
        '''
        return np.eye(self.nx) + self.dt*self.Ac, self.dt*self.Bc

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

    def plot_traj(self, X, U, ref=None):
        '''
        Plot trajectories X, U
        '''
        N = np.shape(U)[0]
        p = X[:,:self.nu]
        v = X[:,self.nu:-1]
        tau = X[:,-1:]
        u = U
        # Create time spans for X and U
        tspan_x = np.linspace(0, N*self.dt, N+1)
        tspan_u = np.linspace(0, N*self.dt, N)
        # Create figs and subplots
        fig_x, ax_x = plt.subplots(3, 1)
        fig_u, ax_u = plt.subplots(1, 1)
        # Plot joints
        ax_x[0].plot(tspan_x, p, 'b-', label='pos')
        ax_x[0].set(xlabel='t (s)', ylabel='p (m)')
        ax_x[0].grid()
        ax_x[1].plot(tspan_x, v, 'g-', label='vel')
        ax_x[1].set(xlabel='t (s)', ylabel='v (m/s)')
        ax_x[1].grid()
        ax_x[2].plot(tspan_x, tau, 'r-', label='torque')
        ax_x[2].set(xlabel='t (s)', ylabel='tau (Nm)')
        ax_x[2].grid()
        # If ref specified
        if(ref is not None):
            ax_x[0].plot(tspan_x, [ref[0]]*(N+1), 'k-.', label='ref')
            ax_x[1].plot(tspan_x, [ref[1]]*(N+1), 'k-.')
            ax_x[2].plot(tspan_x, [ref[2]]*(N+1), 'k-.')
        ax_u.plot(tspan_u, u, 'k-', label='control') 
        ax_u.set(xlabel='t (s)', ylabel='w')
        ax_u.grid()
        # Legend
        handles_x, labels_x = ax_x[0].get_legend_handles_labels()
        fig_x.legend(loc='upper right', prop={'size': 16})
        handles_u, labels_u = ax_u.get_legend_handles_labels()
        fig_u.legend(loc='upper right', prop={'size': 16})
        # Titles
        fig_x.suptitle('State trajectories', size=16)
        fig_u.suptitle('Control trajectory', size=16)
        plt.show()




# import numpy as np
# from matplotlib import pyplot as plt

# # these packages for animating the robot env
# import IPython
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.animation import FuncAnimation


# class TwoDOFManipulator:
#     '''
#     Dynamics model of 2-DoF manipulator 
#     Variables: 
#       State   : x = joint positions, joint velocities = (q1, q2, dq1, dq2)
#       Control : u = joint torques = (tau1, tau2)
#     CT model:
#       state transition : x'(t) = A x(t) + B u(t)
#     DT model:  
#       state transition : x(n+1) = Ad x(n) + Bd u(n)
#     '''
#     def __init__(self, l1=0.1, l2=0.1, m1=0.5, m2=0.5, dt=0.01, integrator='euler'):
#         # Dimensions
#         self.nx = 4
#         self.nu = 2
#         # gravity vector
#         self.g = 9.81 
#         # Links lengths, masses and inertias (around CoM axis and rotor axis)
#         self.l1 = l1
#         self.l2 = l2
#         self.m1 = m1
#         self.m2 = m2
#         self.Il1 = self.m1*(self.l1**2)/12.0 
#         self.Il2 = self.m2*(self.l2**2)/12.0 
#         self.Im1 = 4*self.Il1 
#         self.Im2 = 4*self.Il2 
#         # Integrator
#         self.dt = dt 
#         self.integrator = integrator
        
#     def f(self, x, u):
#         '''
#         Compute the forward dynamics qddot = M^{-1}(tau-h) = b
#         Input:
#             x : state
#             u : control input
#         Output:
#             dxdt : acceleration
#         '''
#         # Extract vars for clarity
#         q1, q2, dq1, dq2, tau1, tau2 = x[0], x[1], x[2], x[3], u[0], u[1]        
#         # Compute generalized inertia matrix (LHS)
#         A = np.zeros((2, 2))
#         A[0,0] = self.Im1 + self.m2*self.l1**2 + self.Im2 + self.m2*self.l1*self.l2*np.cos(q2)
#         A[0,1] = self.Im2 + self.m2*self.l1*self.l2*np.cos(q2)/2.0
#         A[1,0] = self.Im2 + self.m2*self.l1*self.l2*np.cos(q2)/2.0
#         A[1,1] = self.Im2
#         # Compute gravity and coriolis (RHS)
#         b = np.zeros(2)
#         b[0] = tau1 + self.m2*self.l1*self.l2*dq1*dq2*np.sin(q2) + \
#                     self.m2*self.l1*self.l2*(dq2**2)*np.sin(q2)/2.0 - self.m2*self.l2*self.g*np.cos(q1+q2)/2.0 \
#                     - (self.m1*self.l1/2.0 + self.m2*self.l1)*self.g*np.cos(q1)
        
#         b[1] = tau2 - self.m2*self.l1*self.l2*(dq1**2)*np.sin(q2)/2.0 - self.m2*self.l2*self.g*np.cos(q1+q2)/2.0
#         # Forward dynamics (invert A)
#         A_inv = np.zeros((2,2))
#         A_inv[0,0] = A[1, 1]
#         A_inv[1,1] = A[0, 0]
#         A_inv[0,1] = -A[0,1]
#         A_inv[1,0] = -A[1,0]
#         A_inv = (1/np.linalg.det(A))*A_inv
#         # Acceleration
#         dxdt = np.zeros(self.nx)
#         dxdt[0] = dq1
#         dxdt[1] = dq2
#         dxdt[2:] = np.matmul(A_inv, b.T)
#         return dxdt
    
#     def calc(self, x, u):
#         '''
#         DT dynamics [mandatory function] 
#         '''
#         # Euler step
#         if(self.integrator=='euler'):
#             xnext = x + self.f(x,u)*self.dt
#         # RK4 step 
#         if(self.integrator=='rk4'):
#             k1 = self.f(x, u) * self.dt
#             k2 = self.f(x + k1 / 2.0, u) * self.dt
#             k3 = self.f(x + k2 / 2.0, u) * self.dt
#             k4 = self.f(x + k3, u) * self.dt
#             xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6
#         # Exact (default)
#         else:
#             raise Exception('Unknown integrator ! Please use "euler" or "rk4"')
#         return xnext 
    
#     def calcDiff(self, x, u):
#         '''
#         Get partial derivatives f_x, f_u at (x,u)
#         '''
#         f_x = np.zeros((self.nx, self.nx))
#         f_u = np.zeros((self.nu, self.nx)) 
#         # f_x = 
#         return f_x, f_u
    
#     def animate(self, freq = 25):
        
#         sim_data = self.sim_data[:,::freq]

#         fig = plt.figure()
#         ax = plt.axes(xlim=(-self.l1 - self.l2 -1, self.l1 + self.l2 + 1), ylim=(-self.l1 - self.l2 -1, self.l1 + self.l2 + 1))
#         text_str = "Two Dof Manipulator Animation"
#         arm1, = ax.plot([], [], lw=4)
#         arm2, = ax.plot([], [], lw=4)
#         base, = ax.plot([], [], 'o', color='black')
#         joint, = ax.plot([], [], 'o', color='green')
#         hand, = ax.plot([], [], 'o', color='pink')
        
#         def init():
#             arm1.set_data([], [])
#             arm2.set_data([], [])
#             base.set_data([], [])
#             joint.set_data([], [])
#             hand.set_data([], [])
            
#             return arm1, arm2, base, joint, hand
        
#         def animate(i):
#             theta1_t = sim_data[:,i][0]
#             theta2_t = sim_data[:,i][1]
            
#             joint_x = self.l1*np.cos(theta1_t)
#             joint_y = self.l1*np.sin(theta1_t)
            
#             hand_x = joint_x + self.l2*np.cos(theta1_t + theta2_t)
#             hand_y = joint_y + self.l2*np.sin(theta1_t + theta2_t)
            
#             base.set_data([0, 0])
#             arm1.set_data([0,joint_x], [0,joint_y])
#             joint.set_data([joint_x, joint_y])
#             arm2.set_data([joint_x, hand_x], [joint_y, hand_y])
#             hand.set_data([hand_x, hand_y])

#             return base, arm1, joint, arm2, hand
        
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=15,
#         verticalalignment='top', bbox=props)
#         ax.grid()
#         anim = FuncAnimation(fig, animate, init_func=init,
#                                        frames=np.shape(sim_data)[1], interval=25, blit=True)

#         plt.close(fig)
#         plt.close(anim._fig)
#         IPython.display.display_html(IPython.core.display.HTML(anim.to_html5_video()))

#     def plot(self):
#         '''
#         This function plots the joint positions, velocities and torques
#         '''
        
#         fig, axs = plt.subplots(3,1, figsize = (10, 10))
#         axs[0].plot((180/np.pi)*self.sim_data[0], label = 'joint position_1')
#         axs[0].plot((180/np.pi)*self.sim_data[1], label = 'joint position_2')
#         axs[0].grid()
#         axs[0].legend()
#         axs[0].set_ylabel("degrees")

#         axs[1].plot(self.sim_data[2], label = 'joint velocity_1')
#         axs[1].plot(self.sim_data[3], label = 'joint velocity_2')
#         axs[1].grid()
#         axs[1].legend()
#         axs[1].set_ylabel("rad/sec")
    
#         axs[2].plot(self.sim_data[4,:-1], label = 'torque_1')
#         axs[2].plot(self.sim_data[5,:-1], label = 'torque_2')
#         axs[2].grid()
#         axs[2].legend()
#         axs[2].set_ylabel("Newton/(Meter Second)")
    
#         plt.show()



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