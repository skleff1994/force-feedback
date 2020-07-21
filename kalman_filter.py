# Author: Sebastien
# My python implementation of Kalman filter

import numpy as np
import matplotlib.pyplot as plt
import time 

class KalmanFilter:
    '''
    Kalman filter
    '''
    
    def __init__(self, problem):
        '''
        DDP solver
        '''
        # Shooting problem
        self.problem = problem
        # Q-function
        self.Q_x = []
        self.Q_u = []
        self.Q_xx = []
        self.Q_uu = []
        self.Q_ux = []
        # Value function (optimal cost)
        self.V = []
        self.V_x = []
        self.V_xx = []        
        # Gains
        self.k = []
        self.K = []
        self.nx = self.problem.dynamics.nx
        self.nu = self.problem.dynamics.nu
        # State and control trajectories
        self.xs = []
        self.us = []
        self.xs.append(self.problem.x0)
        # Hessian reg 
        self.reg = 1e-6

        self.init_all()
    
    def init_all(self):
        '''
        Initialize Q / V models and trajs / gains to size them
        '''
        # Init all
        for i in range(self.problem.N):
            # Init Q model
            self.Q_x.append(np.zeros((self.nx,1)))
            self.Q_u.append(np.zeros((self.nu,1)))
            self.Q_xx.append(np.zeros((self.nx,self.nx)))
            self.Q_uu.append(np.zeros((self.nu,self.nu)))
            self.Q_ux.append(np.zeros((self.nu,self.nx)))
            # Init V model
            self.V.append(np.zeros((1,1)))
            self.V_x.append(np.zeros((self.nx,1)))
            self.V_xx.append(np.zeros((self.nx,self.nx)))
            # Init gains
            self.k.append(np.zeros((self.nu,1)))
            self.K.append(np.zeros((self.nu,self.nx)))
            # Init traj
            self.us.append(np.array([[0]]))
            self.xs.append(self.problem.dynamics.calc(self.xs[i], self.us[i]))
        # Terminal cost
        self.V.append(np.zeros((1,1)))
        self.V_x.append(np.zeros((self.nx,1)))
        self.V_xx.append(np.zeros((self.nx,self.nx)))

    def solve(self, maxiter=50, tol=1e-6):
        '''
        Main DDP loop: alternate between backward and forward pass
        '''
        start = time.time()
        i = 0
        converged = False
        while (i < maxiter and converged==False):
            self.backward_pass()
            self.forward_pass()
            # if (self.dJ <= tol):
            #     pass
            #     # converged = True
            #     # break
            # else:
            i+=1
        end = time.time()
        print("Converged in "+str(i)+" iterations ("+str(end-start)+" s).")
        return self.xs, self.us

    def backward_pass(self):
        '''
        Backward pass : dynamic programming recursion with quadratic models of cost and dynamics
        i.e. solve a sequence of LQR (Riccati) problems. Get quadratic model of the value function
        and feedforward / feedback gains (locally optimal policy) around (X,U)
        '''
        # Horizon
        N = self.problem.N
        # Compute cost along (xs,us)
        l = self.problem.calc(self.xs, self.us)
        print("Total cost: ", np.sum(l))
        # Compute derivatives   
        f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux = self.problem.calcDiff(self.xs, self.us)
        # Terminal value
        self.V[N] = l[N]
        self.V_x[N] = l_x[N]
        self.V_xx[N] = l_xx[N]
        # DP / Riccati sweep       
        for t in range(N):
            lx = l_x[N-t-1]
            lu = l_u[N-t-1]
            fx = f_x[N-t-1]
            fu = f_u[N-t-1]
            # lxx = l_xx[N-t-1] 
            # luu = l_uu[N-t-1] 
            # lux = l_ux[N-t-1] 
            Vp = self.V[N-1-t]
            Vxp = self.V_x[N-t]
            Vxxp = self.V_xx[N-t]
            # Get quadratic model of Q
            Qx = lx + fx.T.dot(Vxp)
            Qu = lu + fu.T.dot(Vxp)
            Qxx = l_xx[N-t-1] + fx.T.dot(Vxxp).dot(fx) # + contraction tensor
            Quu = l_uu[N-t-1] + fu.T.dot(Vxxp).dot(fu) # + contraction tensor
            Qux = l_ux[N-t-1] + fu.T.dot(Vxxp).dot(fx) # + contraction tensor
            # Invert Q_uu
            # inv_Quu = np.linalg.inv(Quu)
            # inv_Q_uu = np.linalg.inv(self.Q_uu[N-1-t]+ self.reg*np.eye(self.nu)) # with reg
            # Compute gains
            kff = -np.linalg.solve(Quu, Qu)
            kfb = -np.linalg.solve(Quu, Qux)
            # Quadratic model of V
            dV = .5*kff.T.dot(Quu).dot(kff) + kff.dot(Qu)
            self.V[N-1-t] = Vp + dV
            self.V_x[N-1-t] = Qx + kfb.T.dot(Quu).dot(kff) + kfb.T.dot(Qu) + Qux.T.dot(kff) 
            self.V_xx[N-1-t] = Qxx + kfb.T.dot(Quu).dot(kfb) + kfb.T.dot(Qux) + Qux.T.dot(kfb)
            # # Improved (reg)
            # self.V[N-1-t] = .5*self.k[N-1-t].T.dot(self.Q_uu[N-1-t]).dot(self.k[N-1-t]) + self.k[N-1-t].T.dot(self.Q_u[N-1-t]) #self.Q_u[N-1-t].T.dot(inv_Q_uu).dot(self.Q_u[N-1-t])
            # self.V_x[N-1-t] = self.Q_x[N-1-t] - self.Q_u[N-1-t].dot(inv_Q_uu).dot(self.Q_ux[N-1-t]).T
            # self.V_xx[N-1-t] = self.Q_xx[N-1-t] - self.Q_ux[N-1-t].T.dot(inv_Q_uu).dot(self.Q_ux[N-1-t])
            
            # Update gains
            self.k[N-1-t] = kff
            self.K[N-1-t] = kfb

            # Update Q value model
            self.Q_x[N-1-t] = Qx
            self.Q_u[N-1-t] = Qu
            self.Q_xx[N-1-t] = Qxx
            self.Q_uu[N-1-t] = Quu
            self.Q_ux[N-1-t] = Qux

    def forward_pass(self, tol=0.):
        '''
        Rollout from x0 using gains computed in the backward pass
        '''
        N = self.problem.N
        # Store trajs
        X_new = []
        U_new = []
        X_new.append(self.xs[0])
        for t in range(N):
            # Rollout
            U_new.append(self.us[t] + self.k[t] + self.K[t].dot(X_new[t] - self.xs[t]))
            X_new.append(self.problem.dynamics.calc(X_new[t], U_new[t]))
        # Calculate actual reduction in cost
        self.dJ = np.sum(self.problem.calc(self.xs, self.us)) - np.sum(self.problem.calc(X_new, U_new))
        # Update trajectories 
        self.xs = X_new
        self.us = U_new 

    def plot(self):
        '''
        Pot state and control trajectories
        '''
        p = []
        v = []
        u = []
        dt = self.problem.dt
        N = self.problem.N
        for i in range(N):
            p.append(self.xs[i][0])
            v.append(self.xs[i][1])
            u.append(self.us[i][0])
        # Create time spans for X and U
        tspan_x = np.linspace(0, N*dt, N)
        tspan_u = np.linspace(0, N*dt, N)
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

