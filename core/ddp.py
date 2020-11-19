# Title : ddp.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Custom DDP solver

import numpy as np
import matplotlib.pyplot as plt
import time 

class DDPSolver:
    '''
    DDP solver (iLQR)
    '''
    
    def __init__(self, model, dt, reg=1e-6):
        '''
        DDP solver
        '''
        # Robot model
        self.model = model 
        # Discretization 
        self.dt = dt
        # Store cost models and cost value
        self.running_costs = []
        self.terminal_costs = []
        # Hessian regularization
        self.reg = reg
    

    def init_all(self, T, xs_init = None, us_init = None):
        '''
        Initialize the shooting problem with horizon T and initial guess (xs_init, us_init)
        The initial guess MUST be feasible 
        '''
        # Number of knots in horizon
        self.N = int(float(T)/float(self.dt))      
        # Riccati gains
        self.k_ff = []
        self.k_fb = []
        # State and control dimensions
        self.nx = self.model.nx
        self.nu = self.model.nu
        # State and control trajectories
        self.xs = []
        self.us = []
        self.us = [np.zeros((self.nu, 1))]*self.N
        self.xs.append(np.array([[0],[0]]))
        for i in range(self.N):
            self.xs.append(self.model.calc(self.xs[i], self.us[i]))

    def add_running_cost(self, running_cost_model):
        '''
        Add a cost term in the running cost function 
        '''
        self.running_costs.append(running_cost_model)

    def add_terminal_cost(self, terminal_cost_model):
        '''
        Add a cost term in the terminal cost function 
        '''
        self.terminal_costs.append(terminal_cost_model)
    
    def running_calc(self, x, u):
        '''
        Get running cost at (x,u)
        '''
        l = 0
        for cost in self.running_costs:
            l += cost.calc(x, u)
        return l

    def terminal_calc(self, x):
        '''
        Get terminal cost at x
        '''
        l = 0
        for cost in self.terminal_costs:
            l += cost.calc(x)
        return l

    def running_calcDiff(self, x, u):
        '''
        Get partial derivatives of running cost at (x,u)
        '''
        l_x = np.zeros((self.model.nx, 1))
        l_xx = np.zeros((self.model.nx, self.model.nx))
        l_u = np.zeros((self.model.nu, 1))
        l_ux = np.zeros((self.model.nu, self.model.nx))
        l_uu = np.zeros((self.model.nu, self.model.nu))
        for cost in self.running_costs:
            c_x, c_u, c_xx, c_uu, c_ux = cost.calcDiff(x, u)
            l_x += c_x
            l_u += c_u
            l_xx += c_xx
            l_uu += c_uu
            l_ux += c_ux
        return l_x, l_u, l_xx, l_uu, l_ux

    def terminal_calcDiff(self, x):
        '''
        Get partial derivatives of terminal cost at x
        '''
        l_x = np.zeros((self.model.nx, 1))
        l_xx = np.zeros((self.model.nx, self.model.nx))
        for cost in self.terminal_costs:
            c_x, c_xx = cost.calcDiff(x)
            l_x += c_x
            l_xx += c_xx
        return l_x, l_xx


    def solve(self, maxiter=50, tol=1e-4):
        '''
        Main DDP loop: alternate between backward and forward pass
        '''
        start = time.time()
        i = 0
        self.dJ = np.inf
        while (i < maxiter and self.dJ > tol):
            self.backward_pass()  
            print("Iteration "+str(i)+" | COST = "+str(float(self.cost)))        
            self.forward_pass() 
            i+=1
        end = time.time()
        print("Converged in "+str(i)+" iterations ("+str(end-start)+" s).")
        print("Optimal cost = "+str(float(self.cost)))
        return self.xs, self.us


    def backward_pass(self):
        '''
        Backward pass : dynamic programming recursion with quadratic models of cost and dynamics
        i.e. solve a sequence of LQR (Riccati) problems. Get quadratic model of the value function
        and feedforward / feedback gains (locally optimal policy) around (X,U)
        '''
        # Reset current cost to 0
        self.cost = 0.
        # Riccati sweep       
        for t in range(self.N):
            # Value function model at terminal node
            if t==0:
                x = self.xs[-1].copy()
                V = self.terminal_calc(x)
                V_x, V_xx = self.terminal_calcDiff(x) 
                self.cost += V
            # Get current node
            x = self.xs[self.N-t-1].copy()
            u = self.us[self.N-t-1].copy()
            # Get calc and calcDiff at current node
            self.cost += self.running_calc(x, u)
            l_x, l_u, l_xx, l_uu, l_ux = self.running_calcDiff(x, u)
            f_x, f_u = self.model.calcDiff(x, u)
            # Construct quadratic model of Hamiltonian around current node
            Q_x = l_x + f_x.T.dot(V_x)
            Q_u = l_u + f_u.T.dot(V_x)
            Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x) # + contraction tensor
            Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u) # + contraction tensor
            Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x) # + contraction tensor
            # Compute Riccati gains
            k_ff = -np.linalg.solve(Q_uu + self.reg*np.eye(self.nu), Q_u)  #-inv_Quu.dot(Q_u)  
            k_fb = -np.linalg.solve(Q_uu + self.reg*np.eye(self.nu), Q_ux) #-inv_Quu.dot(Q_ux)
            # Quadratic model of V
            dV = -.5*k_ff.T.dot(Q_uu).dot(k_ff) 
            V = V + dV
            V_x = Q_x - k_fb.T.dot(Q_uu).dot(k_ff)  
            V_xx = Q_xx - k_fb.T.dot(Q_uu).dot(k_fb) 
            # Record gains
            self.k_ff.insert(0, k_ff)
            self.k_fb.insert(0, k_fb)
 

    def forward_pass(self, tol=0.):
        '''
        Rollout from x0 using gains computed in the backward pass
        '''
        # Store trajs
        X_new = []
        U_new = []
        X_new.append(self.xs[0])
        new_cost = 0.
        for t in range(self.N):
            # Rollout
            U_new.append(self.us[t] + self.k_ff[t] + self.k_fb[t].dot(X_new[t] - self.xs[t]))
            X_new.append(self.model.calc(X_new[t], U_new[t]))
            new_cost += self.running_calc(X_new[t], U_new[t]) 
        new_cost += self.terminal_calc(X_new[-1])
        # Calculate actual reduction in cost
        self.dJ = self.cost - new_cost
        # Update trajectories 
        self.xs = X_new
        self.us = U_new 
        self.cost = new_cost

    # def forward_pass(self, tol=0.):
    #     '''
    #     Rollout from x0 using gains computed in the backward pass
    #     '''
    #     N = self.problem.N
    #     # line search parameter
    #     alpha = 1.
    #     beta = .5
    #     # ls count
    #     ls_count = 0.
    #     while(ls_count<50):
    #         # Expected cost reduction
    #         dJ_est = 0
    #         # Store trajs
    #         X_new = []
    #         U_new = []
    #         X_new.append(self.xs[0])
    #         for t in range(N):
    #             # Rollout
    #             U_new.append(self.us[t] + alpha*self.k[t] + self.K[t].dot(X_new[t] - self.xs[t]))
    #             X_new.append(self.problem.dynamics.calc(X_new[t], U_new[t]))
    #             # Add up estimate of expected cost reduction 
    #             dJ_est = dJ_est + alpha*self.k[t].T.dot(self.Q_u[t]) + .5*alpha**2*self.k[t].T.dot(self.Q_uu[t]).dot(self.k[t])
    #         # Get actual cost reduction
    #         dJ_real = np.sum(self.problem.calc(self.xs, self.us)) + np.sum(self.problem.calc(X_new, U_new))
    #         print("  Estimated cost reduction : ", dJ_est)
    #         print("  Actual cost reduction : ", dJ_real)
    #         # Compare actual and expected cost reduction
    #         z = dJ_real / dJ_est
    #         # # print("  z= ", z)
    #         if(z>tol):
    #             # print("  z= ", z)
    #             break
    #         else:
    #             alpha = alpha*beta
    #             ls_count += 1
    #     self.xs = X_new
    #     self.us = U_new 


    def plot(self):
        '''
        Pot state and control trajectories
        '''
        p = []
        v = []
        u = []
        dt = self.dt
        N = self.N
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