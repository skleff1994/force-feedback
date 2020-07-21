import numpy as np

class QuadCost:
    '''
    Quadratic cost model to track a reference state + x/u reg
    '''
    def __init__(self, dynamics, x_ref, N, cost_weights):
        # dynamics
        self.dynamics = dynamics
        # Ref state
        self.x_ref = x_ref
        # Horizon
        self.N = N
        # Cost weights
        self.alpha = cost_weights[0]
        self.beta = cost_weights[1]
        self.gamma = cost_weights[2]
        # Weight matrices
        self.Q = np.eye(self.dynamics.nx)
        self.Q[0,0] = 100
        self.Q[1,1] = 100
        self.Qf = np.eye(self.dynamics.nx)
        self.Qf[1,1] = 100
        self.R = np.eye(self.dynamics.nu)
        # Dimension
        self.nx = self.dynamics.nx
        self.nu = self.dynamics.nu

    def calc(self, xs, us):
        '''
        Calculate costs along trajectory (xs,us)
        '''
        # to store costs
        l = []
        # fill running costs
        for i in range(self.N):
            l.append(self.running_cost(xs[i], us[i]))
        # Add terminal cost
        l.append(self.terminal_cost(xs[self.N]))
        return l

    def calcDiff(self, xs, us):
        '''
        Calculate partial derivatives of the cost along (xs,us)
        '''
        # to store partial derivatives
        l_x = []
        l_u = []
        l_xx = []
        l_uu = []
        l_ux = []
        # fill 
        for i in range(self.N):
            l_x.append(self.alpha*(xs[i]-self.x_ref))
            l_u.append(self.beta*us[i])
            l_xx.append(self.alpha*np.eye(self.nx))
            l_uu.append(self.beta*np.eye(self.nu))
            l_ux.append(np.zeros((self.nu,self.nx)))
        # add terminal cost
        l_x.append(self.gamma*xs[self.N])
        l_xx.append(self.gamma*np.eye(self.nx))
        return l_x, l_u, l_xx, l_uu, l_ux 

    def running_cost(self, x, u):
        '''
        Running cost l(x,u)
        '''
        return .5*self.alpha*(x-self.x_ref).T.dot(self.Q).dot(x-self.x_ref) + .5*self.beta*u.T.dot(self.R).dot(u) 

    def terminal_cost(self, xN):
        '''
        Terminal cost l(x)
        '''
        return .5*self.gamma*(xN-self.x_ref).T.dot(self.Qf).dot(xN-self.x_ref)