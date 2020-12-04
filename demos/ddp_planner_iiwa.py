"""
@package ddp_iiwa
@file ddp_planner.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief DDP trajectory optimizer based on Crocoddyl - Python API
"""

import numpy as np
import pinocchio as pin
import time
import matplotlib.pyplot as plt
import crocoddyl


class DDPPlanner:
    '''
    Trajectory optimizer based on DDP (Crocoddyl) + misc. handy functions
    Mirrored by the C++ API 
    '''

    def __init__(self, robot, dt, N):
        '''
        Initialize DDP planner
            robot : pinocchio wrapper 
            dt    : time discretization for OCP
            N     : number of steps in the OCP horizon
        '''
        # Robot
        self.robot = robot
        self.id_endeff = robot.model.getFrameId('contact')
        self.nq = robot.model.nq
        self.nv = robot.model.nv
        self.nu = self.nq
        # Horizon of OCP (number of steps)
        self.N = N
        # Time step size (s)
        self.dt = dt
        # Time horizon (s)
        self.T = self.N*self.dt
        # DDP solver
        self.ddp = None
        # Current plan
        self.xs = np.zeros((N+1, self.nq + self.nv))
        self.us = np.zeros((N, self.nq))
        # count calls to solver (for debugging)
        # self.nb_solve = 0
    

    def init_ocp(self, x0, desiredFramePlacement, runningCostWeights, terminalCostWeights, stateWeights, stateWeightsTerm, framePlacementWeights, interpolation=False):
        ''' 
        Set up the OCP for Crocoddyl define (cost models, shooting problem and solver)
            x0                     : initial state                             
            desiredFramePlacement  : desired end-effector placement (pin.SE3)
            runningCostWeights     : running cost function weights  [frame_placement, state_reg, ctrl_reg, state_limits] 
            terminalCostWeights    : terminal cost function weights [frame_placement, state_reg, state_limits]
            stateWeights           : activation weights for state regularization in running cost [w_x1,...,w_xn]
            stateWeightsTerm       : activation weights for state regularization in terminal cost [w_x1,...,w_xn]
            framePlacementWeights  : activation weights for frame placement in running & terminal cost [w_px, w_py, w_pz, w_Rx, w_Ry, w_Rz]
        The running cost looks like this   : l(x,u) = ||log_SE3|| + ||x|| + ||u|| + QuadBarrier(x)
        The terminale cost looks like this : l(x)   = ||log_SE3|| + ||x|| + QuadBarrier(x)
        '''
        # Set initial state
        self.x0 = x0
        # Warm start : initial state and gravity compensation torque
        self.xs = [self.x0] * (self.N+1)
        self.us = [pin.rnea(self.robot.model, self.robot.data, self.x0[:self.nq], np.zeros((self.nv,1)), np.zeros((self.nq,1)))] * self.N
        # Record cost weights
        self.desiredFramePlacement = desiredFramePlacement
        self.runningCostWeights = runningCostWeights
        self.terminalCostWeights = terminalCostWeights
        self.stateWeights = stateWeights
        self.stateWeightsTerm = stateWeightsTerm
        self.framePlacementWeights = framePlacementWeights
        # State and actuation models
        self.state = crocoddyl.StateMultibody(self.robot.model)
        self.actuation = crocoddyl.ActuationModelFull(self.state)
        # State  & regularizations
        self.xRegCost = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2), self.x0, self.actuation.nu)
        self.xRegCostTerm = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeightsTerm**2), self.x0, self.actuation.nu)
        self.uRegCost = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        # Adding the state limits penalization
        self.xLimitCost = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(self.state.lb, self.state.ub)), 0 * self.x0, self.actuation.nu)

        # Do not interpolate frame placements
        if(interpolation==False):
            # Running and terminal cost model
            self.runningCostModel = crocoddyl.CostModelSum(self.state, nu=self.actuation.nu)
            self.terminalCostModel = crocoddyl.CostModelSum(self.state, nu=self.actuation.nu)
            # Cost on end-effector frame placement 
            self.p_des = self.desiredFramePlacement.translation # record for use in plotting function
            self.framePlacementCost = crocoddyl.CostModelFramePlacement(self.state, crocoddyl.ActivationModelWeightedQuad(self.framePlacementWeights**2), crocoddyl.FramePlacement(self.id_endeff, self.desiredFramePlacement), self.actuation.nu)  
            # Add up running cost terms
            self.runningCostModel.addCost("endeff", self.framePlacementCost, self.runningCostWeights[0]) 
            self.runningCostModel.addCost("stateReg", self.xRegCost, self.runningCostWeights[1])
            self.runningCostModel.addCost("ctrlReg", self.uRegCost, self.runningCostWeights[2]) 
            self.runningCostModel.addCost("stateLim", self.xLimitCost, self.runningCostWeights[3]) 
            # Add up terminal control terms
            self.terminalCostModel.addCost("endeff", self.framePlacementCost, self.terminalCostWeights[0])
            self.terminalCostModel.addCost("stateRegTerm", self.xRegCostTerm, self.terminalCostWeights[1])
            self.terminalCostModel.addCost("stateLim", self.xLimitCost, self.terminalCostWeights[2]) 
            # Create IAMs
            self.runningModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.runningCostModel), self.dt)
            self.runningModel.differential.armature = np.array([.1]*7)
            # self.runningModel.differential.pinocchio.rotorInertia[:] = 1e-5
            # self.runningModel.differential.pinocchio.rotorGearRatio[:] = 1e2
            self.terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.terminalCostModel))
            # Create the shooting problem
            self.runningModels = [self.runningModel] * self.N

        # if interpolation , schedule interpolated frame placements in cost models
        if(interpolation):
            # One running cost model per knot
            self.runningCostModels = []
            self.runningModels = []
            # Initial placement
            M_i = self.robot.data.oMf[self.id_endeff] 
            # Final placement
            M_f = self.desiredFramePlacement
            # Interpolate and create cost models 
            n_reach = int(.75*self.N) # number of steps in reaching phase
            for i in range(self.N):
                # Placement
                if(i<=n_reach): # interpolate (reaching phase)
                    M_interp = M_i * pin.exp6( float(i)/(n_reach-1) * pin.log6(M_i.inverse()*M_f))  #M_i * pin.exp6( float(i)/(self.N-1) * pin.log6(M_i.inverse()*M_f)) 
                else: # maintain final placement
                    M_interp = M_f
                self.framePlacementCostInterp = crocoddyl.CostModelFramePlacement(self.state, crocoddyl.ActivationModelWeightedQuad(self.framePlacementWeights**2), crocoddyl.FramePlacement(self.id_endeff, M_interp), self.actuation.nu)
                # COst models
                self.runningCostModels.append(crocoddyl.CostModelSum(self.state, nu=self.actuation.nu))
                # print("runningcostModel", runningcostModel)
                self.runningCostModels[i].addCost("endeff", self.framePlacementCostInterp, self.runningCostWeights[0]) 
                # Add up running cost terms
                self.runningCostModels[i].addCost("stateReg", self.xRegCost, self.runningCostWeights[1])
                self.runningCostModels[i].addCost("ctrlReg", self.uRegCost, self.runningCostWeights[2]) 
                self.runningCostModels[i].addCost("stateLim", self.xLimitCost, self.runningCostWeights[3]) 
                # IAM
                self.runningModels.append(crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.runningCostModels[i]), self.dt))
                self.runningModels[i].differential.armature = np.array([.1]*7)
            # Terminal placmeent
            self.p_des = self.desiredFramePlacement.translation # record for use in plotting function
            self.terminalPlacementCost = crocoddyl.CostModelFramePlacement(self.state, crocoddyl.ActivationModelWeightedQuad(self.framePlacementWeights**2), crocoddyl.FramePlacement(self.id_endeff, self.desiredFramePlacement), self.actuation.nu)  
            # Create termnial cost
            self.terminalCostModel = crocoddyl.CostModelSum(self.state, nu=self.actuation.nu)
            # Add up terminal control terms
            self.terminalCostModel.addCost("endeff", self.terminalPlacementCost, self.terminalCostWeights[0])
            self.terminalCostModel.addCost("stateRegTerm", self.xRegCostTerm, self.terminalCostWeights[1])
            self.terminalCostModel.addCost("stateLim", self.xLimitCost, self.terminalCostWeights[2]) 
            # Create IAM
            self.terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.terminalCostModel))

        # Create the shooting problem
        self.problem = crocoddyl.ShootingProblem(self.x0, self.runningModels, self.terminalModel)
        # Creating the DDP solver 
        # self.ddp = crocoddyl.SolverDDP(problem)
        self.ddp = crocoddyl.SolverFDDP(self.problem)


    def update_ocp(self, newFramePlacement, interpolation=False):
        ''' 
        Update the OCP with a new reference end-effector placement
        '''
        # TODO : change the frame placement cost reference using Croco access functions 
        self.init_ocp(self.x0, newFramePlacement, self.runningCostWeights, self.terminalCostWeights, self.stateWeights, self.stateWeightsTerm, self.framePlacementWeights, interpolation)
        # # Update frame placement
        # self.desiredFramePlacement = newFramePlacement
        # # Update goal tracking cost model with new frame placemement
        # self.framePlacementCost = crocoddyl.CostModelFramePlacement(self.state, crocoddyl.ActivationModelWeightedQuad(self.framePlacementWeights**2), crocoddyl.FramePlacement(self.id_endeff, self.desiredFramePlacement), self.actuation.nu)  
        # # self.runningCostModel.addCost("endeff", self.framePlacementCost, self.runningCostWeights[0]) 
        

    def solve_ocp(self, x0, max_iter=1, callback=False):
        ''' 
        Solves the OCP and returns state-control optimal trajectoryelf.xs = self.ddp.xs
        self.us = self.ddp.us
            xs_init  : state trajectory initialization for warm-start (optional)
            us_init  : control trajectory initialization for warm-start (optional)
            callback : display logs in Crocoddyl solver
            max_iter : max number of DDP iterations
        '''
        # Update x0
        self.x0 = x0
        # Set callbacks, initial state and maxit
        if callback:
            self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        # set x0
        self.problem.x0 = self.x0 
        self.ddp = crocoddyl.SolverFDDP(self.problem)
        # Solve 
        self.ddp.solve(self.xs, self.us, max_iter, False)
        # Update trajectory
        self.xs = self.ddp.xs
        self.us = self.ddp.us
        return self.xs, self.us


    def update_plan(self, xs, us):
        ''' 
        Update state-control trajectory
            xs : state trajectory
            us : control trajectory
        '''
        self.xs = xs
        self.us = us


    def get_x(self):
        '''
        Returns state trajectory (q,v)
        '''
        x = np.empty((self.N+1, self.nq+self.nv))
        for i in range(self.N+1):
            x[i,:] = self.xs[i].T
        return x


    def get_q(self):
        '''
        Return joint positions q
        '''
        q = np.empty((self.N+1, self.nq))
        for i in range(self.N+1):
            q[i,:] = self.xs[i][:self.nq].T
        return q


    def get_v(self):
        '''
        Returns joint velocities v 
        '''
        v = np.empty((self.N+1, self.nv))
        for i in range(self.N+1):
            v[i,:] = self.xs[i][self.nv:].T
        return v


    def get_u(self):
        '''
        Returns joint torques u
        '''
        u = np.empty((self.N, self.nu))
        for i in range(self.N):
            u[i,:] = self.us[i].T
        return u


    def get_p(self, q):
        '''
        Returns end-effector trajectory for given q trajectory 
            q : joint positions
        '''
        N = np.shape(q)[0]
        p = np.empty((N,3))
        for i in range(N):
            pin.forwardKinematics(self.robot.model, self.robot.data, q[i])
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            p[i,:] = self.robot.data.oMf[self.id_endeff].translation.T
        # print(p)
        return p


    def plot(self):
        ''' 
        Generate nice plots
        '''

        # Joints, torques & endeff trajs
        q = self.get_q()
        v = self.get_v()
        u = self.get_u()
        p = self.get_p(q)

        # Desired endeff position
        p_des = self.p_des 
        # Create time spans for X and U
        tspan_x = np.linspace(0, self.N*self.dt, self.N+1)
        tspan_u = np.linspace(0, self.N*self.dt, self.N)

        # Create figs and subplots
        fig_x, ax_x = plt.subplots(self.nq, 2)
        fig_u, ax_u = plt.subplots(self.nq, 1)
        fig_p, ax_p = plt.subplots(3,1)

        # Plot joints
        for i in range(self.nq):
            ax_x[i,0].plot(tspan_x, q[:,i], 'b.', label='q_ddp', linestyle = 'None')
            ax_x[i,0].plot(tspan_x[-1], q[-1,i], 'ro')
    
            ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{i}$ (rad)')
            ax_x[i,0].grid()

            ax_u[i].plot(tspan_u, u[:,i], 'b.', label='u_ddp', linestyle = 'None') # feedforward term
            ax_u[i].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
            ax_u[i].grid()

            ax_x[i,1].plot(tspan_x, v[:,i], 'b.', label='v_ddp', linestyle = 'None')
            ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{i}$ (rad/s)')
            ax_x[i,1].grid()

            # Legend
            handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
            fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

            handles_u, labels_u = ax_u[i].get_legend_handles_labels()
            fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

        # Plot endeff
        # x
        ax_p[0].plot(tspan_x, p[:,0], 'b.', label='x_ddp', linestyle = 'None')
        ax_p[0].plot(tspan_x, p_des[0]*np.ones(self.N+1), 'k-.', label='x_ref')
        ax_p[0].set_title('x-position')
        ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
        ax_p[0].grid()
        # y
        ax_p[1].plot(tspan_x, p[:,1], 'b.', label='y_ddp', linestyle = 'None')
        ax_p[1].plot(tspan_x, p_des[1]*np.ones(self.N+1), 'k-.', label='y_ref')
        ax_p[1].set_title('y-position')
        ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
        ax_p[1].grid()
        # z
        ax_p[2].plot(tspan_x, p[:,2], 'b.', label='z_ddp', linestyle = 'None')
        ax_p[2].plot(tspan_x, p_des[2]*np.ones(self.N+1), 'k-.', label='z_ref')
        ax_p[2].set_title('z-position')
        ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
        ax_p[2].grid()

        handles_p, labels_p = ax_p[0].get_legend_handles_labels()
        fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

        # Titles
        fig_x.suptitle('Joint trajectories', size=16)
        fig_u.suptitle('Joint torques', size=16)
        fig_p.suptitle('End-effector trajectory', size=16)

        plt.show()