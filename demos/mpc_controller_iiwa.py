"""
@package ddp_iiwa
@file ddp_iiwa/mpc_controller.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief MPC controller based on DDP - Python API
"""

import numpy as np
import pinocchio as pin
import time
import matplotlib.pyplot as plt
from utils import *
from matplotlib.collections import LineCollection
import pybullet as p 
import matplotlib


class MPCController:
    ''' 
    Model Predictive Control based on Differential Dynamic Programming (Crocoddyl) 
        - "Impedance" MPC (i.e. offline MPC) : send Crocoddyl feedforward torque + PD correction term and replan based on predictions
        - "Vanilla" MPC (i.e. online MPC)    : send Crocoddyl feedforward torque and replan based on measurements
    The Python API is used for quick prototyping and is mirrored by the C++ API - which is itself used by the DG entities in the 'iiwa_robot' repo 
    '''
    
    def __init__(self, robot, ddp_planner, pd_gains, ctrl_freq=1e3, plan_freq=1e2, T_tot=5.):
        '''
        Initialize MPC controller
            robot        : pinocchio-bullet wrapper
            ddp_planner  : DDPPlanner object
            pd_gains     : joint impedance controller gains ([Kp,Kd])
            ctrl_freq    : control - simulation - frequency (Hz) 
            plan_freq    : MPC - planning - frequency (Hz) 
            T_tot        : total duration of the simulation (s)
        '''

        # Robot 
        self.robot = robot
        self.rmodel = robot.pin_robot.model
        self.rdata = robot.pin_robot.data
        self.nq = robot.pin_robot.model.nq
        self.nv = robot.pin_robot.model.nv
        self.nx = self.nq+self.nv
        self.nu = self.nq
        self.tau_default = np.zeros((robot.pin_robot.model.nq,1))
        self.id_endeff = self.rmodel.getFrameId('contact')

        # Planner
        self.ddp_planner = ddp_planner            # Trajectory planner
        self.T_tot = T_tot                        # Total duration of the simulation (s)
        
        # MPC & simulation parameters
        self.plan_freq = plan_freq                    # MPC re-planning frequency (Hz)
        self.ctrl_freq = ctrl_freq                    # Control - simulation - frequency (Hz)
        self.N_tot = int(self.T_tot*ctrl_freq)        # Total number of control steps in the simulation (s)
        self.N_c = int(self.ctrl_freq/self.plan_freq) # Number of control steps between each MPC replanning

        # # Target end-effector pose
        self.M_des = self.ddp_planner.desiredFramePlacement

        # PD gains
        self.Kp = pd_gains[0]       # P gain 
        self.Kd = pd_gains[1]       # D gain

        print('------------------')
        print('MPC controller ready')
        print('------------------')
        print('Total simulation duration : T_tot = '+str(self.T_tot)+' s')
        print('Control frequency : f_ctrl = '+str(self.ctrl_freq)+' Hz')
        print('Replanning frequency : f_plan = '+str(self.plan_freq)+' Hz')
        print('MPC horizon : T_ocp = '+str(self.ddp_planner.N*self.ddp_planner.dt)+' s')
        print('    OCP integration step : dt_ocp = '+str(self.ddp_planner.dt)+' s')
        print('    Number of knots in MPC horizon : N_ocp = '+str(self.ddp_planner.N)+' steps')
        print('Number of control steps per planning step = '+str(self.N_c))
        print('Total number of control steps = '+str(self.N_tot))


    def init_data(self):
        '''
        Initialize trajectories containers and variables
        '''
        # Measured states from PyBullet
        self.X_mea = np.zeros((self.N_tot+1, self.nx))
        q_mea_0, v_mea_0 = self.robot.get_state()
        self.X_mea[0, :] = np.concatenate([q_mea_0, v_mea_0]).T

        # Desired states (interpolated predictions)
        self.X_des = np.zeros((self.N_tot+1, self.nx)) 
        q_des_0 = a2m(self.ddp_planner.get_q()[0])
        v_des_0 = a2m(self.ddp_planner.get_v()[0])
        self.X_des[0, :] = np.concatenate([q_des_0, v_des_0]).T

        # Total number of OCPs (replan) solved during the simulation
        self.N_p = int(self.T_tot*self.plan_freq)        
        # Number of knots per OCP
        self.N_h = self.ddp_planner.N        

        # MPC predictions (DDP plans)
        self.X_pred = np.zeros((self.N_p+1, self.N_h+1, self.nx))  
        self.X_pred[0, :, :] = self.ddp_planner.get_x()

        # Initial desired state and control
        q_des_1 = a2m(self.ddp_planner.get_q()[1]) 
        v_des_1 = a2m(self.ddp_planner.get_v()[1]) 
        u_prev = a2m(self.ddp_planner.get_u()[0]) 
        u_next = a2m(self.ddp_planner.get_u()[0]) 

        # Feedforward torques planned by MPC (DDP) 
        self.U_des = np.zeros((self.N_tot, self.nq))

        # Torques sent to PyBullet
        self.U_mea = np.zeros((self.N_tot, self.nq))

        print("Initialized data :")
        print("  x_mea_0 :", np.concatenate([q_mea_0, v_mea_0]).T)
        print("  x_des_0 :", np.concatenate([q_des_0, v_des_0]).T)
        print("  x_des_1:", np.concatenate([q_des_1, v_des_1]).T)
        print("  u_prev :", u_prev.T)
        print("  u_next :", u_next.T)
        print(self.ddp_planner.get_u())
        return q_des_0, v_des_0, q_des_1, v_des_1, u_prev, u_next


    def record_data(self, q_mea, v_mea, q_des, v_des, tau, u_des, i):
        '''
        Record data
        '''
        self.X_mea[i+1, :] = np.concatenate([q_mea, v_mea]).T  # Measured state
        self.X_des[i+1, :] = np.concatenate([q_des, v_des]).T  # Desired state
        self.U_des[i, :] = u_des.T                             # Desired torque
        self.U_mea[i, :] = tau.T                               # Actual torque


    def get_plan(self, x0, t=0):
        ''' 
        Compute motion plan with the DDP motion planner
            xs_init : state trajectory initialization (for DDP warm start)
            us_init : control trajectory initialization (for DDP warm start)
        '''
        # Old
        # self.ddp_planner.init_ocp(x0, self.M_des, runningCostWeights, terminalCostWeights, stateWeights, stateWeightsTerm, framePlacementWeights, interpolation=False)
        # Update target if necesary
        # self.ddp_planner.update_ocp(self.M_des)

        # Solve 
        xs, us = self.ddp_planner.solve_ocp(x0, max_iter=2)
        
        # Warm start handled internally by ddp_planner (using previous trajectory)
        # # Warm start = x0 + previous traj
        # xs_init = self.ddp_planner.xs #[x0] + self.ddp_planner.xs[1:]
        # us_init = self.ddp_planner.us
        # self.ddp_planner.update_plan(xs_init, us_init)

        # # Solve OCP 
        # xs, us = self.ddp_planner.solve_ocp()
        # self.ddp_planner.update_plan(xs, us)
        # print(xs)
        return xs, us


    def pd_control(self, q_des, v_des, q_mea, v_mea):
        '''
        Classical joint-space PD controller
            q_des : desired joint positions
            v_des : desired joint velocities
            q_mea : measured joint positions
            v_mea : measured joint velocities
        '''
        # Error on joints
        err_q = np.array([q_mea]).T - q_des
 
        # Error on joint vel
        err_v = np.array([v_mea]).T - v_des

        # Desired acceleration = PD law (impedance)
        return -self.Kp.dot(err_q) - self.Kd.dot(err_v)


    def interpolate(self, v1, v2, i, N):
        '''
        Returns the value of the linear inteprolation between 
        v1 and v2 at step i when [v1,v2] is divided into N points 
        '''
        if self.N_c != 1:
            coef = float((i)%self.N_c) / (self.N_c )
        else:
            coef = 1
        return (1-coef)*v1 + coef*v2


    def run_impedance(self, riccati=False, with_gravity_compensation=False, ballId=None, max_iter=1):
        '''
        Tracks a ball in PyBullet using the impedance controller [ HEAVY MODIFICATION ]
        '''

        # Timings
        total_solve_time = 0
        total_plan_time = 0
        total_ctrl_time = 0

        # print(self.X_des)
        # Init data
        q_des_0, v_des_0, q_des_1, v_des_1, u_prev, u_next = self.init_data() 

        # Replan counter
        nb_replan = 0

        # Init model and gravity data 
        if(with_gravity_compensation == True):
            model = self.rmodel
            dataGravity = model.createData()

        # MPC loop with PD+
        for i in range(self.N_tot):

            print("Ctrl step "+str(i)+"/"+str(self.N_tot))

            # Replan if we are in a planning cycle
            if(i%self.N_c == 0 and i>0):
                
                print("================================")
                print("  Replan step "+str(nb_replan)+"/"+str(self.T_tot*self.plan_freq))
                print("================================")

                start_plan_time = time.time()

                # Increment replan counter
                nb_replan += 1

                # New initial state for replanning = prediction
                x0 = np.concatenate([q_des_1, v_des_1])
            
                # If dynamic tracking , update target EE pose
                if(ballId is not None):
                    p_des, R_des = p.getBasePositionAndOrientation(ballId)
                    self.M_des = pin.SE3(np.eye(3),  np.matrix(p_des).T)

                # REPLAN
                start_solve_time = time.time()
                # Solve OCP 
                self.ddp_planner.solve_ocp(x0, max_iter)
                end_solve_time = time.time()
                total_solve_time += end_solve_time - start_solve_time

                # UPDATE variables with new plan
                    # Record new set of predictions
                self.X_pred[nb_replan,:,:] = self.ddp_planner.get_x()
                    # Update desired state vars
                q_des_0 = q_des_1.copy()
                q_des_1 = a2m(self.ddp_planner.get_q()[1])
                # print(" NEW plan : ", q_des_1[0])
                v_des_0 = v_des_1.copy()
                v_des_1 = a2m(self.ddp_planner.get_v()[1])
                    # Update desired control 
                u_prev = u_next
                u_next = a2m(self.ddp_planner.get_u()[0])
                # Measure planning time
                end_plan_time = time.time()
                total_plan_time += end_plan_time - start_plan_time
                
            # Measure control time
            start_ctrl_time = time.time() 

            # Measure current state from simulation & update pinocchio model
            q_mea, v_mea = self.robot.get_state()
            self.robot.forward_robot(q_mea, v_mea) 
            self.X_mea[i, :] = np.concatenate([q_mea, v_mea]).T  # Record desired state (next)

            # Desired state = linear interpolation of current and next predictions 
            q_des = self.interpolate(q_des_0, q_des_1, i, self.N_c)
            v_des = self.interpolate(v_des_0, v_des_1, i, self.N_c)

            ### FEEDFORWARD TERM
                # Desired control = linear interpolation of current and next predictions
            u_des = self.interpolate(u_prev, u_next, i, self.N_c)
                # OR optional gravity compensation
            if(with_gravity_compensation == True):
                u_des = np.array([pin.rnea(model, dataGravity, q_des, np.zeros(self.nq), np.zeros(self.nq))]).T

            ### FEEDBACK TERM
            print(v_des)
            self.X_des[i+1, :] = np.concatenate([q_des, v_des]).T  # Record desired state (next)
                # PD control law
            tau_fb = self.pd_control(q_des, v_des, q_mea, v_mea)
                # Optionally use Riccati gains (instead of PD)
            if(riccati==True):
                # Try Riccati gains
                K_riccati = self.ddp_planner.ddp.K[0]
                x_mea = np.array([np.concatenate([q_mea, v_mea])]).T
                x_des = np.concatenate([q_des, v_des])
                tau_fb = -K_riccati.dot(x_mea - x_des) 
            # Total torque command
            tau = tau_fb + u_des
            self.U_des[i, :] = u_des.T                             # Record desired torque (currnt)
            self.U_mea[i, :] = tau.T                               # Record actual torque (current)
            # Send torques to robot
            self.robot.send_joint_command(tau)
            # Step the simulator
            p.stepSimulation()

            # Measure control time
            end_ctrl_time = time.time()
            total_ctrl_time += end_ctrl_time - start_ctrl_time
        
        # Final state
        q_mea, v_mea = self.robot.get_state()
        self.X_mea[-1, :] = np.concatenate([q_mea, v_mea]).T  # Measured state
        self.X_des[-1, :] = np.concatenate([q_des_1, v_des_1]).T  # Desired state

        # # Calculate average time of planning step (DDP)
        # avg_solve_time = total_solve_time/self.N_p
        # avg_plan_time = total_plan_time/self.N_p
        # avg_ctrl_time = total_ctrl_time/self.N_tot
        # # Logs
        # print('------------------')
        # print('Average OCP solving time: '+str(avg_solve_time))
        # print('Average planning loop duration : '+str(avg_plan_time))
        # print('Average control loop duration : '+str(avg_ctrl_time))
        # print('------------------')

        # Disconnect pybullet
        p.disconnect()


    def run_vanilla(self, riccati=False, ballId=None):
        '''
        Vanilla controller : use sensor feedback from simulator direclty in MPC to replan, no impedance controller [ HEAVY MODIF ]
        '''

        # Timings
        total_solve_time = 0
        total_plan_time = 0
        total_ctrl_time = 0

        # Init data
        q0, v0, q_des, v_des, u_prev, u_next = self.init_data()

        # Replan counter
        nb_replan = 0
    
        # MPC loop with PD+
        for i in range(self.N_tot):
            
            print("Sim step "+str(i)+"/"+str(self.N_tot))

            # print("Desired x1 : ", q_des.T, v_des.T)
            self.X_des[i+1, :] = np.concatenate([q_des, v_des]).T  # Record desired state (next)

            # Measure control time
            start_ctrl_time = time.time()

            # Measure current state & update pinocchio model
            q_mea, v_mea = self.robot.get_state()
            self.robot.forward_robot(q_mea, v_mea)  
            # print("Measured x0 : ", q_mea, v_mea)
            self.X_mea[i, :] = np.concatenate([q_mea, v_mea]).T  # Record measured state
            
            # Desired control = linear interpolation of current and next predictions
            u_des = self.interpolate(u_prev, u_next, i, self.N_c)
            # Torque control law
            tau = u_des   
            # Optionally add Riccati feedback term to feedforward torque
            if(riccati==True):
                # Try Riccati gains
                K_riccati = self.ddp_planner.ddp.K[0] 
                x_mea = np.array([np.concatenate([q_mea, v_mea])]).T
                x_des = np.concatenate([q_des, v_des])
                u_riccati = K_riccati.dot(x_mea - x_des) 
                tau = u_des - u_riccati 
            # print("Desired u0: ", tau)
            # print("Total u0 : ", u_des)
            self.U_des[i, :] = u_des.T                             # Record desired torque (currnt)
            self.U_mea[i, :] = tau.T                               # Record actual torque (current)
            # Send torques to robot
            self.robot.send_joint_command(tau)

            # Step the simulator
            p.stepSimulation()

            # Replan if we are in a planning cycle
            if(i%self.N_c == 0 and i>0):
                
                print("================================")
                print("  Replan...")
                print("================================")

                start_plan_time = time.time()

                # Increment replan counter
                nb_replan += 1

                # Initial step for replan = measured state 
                x0 = np.concatenate([q_mea, v_mea])
                print(x0)
                # If dynamic tracking , update target EE pose
                if(ballId is not None):
                    p_des, R_des = p.getBasePositionAndOrientation(ballId)
                    self.M_des = pin.SE3(np.eye(3),  np.matrix(p_des).T)

                # REPLAN
                start_solve_time = time.time()
                # Solve OCP 
                self.get_plan(x0) 
                end_solve_time = time.time()
                total_solve_time += end_solve_time - start_solve_time

                # UPDATE based on new plan
                # Record current prediction (DDP plan)
                self.X_pred[nb_replan,:,:] = self.ddp_planner.get_x()
                # Update desired state from new plan
                q_des = a2m(self.ddp_planner.get_q()[1])
                v_des = a2m(self.ddp_planner.get_v()[1])
                # Update desired control from new plan
                u_prev = u_next
                u_next = a2m(self.ddp_planner.get_u()[0])

                # # Measure planning time
                end_plan_time = time.time()
                total_plan_time += end_plan_time - start_plan_time

            # Measure control time
            end_ctrl_time = time.time()
            total_ctrl_time += end_ctrl_time - start_ctrl_time

        # Final state
        q_mea, v_mea = self.robot.get_state()
        self.X_mea[-1, :] = np.concatenate([q_mea, v_mea]).T  # Measured state
        # self.X_des[-1, :] = np.concatenate([q_des, v_des]).T  # Desired state

        # Calculate average time of planning step (DDP)
        avg_solve_time = total_solve_time/self.N_p
        avg_plan_time = total_plan_time/self.N_p
        avg_ctrl_time = total_ctrl_time/self.N_tot
        # Logs
        print('------------------')
        print('Average OCP solving time: '+str(avg_solve_time))
        print('Average planning loop duration : '+str(avg_plan_time))
        print('Average control loop duration : '+str(avg_ctrl_time))
        print('------------------')

        # Disconnect pybullet
        p.disconnect()


    def run_hybrid (self, riccati=False, with_gravity_compensation=False, ballId=None):
        '''
        ##### DEPRECATED #######
        Hybrid controller : use sensor feedback from simulator both in MPC to replan and in impedance controller
        '''

        # Timings
        total_solve_time = 0
        total_plan_time = 0
        total_ctrl_time = 0

        # Init data
        q_des_0, v_des_0, q_des_1, v_des_1, u_prev, u_next = self.init_data() 
        print("  interpolate from "+str(u_prev.T)+" to "+str(u_next.T))
        # Replan counter
        nb_replan = 0

        # Init model and gravity data 
        if(with_gravity_compensation == True):
            model = self.rmodel
            dataGravity = model.createData()

        # MPC loop with PD+
        for i in range(self.N_tot):

            print("Ctrl step "+str(i)+"/"+str(self.N_tot))

            # Replan if we are in a planning cycle
            if(i%self.N_c == 0 and i>0):
                
                print("================================")
                print("  Replan step "+str(nb_replan)+"/"+str(self.T_tot*self.plan_freq))
                print("================================")
                start_plan_time = time.time()
                # Increment replan counter
                nb_replan += 1
                # New initial state for replanning = prediction
                x0 = np.concatenate([q_mea, v_mea])
            
                # If dynamic tracking , update target EE pose
                if(ballId is not None):
                    p_des, R_des = p.getBasePositionAndOrientation(ballId)
                    self.M_des = pin.SE3(np.eye(3),  np.matrix(p_des).T)

                # REPLAN
                start_solve_time = time.time()
                # Solve OCP 
                self.get_plan(x0)
                end_solve_time = time.time()
                total_solve_time += end_solve_time - start_solve_time

                # UPDATE variables with new plan
                    # Record new set of predictions
                self.X_pred[nb_replan,:,:] = self.ddp_planner.get_x()
                    # Update desired state vars
                q_des_0 = q_des_1
                q_des_1 = a2m(self.ddp_planner.get_q()[1])
                v_des_0 = v_des_1
                v_des_1 = a2m(self.ddp_planner.get_v()[1])
                    # Update desired control 
                u_prev = u_next
                u_next = a2m(self.ddp_planner.get_u()[0])
                print("  interpolate from "+str(u_prev.T)+" to "+str(u_next.T))
                # Measure planning time
                end_plan_time = time.time()
                total_plan_time += end_plan_time - start_plan_time

            # Measure control time
            start_ctrl_time = time.time() 
            # Measure current state from simulation & update pinocchio model
            q_mea, v_mea = self.robot.get_state()
            self.robot.forward_robot(q_mea, v_mea) 
            ### FEEDFORWARD TERM
                # Desired control = linear interpolation of current and next predictions
            u_des = self.interpolate(u_prev, u_next, i, self.N_c)
            print("      > u_des "+str(u_des.T))
                # OR optional gravity compensation
            if(with_gravity_compensation == True):
                u_des = np.array([pin.rnea(model, dataGravity, q_mea, v_mea, np.zeros((self.nq,1)))]).T
            ### FEEDBACK TERM
                # Desired state = linear interpolation of current and next predictions 
            q_des = self.interpolate(q_des_0, q_des_1, i, self.N_c)
            v_des = self.interpolate(v_des_0, v_des_1, i, self.N_c)
                # PD control law
            tau_fb = self.pd_control(q_des, v_des, q_mea, v_mea)
                # Optionally use Riccati gains (instead of PD)
            if(riccati==True):
                # Try Riccati gains
                K_riccati = self.ddp_planner.ddp.K[0]
                x_mea = np.array([np.concatenate([q_mea, v_mea])]).T
                x_des = np.concatenate([q_des, v_des])
                tau_fb = -K_riccati.dot(x_mea - x_des) 

            # Total torque command
            tau = tau_fb + u_des
            # Send torques to robot
            self.robot.send_joint_command(tau)
            # Step the simulator
            p.stepSimulation()
            # Record trajectories
            self.record_data(q_mea, v_mea, q_des, v_des, tau, u_des, i)
            # Measure control time
            end_ctrl_time = time.time()
            total_ctrl_time += end_ctrl_time - start_ctrl_time
        
        # Final state
        q_mea, v_mea = self.robot.get_state()
        self.X_mea[-1, :] = np.concatenate([q_mea, v_mea]).T  # Measured state
        self.X_des[-1, :] = np.concatenate([q_des_1, v_des_1]).T  # Desired state

        # Calculate average time of planning step (DDP)
        avg_solve_time = total_solve_time/self.N_p
        avg_plan_time = total_plan_time/self.N_p
        avg_ctrl_time = total_ctrl_time/self.N_tot
        # Logs
        print('------------------')
        print('Average OCP solving time: '+str(avg_solve_time))
        print('Average planning loop duration : '+str(avg_plan_time))
        print('Average control loop duration : '+str(avg_ctrl_time))
        print('------------------')

        # Disconnect pybullet
        p.disconnect()


    def plot(self, with_predictions=False):
        ''' 
        Generate nice plots
            X_mea            : measured state trajectory
            X_des            : desired sate trajectory (linear interpolation of X_pred)
            Tau              : actual control trajectory (tau_ff+tau_fb)
            Tau_des          : desired torque trajectory (DDP plan = tau_ff)
            X_pred           : desired state trajectory (DDP plan)
            with_predictions : optionally plot predictions
        '''

        # Total number of control knots in the full trajectory = N_tot*N_ctrl 
        N = self.N_tot

        # Total number of planning steps = N_tot+1
        N_p = int(self.T_tot*self.plan_freq)

        # Number of planning steps dividing the MPC horizon (lookahead)
        N_h = self.ddp_planner.N   

        # Robot dimensions
        nq = self.nq
        nv = self.nv
        nu = self.nu

        # Time step duration of the control loop
        dt_ctrl = float(1./self.ctrl_freq)
        # Time step duration of planning loop
        dt_plan = float(1./self.plan_freq)

        # Joints & torques
            # State predictions (MPC)
        q_pred = self.X_pred[:,:,:nq]
        v_pred = self.X_pred[:,:,nv:]

            # State measurements (PyBullet)
        q_mea = self.X_mea[:,:nq]
        v_mea = self.X_mea[:,nv:]
            # Control sent to PyBullet = u_des (ff) + u_fb
        u = self.U_mea
            # 'Desired' state = interpolated predictions
        q_des = self.X_des[:,:nq]
        v_des = self.X_des[:,nv:]
            # 'Desired' control = interpolation of DDP ff torques 
        u_des = self.U_des

        # End-effector
        # print(np.shape(q_des))
        p_mea = self.ddp_planner.get_p(q_mea) #np.empty((N+1,3))
        p_des = self.ddp_planner.get_p(q_des) #np.empty((N+1,3))

        # Create time spans for X and U
        tspan_x = np.linspace(0, self.T_tot, N+1)
        tspan_u = np.linspace(0, self.T_tot, N)

        # Create figs and subplots
        fig_x, ax_x = plt.subplots(nq, 2)
        fig_u, ax_u = plt.subplots(nq, 1)
        fig_p, ax_p = plt.subplots(3,1)

        # For each joint
        for i in range(nq):

            # Extract state predictions of i^th joint
            q_pred_i = q_pred[:,:,i]
            v_pred_i = v_pred[:,:,i]

            if(with_predictions):
                # For each planning step in the trajectory
                for j in range(N_p):
                    # Receding horizon = [j,j+N_h]
                    tspan_x_pred = np.linspace(j*dt_plan, (j+N_h)*dt_plan, N_h+1)
                    # Set up lists of (x,y) points for predicted positions and velocities
                    points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                    points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                    # Set up lists of segments
                    segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                    segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                    # Make collections segments
                    cm = plt.get_cmap('Greys_r') #Greys_r
                    lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                    lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                    lc_q.set_array(tspan_x_pred) #np.linspace(0, 1, N_h+1))
                    lc_v.set_array(tspan_x_pred) #np.linspace(0, 1, N_h+1))
                    # Customize
                    lc_q.set_linestyle('-')
                    lc_v.set_linestyle('-')
                    lc_q.set_linewidth(1)
                    lc_v.set_linewidth(1)
                    # Plot collections
                    ax_x[i,0].add_collection(lc_q)
                    ax_x[i,1].add_collection(lc_v)
                    # Scatter to highlight points
                    colors = np.r_[np.linspace(0.1, 1, N_h+1), np.linspace(0.1, 1, N_h+1)]
                    my_colors = cm(colors)
                    ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                    ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
            

            # Desired joint position (interpolated from prediction)
            ax_x[i,0].plot(tspan_x, q_des[:,i], 'b-', label='Desired')
            # Measured joint position (PyBullet)
            # ax_x[i,0].plot(tspan_x, q_mea[:,i], 'r-', label='Measured')
            ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{i}$ (rad)')
            ax_x[i,0].grid()

            # Desired joint velocity (interpolated from prediction)
            ax_x[i,1].plot(tspan_x, v_des[:,i], 'b-', label='Desired')
            # Measured joint velocity (PyBullet)
            # ax_x[i,1].plot(tspan_x, v_mea[:,i], 'r-', label='Measured')
            ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{i}$ (rad/s)')
            ax_x[i,1].grid()

            # Desired joint torque (interpolated feedforward)
            ax_u[i].plot(tspan_u, u_des[:,i], 'b-.', label='Desired ff (MPC)')
            # Feedback term
            # ax_u[i].plot(tspan_u, u[:,i]-u_des[:,i], 'g-.', label='Desired fb (PD)') 
            # Total torque applied = ff + fb
            # ax_u[i].plot(tspan_u, u[:,i], 'r-', label='Total ff + fb')
            ax_u[i].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
            ax_u[i].grid()

            # Legend
            handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
            fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

            handles_u, labels_u = ax_u[i].get_legend_handles_labels()
            fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

        # Plot endeff
        # x
        ax_p[0].plot(tspan_x, [self.M_des.translation[0]]*(N+1), 'k-.', label='REF')
        ax_p[0].plot(tspan_x, p_des[:,0], 'b-', label='des (ddp)')
        ax_p[0].plot(tspan_x, p_mea[:,0], 'r-.', label='sim')
        ax_p[0].set_title('x-position')
        ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
        # y
        ax_p[1].plot(tspan_x, [self.M_des.translation[1]]*(N+1), 'k-.', label='REF')
        ax_p[1].plot(tspan_x, p_des[:,1], 'b-', label='err_y (m)')
        ax_p[1].plot(tspan_x, p_mea[:,1], 'r-.', label='sim')
        ax_p[1].set_title('y-position')
        ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
        # z
        ax_p[2].plot(tspan_x, [self.M_des.translation[2]]*(N+1), 'k-.', label='REF')
        ax_p[2].plot(tspan_x, p_des[:,2], 'b-', label='err_z (m)')
        ax_p[2].plot(tspan_x, p_mea[:,2], 'r-.', label='sim')
        ax_p[2].set_title('z-position')
        ax_p[2].set(xlabel='t (s)', ylabel='z (m)')

        handles_p, labels_p = ax_p[0].get_legend_handles_labels()
        fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

        # Titles
        fig_x.suptitle('Joint trajectories: des. vs sim. (DDP-based MPC)', size=16)
        fig_u.suptitle('Joint torques: des. vs sim. (DDP-based MPC)', size=16)
        fig_p.suptitle('End-effector: ref. vs des. vs sim. (DDP-based MPC)', size=16)

        plt.show() 