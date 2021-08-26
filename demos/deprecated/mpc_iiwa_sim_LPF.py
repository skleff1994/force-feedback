"""
@package force_feedback
@file mpc_iiwa_sim_LPF.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Closed-loop MPC for force task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl in closed-loop MPC (feedback from stateLPF x=(q,v,tau))
Using PyBullet simulator for rigid-body dynamics 
Using PyBullet GUI for visualization

The goal of this script is to simulate the LPF MPC, i.e. torque feedback
where the actuation is modeled as a low pass filter . 
'''

import numpy as np  
from utils import path_utils, sim_utils, ocp_utils, pin_utils, plot_utils
import pybullet as p
import time 

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_lpf')
# Create a Pybullet simulation environment + set simu freq
simu_freq = config['simu_freq']  
dt_simu = 1./simu_freq
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
pybullet_simulator = sim_utils.init_kuka_simulator(dt=dt_simu, x0=x0)
# Get pin wrapper
robot = pybullet_simulator.pin_robot
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")


#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
# u0 = np.asarray(config['tau0'])
ug = pin_utils.get_u_grav(q0, robot)
y0 = np.concatenate([x0, ug])
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, f_c=config['f_c'])
# Solve and extract solution trajectories
xs_init = [y0 for i in range(N_h+1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
xs = np.array(ddp.xs) # optimal (q,v,u) traj
us = np.array(ddp.us) # optimal   (w)   traj
# Plot
plot_utils.plot_ddp_results_LPF(ddp, robot, id_endeff)

##################
# MPC SIMULATION #
##################
# MPC & simulation parameters
maxit = 2
T_tot = .5
plan_freq = 1000                      # MPC re-planning frequency (Hz)
ctrl_freq = 1000                      # Control - simulation - frequency (Hz)
N_tot = int(T_tot*ctrl_freq)          # Total number of control steps in the simulation (s)
N_p = int(T_tot*plan_freq)            # Total number of OCPs (replan) solved during the simulation
T_h = N_h*dt                          # Duration of the MPC horizon (s)
# Initialize data : in simulation, x=(q,v) u=tau !!!
nx = nq+nv+nu
X_mea = np.zeros((N_tot+1, nx))       # Measured states x=(q,v,tau) 
X_des = np.zeros((N_tot+1, nx))       # Desired states x=(q,v,tau)
X_pred = np.zeros((N_p, N_h+1, nx))   # MPC predictions (state) (t,q,v,tau)
U_pred = np.zeros((N_p, N_h, nu))     # MPC predictions (control) (t,w)
U_des = np.zeros((N_tot, nq))         # Unfiltered torques planned by MPC u=w

# Logs
print('                  ************************')
print('                  * MPC controller ready *') 
print('                  ************************')        
print('---------------------------------------------------------')
print('- Total simulation duration            : T_tot  = '+str(T_tot)+' s')
print('- Control frequency                    : f_ctrl = '+str(ctrl_freq)+' Hz')
print('- Replanning frequency                 : f_plan = '+str(plan_freq)+' Hz')
print('- Total # of control steps             : N_tot  = '+str(N_tot))
print('- Duration of MPC horizon              : T_ocp  = '+str(T_h)+' s')
print('- Total # of replanning knots          : N_p    = '+str(N_p))
print('- OCP integration step                 : dt     = '+str(dt)+' s')
print('---------------------------------------------------------')
print("Simulation will start...")
time.sleep(1)

# Measure initial state from simulation environment &init data
q_mea, v_mea = pybullet_simulator.get_state()
pybullet_simulator.forward_robot(q_mea, v_mea)
u_mea = pin_utils.get_u_mea(q_mea, v_mea, robot)
x0 = np.concatenate([q_mea, v_mea, u_mea]).T
print("Initial state ", str(x0))
X_mea[0, :] = x0
X_des[0, :] = x0
# Replan counter
nb_replan = 0
# SIMULATION LOOP
log_rate = 10
for i in range(N_tot): 

    if(i%log_rate==0): 
        print("  ")
        print("Sim step "+str(i)+"/"+str(N_tot))
    
    # Solve OCP if we are in a planning cycle
    if(i%int(ctrl_freq/plan_freq) == 0):
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = X_mea[i, :].T 
        # Warm-start 
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]] 
        xs_init[0] = X_mea[i, :].T
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        X_pred[nb_replan, :, :] = np.array(ddp.xs)# # (t,q,v)
        U_pred[nb_replan, :, :] = np.array(ddp.us)# # (t,u)
        # Extract 1st control and 2nd state
        u_des = U_pred[nb_replan, 0, :] 
        x_des = X_pred[nb_replan, 1, :]
        # Increment replan counter
        nb_replan += 1

    # Record the 1st control : desired torque = unfiltered torque output by DDP
    U_des[i, :] = u_des
    # u_full = u_des + ddp.K[0].dot(X_mea[i, :] - x_des)
    # Select filtered torque = integration of LPF(u_des) = x_des ? Or integration over a control step only ?
    tau_des = x_des[nq+nv:] #alpha*X_mea[i, :][-nu:] + (1-alpha)*u_des  #x_des[nq+nv:] # same as : alpha*x0[-nu:] + (1-alpha)*u_des 
    # Send control to simulation & step u
    # robot.send_joint_command(u_des + ddp.K[0].dot(X_mea[i, :] - x_des)) # with Ricatti gain
    pybullet_simulator.send_joint_command(tau_des)
    p.stepSimulation()
    # Measure new state from simulation and record data
    q_mea, v_mea = pybullet_simulator.get_state()
    pybullet_simulator.forward_robot(q_mea, v_mea)

    # Simulate torque measurement : 3 options
      # 1. Integrate LPF (no torque feedback = open-loop)
      # 2. Add elastic element in the robot
      # 3. Use PyBullet measured forces  

    # 1. Integrate LPF
    tau_mea = tau_des # new_alpha*X_mea[i, -nu:] + (1-new_alpha)*u_des   # tau_des # new_alpha =  np.sin(i*1e-3) / (1 + 2*np.pi*1e-3*500)

    # # 3. Measure contact force in PyBullet    
    # ids, forces = robot.get_force()
    #   # Express in local EE frame (minus because force env-->robot)
    # F_mea_pyb[i,:] = -robot.pin_robot.data.oMf[id_endeff].actionInverse.dot(forces[0])
    #   # FD estimate of joint accelerations
    # if(i==0):
    #   a_mea = np.zeros(nq)
    # else:
    #   a_mea = (v_mea - X_mea[i,nq:nq+nv])/1e-3
    #   # ID
    # f = StdVec_Force()
    # for j in range(robot.pin_robot.model.njoints):
    #   f.append(pin.Force.Zero())
    # f[-1].linear = F_mea_pyb[i,:3]
    # f[-1].angular = F_mea_pyb[i,3:]
    #   # Project EE force in joint space through J.T
    # tau_mea = pin.rnea(robot.pin_robot.model, robot.pin_robot.data, q_mea, v_mea, a_mea, f) 

    # Record measurements
    x_mea = np.concatenate([q_mea, v_mea, tau_mea]).T 
    X_mea[i+1, :] = x_mea                    # Measured state
    X_des[i+1, :] = x_des                    # Desired state

# GENERATE NICE PLOT OF SIMULATION
with_predictions = False
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
# Time step duration of the control loop
dt_ctrl = float(1./ctrl_freq)
# Time step duration of planning loop
dt_plan = float(1./plan_freq)
# Reshape trajs if necessary 
q_pred = X_pred[:,:,:nq]
v_pred = X_pred[:,:,nq:nq+nv]
tau_pred = X_pred[:,:,nq+nv:]
q_mea = X_mea[:,:nq]
v_mea = X_mea[:,nq:nq+nv]
tau_mea  = X_mea[:,nq+nv:]
q_des = X_des[:,:nq]
v_des = X_des[:,nq:nq+nv]
tau_des = X_des[:,nq+nv:]
p_mea = pin_utils.get_p(q_mea, robot, id_endeff)
p_des = pin_utils.get_p(q_des, robot, id_endeff) 
# Create time spans for X and U + Create figs and subplots
tspan_x = np.linspace(0, T_tot, N_tot+1)
tspan_u = np.linspace(0, T_tot-dt_ctrl, N_tot)
fig_x, ax_x = plt.subplots(nq, 3)
fig_u, ax_u = plt.subplots(nq, 1)
fig_p, ax_p = plt.subplots(3,1)
# For each joint
for i in range(nq):
    # Extract state predictions of i^th joint
    q_pred_i = q_pred[:,:,i]
    v_pred_i = v_pred[:,:,i]
    tau_pred_i = tau_pred[:,:,i]
    u_pred_i = U_pred[:,:,i]
    # print(u_pred_i[0,0])
    if(with_predictions):
        # For each planning step in the trajectory
        for j in range(N_p):
            # Receding horizon = [j,j+N_h]
            t0_horizon = j*dt_plan
            tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
            tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
            # Set up lists of (x,y) points for predicted positions and velocities
            points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
            points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
            points_tau = np.array([tspan_x_pred, tau_pred_i[j,:]]).transpose().reshape(-1,1,2)
            points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
            # Set up lists of segments
            segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
            segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
            segs_tau = np.concatenate([points_tau[:-1], points_tau[1:]], axis=1)
            segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
            # Make collections segments
            cm = plt.get_cmap('Greys_r') 
            lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
            lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
            lc_tau = LineCollection(segs_tau, cmap=cm, zorder=-1)
            lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
            lc_q.set_array(tspan_x_pred)
            lc_v.set_array(tspan_x_pred) 
            lc_tau.set_array(tspan_x_pred) 
            lc_u.set_array(tspan_u_pred)
            # Customize
            lc_q.set_linestyle('-')
            lc_v.set_linestyle('-')
            lc_tau.set_linestyle('-')
            lc_u.set_linestyle('-')
            lc_q.set_linewidth(1)
            lc_v.set_linewidth(1)
            lc_tau.set_linewidth(1)
            lc_u.set_linewidth(1)
            # Plot collections
            ax_x[i,0].add_collection(lc_q)
            ax_x[i,1].add_collection(lc_v)
            ax_x[i,2].add_collection(lc_tau)
            ax_u[i].add_collection(lc_u)
            # Scatter to highlight points
            colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
            my_colors = cm(colors)
            ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
            ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
            ax_x[i,2].scatter(tspan_x_pred, tau_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
            ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 
    

    # Joint positions
    ax_x[i,0].plot(tspan_x, q_des[:,i], 'b-', label='Desired')
    # Measured joint position (PyBullet)
    ax_x[i,0].plot(tspan_x, q_mea[:,i], 'r-', label='Measured')
    ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{i}$ (rad)')
    ax_x[i,0].grid()

    # Joint velocities
    ax_x[i,1].plot(tspan_x, v_des[:,i], 'b-', label='Desired')
    # Measured joint velocity (PyBullet)
    ax_x[i,1].plot(tspan_x, v_mea[:,i], 'r-', label='Measured')
    ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{i}$ (rad/s)')
    ax_x[i,1].grid()

    # Joint torques (filtered) = part of the state
    ax_x[i,2].plot(tspan_x, tau_des[:,i], 'b-', label='Desired')
    # Measured joint velocity (PyBullet)
    ax_x[i,2].plot(tspan_x, tau_mea[:,i], 'r-', label='Measured')
    ax_x[i,2].set(xlabel='t (s)', ylabel='$tau_{i}$ (Nm)')
    ax_x[i,2].grid()

    # Joint torques (unfiltered) = control input
    ax_u[i].plot(tspan_u, U_des[:,i], 'b-', label='Desired')
    # Total
    # ax_u[i].plot(tspan_u[0], u_mea[0,i], 'co', label='Initial')
    # print(" U0 mea plotted = "+str(u_mea[0,i]))
    # ax_u[i].plot(tspan_u, u_mea[:,i]-u_des[:,i], 'g-', label='Riccati (fb)')
    # Total torque applied
    ax_u[i].set(xlabel='t (s)', ylabel='$u_{i}$ (Nm)')
    ax_u[i].grid()

    # Legend
    handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
    fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

    handles_u, labels_u = ax_u[i].get_legend_handles_labels()
    fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

# Plot endeff
# x
ax_p[0].plot(tspan_x, p_des[:,0], 'b-', label='Desired')
ax_p[0].plot(tspan_x, p_mea[:,0], 'r-.', label='Measured')
ax_p[0].set_title('x-position')
ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
ax_p[0].grid()
# y
ax_p[1].plot(tspan_x, p_des[:,1], 'b-', label='Desired')
ax_p[1].plot(tspan_x, p_mea[:,1], 'r-.', label='Measured')
ax_p[1].set_title('y-position')
ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
ax_p[1].grid()
# z
ax_p[2].plot(tspan_x, p_des[:,2], 'b-', label='Desired')
ax_p[2].plot(tspan_x, p_mea[:,2], 'r-.', label='Measured')
ax_p[2].set_title('z-position')
ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
ax_p[2].grid()
# Add frame ref if any
p_ref = config['p_des']
ax_p[0].plot(tspan_x, [p_ref[0]]*(N_tot+1), 'ko', label='reference', alpha=0.5)
ax_p[1].plot(tspan_x, [p_ref[1]]*(N_tot+1), 'ko', label='reference', alpha=0.5)
ax_p[2].plot(tspan_x, [p_ref[2]]*(N_tot+1), 'ko', label='reference', alpha=0.5)
handles_p, labels_p = ax_p[0].get_legend_handles_labels()
fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

# Titles
fig_x.suptitle('Joint positions, velocities and (filtered) torques ', size=16)
fig_u.suptitle('Joint command (unfiltered) torques ', size=16)
fig_p.suptitle('End-effector position ', size=16)

plt.show() 

