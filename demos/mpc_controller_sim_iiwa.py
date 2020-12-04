"""
@package ddp_iiwa
@file ddp_iiwa/mpc_controller_sim.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Demo file for the DDP-based MPC control of the KUKA arm: reach EE pose
"""

import numpy as np  
import pinocchio as pin
from py_robot_properties_iiwa.robot import IiwaRobot
from py_ddp_iiwa.mpc_controller import MPCController
from py_ddp_iiwa.ddp_planner import DDPPlanner


############################################
### ROBOT MODEL & SIMULATION ENVIRONMENT ###
############################################
# Create a robot instance. This initializes the simulator as well.
robot = IiwaRobot()
id_endeff = robot.pin_robot.model.getFrameId('contact')
nq = robot.pin_robot.model.nq 
nv = robot.pin_robot.model.nv
# Hard-coded initial state of StartDGM application in KUKA sunrise control panel
q0 = np.array([3.0020535764625e-05, 0.3491614109215945, -6.5913231790875e-05, -0.8727514773831355, -7.1713659020325e-05, -5.794373118614063e-05, 0.00010090719233645313])
dq0 = pin.utils.zero(nv)
# Reset PyBullet to that state
robot.reset_state(q0, dq0)
# Update pinocchio model with forward kinematics and get frame initial placemnent + desired frame placement
pin.forwardKinematics(robot.pin_robot.model, robot.pin_robot.data, q0)
pin.updateFramePlacements(robot.pin_robot.model, robot.pin_robot.data)
M0 = robot.pin_robot.data.oMf[id_endeff]
# p_target = np.array([-0.7, -0.6, 0.5]) #np.array([0.5, 0.5, 0.5]) #M0.translation #np.array([0.522418, 0.0448216, 0.824988])
p_target = np.array([-0.5, -0.5, 0.5])

# # Generate a random quaternion ( see here http://planning.cs.uiuc.edu/node198.html)
# from numpy.random import rand
# u1, u2, u3 = rand(3)
# (x,y,z,w) = (np.sqrt(1 - u1)*np.sin(2*np.pi*u1), 
#         np.sqrt(1 - u1)*np.cos(2*np.pi*u2), 
#         np.sqrt(u1)*np.sin(2*np.pi*u3), 
#         np.sqrt(u1)*np.cos(2*np.pi*u3))
# quat = pin.Quaternion(x,y,z,w)
# # Convert into rotation matrix
# R_target = quat.toRotationMatrix()
# R_target =  np.matrix([[0.299515, -0.423712,   1.12613],
#                        [-1.18373,-0.388225,   0.71632],
#                        [-0.215603 , -1.31712 , -0.74839]])
# M_target = pin.SE3(R_target, p_target)
M_target = pin.SE3(np.eye(3), p_target)

###################
### DDP PLANNER ###
###################
# Integration step for DDP (s) 
dt = 5e-2               
# Number of knots in the MPC horizon 
N = 20        
# Planner
ddp_planner = DDPPlanner(robot.pin_robot, dt, N)
# Set costs weights
running_costs = [10, 1e-3, 1e-3, 10.]                 # endeff, xreg, ureg, xlim
terminal_costs = [1e6, 1e-3, 10.]                        # endeff, xreg, xlim
state_weights = np.array([1.]*nq + [1.]*nv)        # size nq + nv
state_weights_term = np.array([1.]*nq + [1.]*nv)  # size nq + nv
frame_weights = np.array([1.] * 3 + [.1] * 3)     # size 6 (3 pos + 3 rot)

# # # # for 2e-2 and 20 knots ok
# # # running_costs = [1e-3, 3e-3, 2e-4 , 100]               # endeff, xreg, ureg, xlim
# # # terminal_costs = [4e2, 3e-2, 100]                   # endeff, xreg, xlim
# # # state_weights = np.array([1.]*nq + [10.]*nv)        # size nq + nv
# # # state_weights_term = np.array([.0]*nq + [10.]*nv)  # size nq + nv
# # # frame_weights = np.array([1.] * 3 + [1.] * 3)      # size 6 (3 pos + 3 r

# running_costs = [0., 1., 2e-4 , 100]               # endeff, xreg, ureg, xlim
# terminal_costs = [0., 1., 100]                   # endeff, xreg, xlim
# state_weights = np.array([1.]*nq + [10.]*nv)        # size nq + nv
# state_weights_term = np.array([.0]*nq + [10.]*nv)  # size nq + nv
# frame_weights = np.array([1.] * 3 + [1.] * 3)      # size 6 (3 pos + 3 r

# running_costs = [0., 1., 1e-1, 100.]               # endeff, xreg, ureg, xlim
# terminal_costs = [0., 1., 100.]                   # endeff, xreg, xlim
# state_weights = np.array([1.]*nq + [10.]*nv)        # size nq + nv
# state_weights_term = np.array([1.]*nq + [10.]*nv)  # size nq + nv
# frame_weights = np.array([10.] * 3 + [1.] * 3)      # size 6 (3 pos + 3 r


# Initialize OCP
x0 = np.concatenate([q0, dq0])
ddp_planner.init_ocp(x0, M_target, running_costs, terminal_costs, state_weights, state_weights_term, frame_weights, interpolation=False)
# Solve OCP
ddp_planner.solve_ocp(x0, max_iter=1, callback=False)

# Plot inital plan
# ddp_planner.plot()

# PID gains 
kp = np.diag(np.array([42., 26., 20., 8., 25., 5., 5.]))*1.5 #  #
kd = np.diag(np.array([28., 19., 18., 6., 15., 4., 4.]))*.6 # 28., 19., 18., 6., 15., 4., 4. # np.diag(np.array([5, 5, 2, 2, 2, 2, 2]))*1.
# Create MPC controller
sim_freq = 1000      # PyBullet freq (hardcoded in iiwawrapper, 1kHz)
mpc_freq = 100       # replanning frequency (Hz)
T_total = 1.     # Total duration of the simulation (s)
mp_controller = MPCController(robot, ddp_planner, [kp, kd], sim_freq, mpc_freq, T_total)

# Run simulation
mp_controller.run_impedance(with_gravity_compensation=True)
# mp_controller.run_vanilla()
# mp_controller.run_hybrid(with_gravity_compensation=False)

# # Plots
mp_controller.plot(with_predictions=False)

# # Dump in data file for dg_demos impedance_offline.py
# # Reshape trajs 
# Xs = mp_controller.X_des 
# Us = mp_controller.U_des
# qs = Xs[:,:nq]
# vs = Xs[:,nv:]
# # Add constant desired state at the end of the trajectory to avoid reader bug (loops back to start)
# N_end = 10000
# qs_tot = np.vstack([qs, np.array([qs[-1,:]]*N_end)])
# vs_tot = np.vstack([vs, np.array([vs[-1,:]]*N_end)])
# # Same for controls, using gravity compensation
# qs_end = qs_tot[-1,:]
# u_grav = pin.rnea(robot.pin_robot.model, robot.pin_robot.data, qs_end, np.zeros((nv,1)), np.zeros((nq,1)))
# us_tot = np.vstack([Us, np.array([u_grav]*N_end)])
# # Dump to data files 
# np.savetxt("/tmp/iiwa_ddp_pos_traj.dat", qs_tot, delimiter=" ")
# np.savetxt("/tmp/iiwa_ddp_vel_traj.dat", vs_tot, delimiter=" ")
# np.savetxt("/tmp/iiwa_ddp_tau_traj.dat", us_tot, delimiter=" ")
