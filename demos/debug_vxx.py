import numpy as np
import crocoddyl
from utils import path_utils, data_utils, ocp_utils, plot_utils
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin

# Read config file
config = path_utils.load_config_file('static_reaching_task3')
simu_freq = 20e3
# Robot pin wrapper
robot = RobotWrapper.BuildFromURDF(path_utils.get_urdf_path('iiwa'), path_utils.get_mesh_dir())
nq, nv = robot.model.nq, robot.model.nv
nu = nq
q0 = np.asarray(config['q0'])
dq0 = np.asarray(config['dq0'])
robot.forwardKinematics(q0, dq0)
robot.framesForwardKinematics(q0)
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]

# Load data 
d = data_utils.load_data('/home/skleff/force-feedback/data/DATASET3_change_task_increase_freq/10000/tracking=False_10000Hz__exp_9.npz')
plan_freq = 10e3
# Change costs as in recorded simulation
config['frameWeight'] = 51200
config['xRegWeight'] = 1.953125e-5
config['uRegWeight'] = 3.90625e-5
# Select a state at right times
ta = 0.5 
tb = 1.0
k_simu_a = int(simu_freq*ta)
k_simu_b = int(simu_freq*tb)
k_plan_a = int(plan_freq*ta)
k_plan_b = int(plan_freq*tb)
x0a = np.concatenate([d['q_mea'][k_simu_a, :], d['v_mea'][k_simu_a, :]])
x0b = np.concatenate([d['q_mea'][k_simu_b, :], d['v_mea'][k_simu_b, :]])
lambda_a = d['Vxx_eigval'][k_plan_a, 0]
lambda_b = d['Vxx_eigval'][k_plan_b, 0]
# Check VP values
print(lambda_a)
print(lambda_b)
# Creating the DDP solver 
ddp_a = ocp_utils.init_DDP(robot, config, x0a)
ddp_b = ocp_utils.init_DDP(robot, config, x0b)
# solve for each point
ddp_a.setCallbacks([crocoddyl.CallbackLogger(),
                   crocoddyl.CallbackVerbose()])
ddp_b.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose()])
ddp_a.solve(ddp_a.xs, ddp_a.us, maxiter=10, isFeasible=False)
ddp_b.solve(ddp_b.xs, ddp_b.us, maxiter=10, isFeasible=False)

import time
import matplotlib.pyplot as plt
# plt.ion()
# fig, ax = plot_utils.plot_ddp_endeff(ddp_a, robot, id_endeff) # label='x0_a')
# plot_utils.plot_ddp_endeff(ddp_b, robot, id_endeff, fig=fig, ax=ax)
plot_utils.plot_ddp_results([ddp_a, ddp_b], robot, id_endeff)
# plt.show()
# plt.close('all')
# plot_utils.plot_ddp_control(ddp_a)
plt.close('all')
# plot_utils.plot_ddps(ddp_a, robot)

# #################################
# ### EXTRACT SOLUTION AND PLOT ###
# #################################
# print("Extracting solution...")
# # Extract solution trajectories
# qa = np.empty((d['N_h']+1, nq))
# va = np.empty((d['N_h']+1, nv))
# qb = np.empty((d['N_h']+1, nq))
# vb = np.empty((d['N_h']+1, nv))
# p_eea = np.empty((d['N_h']+1, 3))
# p_eeb = np.empty((d['N_h']+1, 3))

# for i in range(d['N_h']+1):
#     qa[i,:] = ddp_a.xs[i][:nq].T
#     va[i,:] = ddp_a.xs[i][nv:].T
#     qb[i,:] = ddp_b.xs[i][:nq].T
#     vb[i,:] = ddp_b.xs[i][nv:].T
#     pin.forwardKinematics(robot.model, robot.data, qa[i])
#     pin.updateFramePlacements(robot.model, robot.data)
#     p_eea[i,:] = robot.data.oMf[id_endeff].translation.T
#     pin.forwardKinematics(robot.model, robot.data, qb[i])
#     pin.updateFramePlacements(robot.model, robot.data)
#     p_eeb[i,:] = robot.data.oMf[id_endeff].translation.T
# ua = np.empty((d['N_h'], nu))
# ub = np.empty((d['N_h'], nu))

# for i in range(d['N_h']):
#     ua[i,:] = ddp_a.us[i].T
#     ub[i,:] = ddp_b.us[i].T

# import matplotlib.pyplot as plt #; plt.ion()
# # Create time spans for X and U + figs and subplots
# tspan_x = np.linspace(0, d['N_h']*config['dt'], d['N_h']+1)
# tspan_u = np.linspace(0, d['N_h']*config['dt'], d['N_h'])
# fig_x, ax_x = plt.subplots(nq, 2)
# fig_u, ax_u = plt.subplots(nq, 1)
# fig_p, ax_p = plt.subplots(3,1)
# # Plot joints pos, vel , acc, torques
# for i in range(nq):
#     # Positions
#     ax_x[i,0].plot(tspan_x, qa[:,i], 'b', label='BEFORE')
#     ax_x[i,0].plot(tspan_x, qb[:,i], 'r', label='AFTER')
#     ax_x[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
#     ax_x[i,0].grid()
#     # Velocities
#     ax_x[i,1].plot(tspan_x, va[:,i], 'b', label='BEFORE')
#     ax_x[i,1].plot(tspan_x, vb[:,i], 'r', label='AFTER')
#     ax_x[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
#     ax_x[i,1].grid()
#     # Torques
#     ax_u[i].plot(tspan_u, ua[:,i], 'b', label='BEFORE') # feedforward term
#     ax_u[i].plot(tspan_u, ub[:,i], 'r', label='AFTER') # feedforward term
#     ax_u[i].set_ylabel(ylabel='$u_%d$'%i, fontsize=16)
#     ax_u[i].grid()
#     # Remove xticks labels for clarity 
#     if(i != nq-1):
#         for j in range(2):
#             ax_x[i,j].set_xticklabels([])
#         ax_u[i].set_xticklabels([])
#     # Set xlabel on bottom plot
#     if(i == nq-1):
#         for j in range(2):
#             ax_x[i,j].set_xlabel('t (s)', fontsize=16)
#         ax_u[i].set_xlabel('t (s)', fontsize=16)
#     # Legend
#     handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
#     fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
#     handles_u, labels_u = ax_u[i].get_legend_handles_labels()
#     fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
# # Plot EE
# ylabels_p = ['Px', 'Py', 'Pz']
# for i in range(3):
#     ax_p[i].plot(tspan_x, p_eea[:,i], 'b', label='BEFORE')
#     ax_p[i].plot(tspan_x, p_eeb[:,i], 'r', label='AFTER')
#     ax_p[i].set_ylabel(ylabel=ylabels_p[i], fontsize=16)
#     ax_p[i].grid()
#     handles_p, labels_p = ax_p[i].get_legend_handles_labels()
#     fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
# ax_p[-1].set_xlabel('t (s)', fontsize=16)
# # Align labels + set titles
# fig_x.align_ylabels()
# fig_u.align_ylabels()
# fig_p.align_ylabels()
# fig_x.suptitle('Joint trajectories', size=16)
# fig_u.suptitle('Joint torques', size=16)
# fig_p.suptitle('End-effector trajectory', size=16)
# plt.show()
