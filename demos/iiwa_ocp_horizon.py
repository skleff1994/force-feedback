"""
@package force_feedback
@file iiwa_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import crocoddyl
import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig

np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_ocp')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper
robot = IiwaConfig.buildRobotWrapper()
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
# Update robot model with initial state
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)

#################
### OCP SETUP ###
#################

# Horizons to be tested
# HORIZONS = [200, 250, 300, 350, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
HORIZONS = [200]#, 250, 300, 350, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
DDP_DATA = []
COSTS = []
RESIDUALS = []
WARM_START = False # warm start each OCP with previous OCP solution (duplicate last state)
i = 0
for N_h in HORIZONS:
    # Create solver with custom horizon
    ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=False, 
                                            which_costs=['translation', 
                                                         'ctrlReg', 
                                                         'stateReg',
                                                         'stateLim' ],
                                            dt = None, N_h=N_h) 
    # Warm-start
    if(WARM_START==True and i>0):
        xs_init = list(DDP_DATA[i-1]['xs'])+[DDP_DATA[i-1]['xs'][-1]]*(N_h-HORIZONS[i-1])
        us_init = list(DDP_DATA[i-1]['us'])+[DDP_DATA[i-1]['us'][-1]]*(N_h-HORIZONS[i-1])
    else:
        ug = pin_utils.get_u_grav(q0, robot)
        xs_init = [x0 for i in range(N_h+1)]
        us_init = [ug  for i in range(N_h)]
    # Solve
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    # Print VF and record
    print("T=", N_h, " : cost = ", ddp.cost, " | Residual = ", ddp.problem.runningDatas[-1].cost)
    COSTS.append(ddp.cost)
    RESIDUALS.append(ddp.problem.runningDatas[-1].cost)
    ddp_data = data_utils.extract_ddp_data(ddp)
    DDP_DATA.append(ddp_data)
    i+=1


import matplotlib.pyplot as plt

# Plot results
fig, ax = plot_utils.plot_ddp_results(DDP_DATA, which_plots=['x','u','p'], SHOW=False)

# Add ref pos EE
p_des = np.asarray(config['p_des']) 
for i in range(3):
    # Plot a posteriori integration to check IAM
    ax['p'][i,0].plot(np.linspace(0, N_h*config['dt'], N_h+1), [p_des[i]]*(N_h+1), 'r-.', label='Desired')
    ax['p'][i,1].plot(np.linspace(0, N_h*config['dt'], N_h+1), [0.]*(N_h+1), 'r-.', label='Desired')
handles_x, labels_x = ax['p'][i,0].get_legend_handles_labels()
fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

# Plot inverse kinematics to check
# x = np.array(DDP_DATA[0]['xs'])
# q_IK = np.zeros((N_h+1, nq))
# v_IK = np.zeros((N_h+1, nv))
# print("Compute IK to check ")
# for i in range(N_h+1):
#     q, v, _ = pin_utils.IK_position(robot, q0, id_endeff, p_des, DT=1e-1, IT_MAX=1000, EPS=1e-6)
#     q_IK[i,:] = q
#     v_IK[i,:] = v
# q_IK, v_IK, _ = pin_utils.IK_position(robot, q0, id_endeff, p_des, DT=1e-1, IT_MAX=1000, EPS=1e-6)
# for i in range(nq):
#     ax['x'][i,0].plot(N_h*config['dt'], q_IK[i], 'go', label='IK', alpha=1.)
#     ax['x'][i,1].plot(N_h*config['dt'], v_IK[i], 'go', label='IK', alpha=1.)

# Plot VF
fig['vf'], ax['vf'] = plt.subplots(1, 1)
ax['vf'].plot(HORIZONS, COSTS, 'ro', label='integral cost')
ax['vf'].set_yscale('log')
handles, labels = ax['vf'].get_legend_handles_labels()
fig['vf'].legend(handles, labels, loc='upper right', prop={'size': 16})

# Plot residuals
fig['res'], ax['res'] = plt.subplots(1, 1)
ax['res'].plot(HORIZONS, RESIDUALS, 'ro', label='end residual')
ax['res'].set_yscale('log')
handles, labels = ax['res'].get_legend_handles_labels()
fig['res'].legend(handles, labels, loc='upper right', prop={'size': 16})
plt.grid()
plt.show()