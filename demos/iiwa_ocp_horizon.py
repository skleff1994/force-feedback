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
from utils import path_utils, ocp_utils, pin_utils, plot_utils
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
M_ee = robot.data.oMf[id_endeff]
print("Initial placement : \n")
print(M_ee)

#################
### OCP SETUP ###
#################

# Horizons to be tested
HORIZONS = [800, 1000] #, 1500, 2000] #, 3000, 5000]
DDPS = []
COSTS = []
for N_h in HORIZONS:
    # Create solver with custom horizon
    ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, 
                                            which_costs=['placement', 'ctrlReg', 'stateReg', 'velocity' ],
                                            dt = None, N_h=N_h) 
    # Warm-start
    ug = pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    # Solve
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    # Print VF and record
    COSTS.append(ddp.cost)
    DDPS.append(ddp)

#  Plot
fig, ax = plot_utils.plot_ddp_results(DDPS, robot, which_plots=['x','u','p'], SHOW=True)
# plot_utils.plot_ddp_results(DDPS[1], robot, which_plots=['x','u','p'], SHOW=False)

# p_des = np.asarray(config['p_des']) 
# for i in range(3):
#     # Plot a posteriori integration to check IAM
#     ax['p'][i].plot(np.linspace(0, N_h*config['dt'], N_h+1), [p_des[i]]*(N_h+1), 'r-.', label='Desired')
# import matplotlib.pyplot as plt
# handles_x, labels_x = ax['p'][i].get_legend_handles_labels()
# fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
# fig, ax = plt.subplots(1, 1)
# ax.plot(HORIZONS, COSTS, 'ro', label='V.F.')
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', prop={'size': 16})
# plt.show()
