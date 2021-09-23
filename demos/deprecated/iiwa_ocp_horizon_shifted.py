"""
@package force_feedback
@file iiwa_ocp_horizon_shifted.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl
The goal of this script is to debug 
'''

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

# Create solver with custom horizon
N_h = 1000
ddp1 = ocp_utils.init_DDP(robot, config, x0, callbacks=False, 
                                        which_costs=['translation', 
                                                     'ctrlReg', 
                                                     'stateReg',
                                                     'stateLim' ],
                                        dt = None, N_h=N_h) 
ug = pin_utils.get_u_grav(q0, robot)
xs_init = [x0 for i in range(N_h+1)]
us_init = [ug  for i in range(N_h)]
print("Initial state : ", x0)
ddp1.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
ddp_data1 = data_utils.extract_ddp_data(ddp1)
print("T=", N_h, " : cost = ", ddp1.cost, " | Residual = ", ddp1.problem.runningDatas[-1].cost)

SHIFT = int(60)

# Solve same problem with shifted horizon
x0_shift = ddp_data1['xs'][SHIFT]
ddp2 = ocp_utils.init_DDP(robot, config, x0_shift, callbacks=False, 
                                                   which_costs=['translation', 
                                                               'ctrlReg', 
                                                               'stateReg',
                                                               'stateLim' ],
                                                   dt = None, N_h=N_h) 
ug = pin_utils.get_u_grav(x0_shift[:nq], robot)
xs_init = [x0_shift for i in range(N_h+1)]
us_init = [ug  for i in range(N_h)]
print("Shifted state : ", x0_shift)
ddp2.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
ddp_data2 = data_utils.extract_ddp_data(ddp2)
print("T=", N_h, " : cost = ", ddp2.cost, " | Residual = ", ddp2.problem.runningDatas[-1].cost)


# Plot results
import matplotlib.pyplot as plt
fig, ax = plot_utils.plot_ddp_results(ddp_data1, which_plots=['x','u','p'], SHOW=False)


# Add ref pos EE and shifted sol
p_des = np.asarray(config['p_des']) 
p_shift = pin_utils.get_p_(np.array(ddp_data2['xs'])[:,:nq], ddp_data2['pin_model'], ddp_data2['frame_id'])
p = pin_utils.get_p_(np.array(ddp_data1['xs'])[:,:nq], ddp_data1['pin_model'], ddp_data1['frame_id'])
# p_shift = pin_utils.get_p_([x0_shift[:nq]], ddp_data1['pin_model'], ddp_data1['frame_id'])
dt = config['dt']
fig, ax = plt.subplots(3, 1, sharex='col') 
diff = p[SHIFT:,:] - p_shift[:N_h+1-SHIFT,:]
for i in range(3):
    # Plot a posteriori integration to check IAM
    # ax['p'][i].plot(np.linspace(0., N_h*dt, N_h+1), p[:,i], linestyle='', marker='o', label='Original', alpha=1.)
    # ax['p'][i].plot(np.linspace(SHIFT*dt, (N_h+SHIFT)*dt, (N_h)+1), p_shift[:940,i], linestyle='', marker='o', label='Shifted', alpha=0.2)
    ax[i].plot(np.linspace(SHIFT*dt, N_h*dt, N_h+1-SHIFT), diff[:,i], linestyle='-', marker='o', label='Difference', alpha=0.2)
    ax[i].grid(True)
    # ax['p'][i].plot(SHIFT*dt, p_shift[0,i], 'ro')
handles_x, labels_x = ax[i].get_legend_handles_labels()
fig.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

plt.show()

