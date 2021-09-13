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
N_h = config['N_h']
dt = config['dt']

ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, 
                                            which_costs=['placement', 'ctrlReg', 'stateReg', 'velocity' ] ) 

for i in range(N_h-1):
    ddp.problem.runningModels[i].differential.costs.costs['placement'].weight = ocp_utils.cost_weight_linear(i, N_h, min_weight=1., max_weight=10.)
    # ddp.problem.runningModels[i].differential.costs.costs['placement'].weight = ocp_utils.cost_weight_tanh(i, N_h, max_weight=10., alpha=4., alpha_cut=0.1)
    # ddp.problem.runningModels[i].differential.costs.costs['stateReg'].weight = 1./ocp_utils.cost_weight_linear(i, N_h, min_weight=10., max_weight=1000.)
    ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].weight = ocp_utils.cost_weight_parabolic(i, N_h, min_weight=0.03, max_weight=0.5)
    ddp.problem.runningModels[i].differential.costs.costs['velocity'].weight = ocp_utils.cost_weight_parabolic(i, N_h, min_weight=0.001, max_weight=10.)

ug = pin_utils.get_u_grav(q0, robot)

# Solve and extract solution trajectories
xs_init = [x0 for i in range(N_h+1)]
us_init = [ug  for i in range(N_h)]

ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)


VISUALIZE = True
pause = 0.05 # in s
if(VISUALIZE):
    import time
    robot.initDisplay(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer
    # viewer.gui.addFloor('world/floor')
    # viewer.gui.refresh()
    log_rate = int(N_h/10)
    print("Visualizing...")
    time.sleep(1.)
    for i in range(N_h):
        # Iter log
        viewer.gui.refresh()
        robot.display(ddp.xs[i][:nq])
        if(i%log_rate==0):
            print("Display config n°"+str(i))
        time.sleep(pause)


#  Plot
fig, ax = plot_utils.plot_ddp_results(ddp, robot, which_plots=['x','u','p'], SHOW=False)

p_des = np.asarray(config['p_des']) 
for i in range(3):
    # Plot a posteriori integration to check IAM
    ax['p'][i].plot(np.linspace(0, N_h*dt, N_h+1), [p_des[i]]*(N_h+1), 'r-.', label='Desired')
import matplotlib.pyplot as plt
handles_x, labels_x = ax['p'][i].get_legend_handles_labels()
fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
plt.show()
