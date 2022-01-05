"""
@package force_feedback
@file talos_reaching_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with TALOS robot 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl
The goal of this script is to setup the OCP (a.k.a. play with weights)
'''

import sys
sys.path.append('.')

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
import example_robot_data

np.set_printoptions(precision=4, linewidth=180)

import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('talos_reaching_OCP')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper
robot = example_robot_data.load('talos_arm')
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('gripper_left_joint')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
# Update robot model with initial state
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
M_ee = robot.data.oMf[id_endeff]



# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
N_h = config['N_h']
dt = config['dt']
# Setup Croco OCP and create solver
ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, 
                                            WHICH_COSTS=config['WHICH_COSTS']) 
# Warmstart and solve
ug = pin_utils.get_u_grav(q0, robot.model)
xs_init = [x0 for i in range(N_h+1)]
us_init = [ug  for i in range(N_h)]
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)


#  Plot
PLOT = True
if(PLOT):
    ddp_data = data_utils.extract_ddp_data(ddp, frame_of_interest='gripper_left_joint')
    fig, ax = plot_utils.plot_ddp_results(ddp_data, which_plots=['all'], markers=['.'], colors=['b'], SHOW=True)

VISUALIZE = True
pause = 0.01 # in s
if(VISUALIZE):
    import time
    robot.initDisplay(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer
    # viewer.gui.addFloor('world/floor')
    # viewer.gui.refresh()
    log_rate = int(N_h/10)
    logger.info("Visualizing...")
    time.sleep(1.)
    for i in range(N_h):
        # Iter log
        viewer.gui.refresh()
        robot.display(ddp.xs[i][:nq])
        if(i%log_rate==0):
            logger.info("Display config n°"+str(i))
        time.sleep(pause)


