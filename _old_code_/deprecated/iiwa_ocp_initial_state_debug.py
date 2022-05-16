"""
@package force_feedback
@file iiwa_ocp_initial_state_debug.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl
The goal of this script is to solve the OCP from 
predefined initial states, plot results and animate in gepetto-viewer
'''

import crocoddyl
import numpy as np  
from core_mpc import ocp, path_utils, pin_utils, plot_utils
from robot_properties_kuka.config import IiwaConfig

np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_ocp')
# Get pin wrapper
robot = IiwaConfig.buildRobotWrapper()
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
INIT_STATES = [np.concatenate([np.array([ 2.7788, -0.7815,  1.1413,  1.5766,  2.3417, -1.7382, -2.8157]), np.zeros(nv)]), 
               np.concatenate([np.array([-1.3029,  1.2117, -2.3545, -0.2183,  2.4247, -0.8645, -1.2964]), np.zeros(nv)])]
#################
### OCP SETUP ###
#################

# Horizons to be tested
 #, 1500, 2000] #, 3000, 5000]
DDPS = []
COSTS = []
N_h = config['N_h']
dt = config['dt']
for x0 in INIT_STATES:
    q0 = x0[:nq]
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    M_ee = robot.data.oMf[id_endeff]
    print("Initial placement : \n")
    print(M_ee)

    # Create solver with custom horizon
    ddp = ocp.init_DDP(robot, config, x0, callbacks=True, 
                                            which_costs=['translation', 'ctrlReg', 'stateReg', 'velocity' ],
                                            dt = None, N_h=None) 
    # Warm-start
    ug = pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    # Solve
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    # Print VF and record
    COSTS.append(ddp.cost)
    DDPS.append(ddp)

# Plot results
fig, ax = plot_utils.plot_ddp_results(DDPS, robot, which_plots=['x','u','p'], SHOW=True)

# Visualize
import time
VISUALIZE = True
pause = 0.01 # in s
if(VISUALIZE):
    n_ddp=0
    for x0 in INIT_STATES:
        q0 = x0[:nq]
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
            robot.display(DDPS[n_ddp].xs[i][:nq])
            if(i%log_rate==0):
                print("Display config n°"+str(i))
            time.sleep(pause)
        n_ddp+=1
