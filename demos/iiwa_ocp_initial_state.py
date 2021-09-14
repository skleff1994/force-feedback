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
# Get pin wrapper
robot = IiwaConfig.buildRobotWrapper()
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
INIT_STATES = []
N_STATES = 10
state = crocoddyl.StateMultibody(robot.model)
np.random.seed(12)
# low = [-2.9671, -2.0944 ,-2.9671 ,-2.0944 ,-2.9671, -2.0944, -3.0543]
# high = [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543]
# for i in range(10):
#     q0 = np.random.uniform(low=low, high=high, size=(nq,))
#     INIT_STATES.append(np.concatenate([q0, np.zeros(nv)]))

def get_samples(nb_samples:int):
    '''
    Samples initial states x = (q,v)within conservative state range
     95% of q limits
     [-1,+1] for v
    '''
    samples = []
    q_max = 0.95*np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    v_max = 0.5*np.ones(nv)
    x_max = np.concatenate([q_max, v_max])   
    for i in range(nb_samples):
        samples.append( np.random.uniform(low=-x_max, high=+x_max, size=(nx,)))
    return samples

INIT_STATES = get_samples(10)

#################
### OCP SETUP ###
#################

# Horizons to be tested
DDPS = []
COSTS = []
N_h = config['N_h']
dt = config['dt']
for x0 in INIT_STATES:
    q0 = x0[:nq]
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)

    # Create solver with custom horizon
    ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, 
                                            which_costs=['translation', 
                                                         'ctrlReg', 
                                                         'stateReg', 
                                                         'velocity',
                                                         'stateLim'],
                                            dt = None, N_h=None) 
    # Warm-start
    ug = pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    # Solve
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    # Print VF and record
    print("q0   = ", q0)
    print("v0   = ", x0[nq:])
    print("COST = ", ddp.cost)
    COSTS.append(ddp.cost)
    DDPS.append(ddp)

# Plot results
fig, ax = plot_utils.plot_ddp_results(DDPS, robot, which_plots=['x','u','p'], SHOW=True)


# Add ELBOW
import matplotlib.pyplot as plt
id_hand = robot.model.getFrameId('A6')
frameTranslationGround = robot.data.oMf[id_endeff].act(np.zeros(3))
p_des = np.asarray(config['p_des']) 
for i in range(3):
    # Plot a posteriori integration to check IAM
    ax['p'][i].plot(np.linspace(0, N_h*config['dt'], N_h+1), [p_des[i]]*(N_h+1), 'r-.', label='Desired')
handles_x, labels_x = ax['p'][i].get_legend_handles_labels()
fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
plt.show()


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



# # import matplotlib.pyplot as plt

# # # Plot results
# # fig, ax = plot_utils.plot_ddp_results(DDPS, robot, which_plots=['x','u','p'], SHOW=True)

# # # Add ref pos EE
# # p_des = np.asarray(config['p_des']) 
# # for i in range(3):
# #     # Plot a posteriori integration to check IAM
# #     ax['p'][i].plot(np.linspace(0, N_h*config['dt'], N_h+1), [p_des[i]]*(N_h+1), 'r-.', label='Desired')
# # handles_x, labels_x = ax['p'][i].get_legend_handles_labels()
# # fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
# # plt.show()

# # # Plot VF
# # fig, ax = plt.subplots(1, 1)
# # ax.plot(HORIZONS, COSTS, 'ro', label='V.F.')
# # handles, labels = ax.get_legend_handles_labels()
# # fig.legend(handles, labels, loc='upper right', prop={'size': 16})
# # plt.show()
