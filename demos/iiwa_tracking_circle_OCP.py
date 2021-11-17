"""
@package force_feedback
@file iiwa_tracking_cirlce_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for tracking EE circle with the KUKA iiwa 
"""

'''
The robot is tasked with tracking a circle EE trajectory
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig

np.set_printoptions(precision=4, linewidth=180)


import time
import matplotlib.pyplot as plt



# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
print("--------------------------------------")
print("              LOAD MODEL              ")
print("--------------------------------------")
# Read config file
config = path_utils.load_config_file('iiwa_tracking_circle_OCP')
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



# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
print("--------------------------------------")
print("              INIT OCP                ")
print("--------------------------------------")
N_h = config['N_h']
dt = config['dt']
# Setup Croco OCP and create solver
ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, 
                                            WHICH_COSTS=config['WHICH_COSTS']) 
# print(pin_utils.IK_position(robot, q0, id_endeff, M_ee.translation, DT=0.01, IT_MAX=1000, sleep=0., LOGS=False))
# time.sleep(100)
# Create circle 
EE_ref = ocp_utils.circle_trajectory_WORLD(M_ee.copy(), dt=0.01, radius=.1, omega=3.)
# EE_ref_LOCAL = np.zeros(EE_ref.shape)
# for i in range(EE_ref.shape[0]):
#     EE_ref_LOCAL[i,:] = M_ee.actInv(EE_ref[i,:])
# # plt.plot(EE_ref[:,0], EE_ref[:,1])
# plt.plot(EE_ref_LOCAL[:,0], EE_ref_LOCAL[:,1])
# plt.show()
models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
# xs_init = [] 
# us_init = []
# q_ws = q0

# Reference for EE translation = circle trajectory
for k,m in enumerate(models):
    if(k<EE_ref.shape[0]):
    
        m.differential.costs.costs['translation'].cost.residual.reference = EE_ref[k]
    else:
        m.differential.costs.costs['translation'].cost.residual.reference = EE_ref[-1]
    # # Warm start state = IK of circle trajectory
    # M_des = M_ee.copy()
    # M_des.translation = EE_ref[k]
    # q_ws, v_ws, eps = pin_utils.IK_placement(robot, q0, id_endeff, M_des, DT=1e-2, IT_MAX=100)
    # print(q_ws, v_ws)
    # # print(eps[-1])
    # xs_init.append(np.concatenate([q_ws, v_ws]))
    # # Warm start control = gravity compensation of xs_init 
    # if(k<N_h):
    #     us_init.append(pin_utils.get_u_grav(q_ws, robot.model))
ug  = pin_utils.get_u_grav(q0, robot.model)
xs_init = [x0 for i in range(config['N_h']+1)]
us_init = [ug for i in range(config['N_h'])]
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

# p = pin_utils.get_p_(np.array(xs_init)[:,:nq], robot.model, id_endeff)
# import matplotlib.pyplot as plt
# plt.plot(p[:,0], p[:,1])
# plt.show()

#  Plot
PLOT = True
if(PLOT):
    ddp_data = data_utils.extract_ddp_data(ddp)
    fig, ax = plot_utils.plot_ddp_results(ddp_data, which_plots=['p'], markers=['.'], colors=['b'], SHOW=True)


# Visualize
VISUALIZE = True
pause = 0.05 # in s
if(VISUALIZE):
    import time
    import pinocchio as pin
    robot.initDisplay(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer
    log_rate = int(N_h/10)
    draw_rate = int(N_h/100)
    print("Visualizing...")
    time.sleep(1)
    # Clean previous node if any
    for i in range(N_h):
        if(viewer.gui.nodeExists('world/EE_'+str(i))):
            viewer.gui.deleteNode('world/EE_'+str(i), True)
    if(viewer.gui.nodeExists('world/ee')):
        viewer.gui.deleteNode('world/ee', True)
    viewer.gui.addSphere('world/ee', .03, [0. ,1. ,0, 1.])
    viewer.gui.addLandmark('world/ee', .5)
    viewer.gui.applyConfiguration('world/ee', list(pin.SE3ToXYZQUAT(M_ee.copy())))
    viewer.gui.refresh()
    for i in range(N_h):
        # Iter log
        viewer.gui.refresh()
        robot.display(ddp.xs[i][:nq])
        if(i%draw_rate==0):
            m_ee = M_ee.copy()
            m_ee.translation = models[i].differential.costs.costs['translation'].cost.residual.reference
            tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
            viewer.gui.addSphere('world/EE_'+str(i), .01, [1. ,0 ,0, 1.])
            viewer.gui.applyConfiguration('world/EE_'+str(i), tf_ee)
        if(i%log_rate==0):
            print("Display config n°"+str(i))
        time.sleep(pause)
    # time.sleep(5)
    # for i in range(N_h):
    #     if(viewer.gui.nodeExists('world/EE_'+str(i))):
    #         viewer.gui.deleteNode('world/EE_'+str(i), True)
