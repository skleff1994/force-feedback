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

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig

np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_contact_task_ocp')
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

# print("Created contact plane (id = "+str(contactId)+")")
# print("  Contact ref. placement in WORLD frame : ")
# print(M_ct)
# print("  Detect contact points : ")
# import pybullet as p
# p.stepSimulation()
# contact_points = p.getContactPoints(1, 2)
# for k,i in enumerate(contact_points):
#   print("      Contact point n°"+str(k)+" : distance = "+str(i[8])+" (m) | force = "+str(i[9])+" (N)")
# # print("  Closest pointd between Robot and ContactPlane : ")
# # print(p.getClosestPoints(1, 2, 0.065))

# # Desired wrench in LOCAL (contact) frame
# F_des_LOCAL = pin.Force(np.array([0., 0., -50., 0., 0., 0.]))
# print("  Desired contact wrench in LOCAL frame : ")
# print(F_des_LOCAL)

#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']

ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True,
                                            WHICH_COSTS=config['WHICH_COSTS'],
                                            CONTACT=True) 

# # Half reach time (in OCP nodes)
# PHASE = 50
# for i in range(N_h-1):
#     ddp.problem.runningModels[i].differential.costs.costs['placement'].weight = ocp_utils.cost_weight_linear(i, PHASE, min_weight=.1, max_weight=10.)
#     # ddp.problem.runningModels[i].differential.costs.costs['stateReg'].weight = ocp_utils.cost_weight_normal_clamped(i, PHASE, min_weight=0.01, max_weight=10., peak=2)
#     # print(ddp.problem.runningModels[i].differential.costs.costs['stateReg'].weight)
#     ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].weight = ocp_utils.cost_weight_parabolic(i, PHASE, min_weight=0.05, max_weight=0.5)
#     ddp.problem.runningModels[i].differential.costs.costs['velocity'].weight = ocp_utils.cost_weight_parabolic(i, PHASE, min_weight=0.001, max_weight=10.)

# import time
# time.sleep(1.)
import pinocchio as pin
# f_ext = [] # pin.Force(np.asarray(config['f_des']))
# for i in range(nq+1):
#     f_ext.append(pin.Force.Zero())
# u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model)
# print("u0 = ", u0)
u0= pin_utils.get_u_grav(q0, robot)

# Solve and extract solution trajectories
xs_init = [x0 for i in range(N_h+1)]
us_init = [u0  for i in range(N_h)]

ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)


VISUALIZE = False
pause = 0.01 # in s
if(VISUALIZE):
    import time
    import pinocchio as pin
    robot.initViewer(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer; gui = viewer.gui
    
    # Display force if any
    if('force' in config['WHICH_COSTS']):
        # Display placement of contact in WORLD frame
        M_contact = M_ee.copy()
        offset = np.array([0., 0., 0.03])
        M_contact.translation = M_contact.act(offset)
        tf_contact = list(pin.SE3ToXYZQUAT(M_contact))
        if(gui.nodeExists('world/contact_point')):
            gui.deleteNode('world/contact_point', True)
            gui.deleteLandmark('world/contact_point')
        gui.addSphere('world/contact_point', .01, [1. ,0 ,0, 1.])
        gui.addLandmark('world/contact_point', .3)
        gui.applyConfiguration('world/contact_point', tf_contact)
        # Display contact force
        f_des_LOCAL = np.asarray(config['f_des'])
        M_contact_aligned = M_contact.copy()
        M_contact_aligned.rotation = M_contact_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))#.dot(M_contact_aligned.rotation) 
        tf_contact_aligned = list(pin.SE3ToXYZQUAT(M_contact_aligned))
        arrow_length = 0.02*np.linalg.norm(f_des_LOCAL)
        if(gui.nodeExists('world/ref_wrench')):
            gui.deleteNode('world/ref_wrench', True)
        gui.addArrow('world/ref_wrench', .01, arrow_length, [.5, 0., 0., 1.])
        gui.applyConfiguration('world/ref_wrench', tf_contact_aligned )
        # tf = viewer.gui.getNodeGlobalTransform('world/pinocchio/visuals/contact_0')
    # viewer.gui.addFloor('world/floor')
    viewer.gui.refresh()
    log_rate = int(N_h/10)
    f = [ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(N_h)]
    print("Visualizing...")

    # Clean arrows if any
    if(gui.nodeExists('world/force')):
        gui.deleteNode('world/force', True)
    gui.addArrow('world/force', .02, arrow_length, [.0, 0., 0.5, 0.3])

    time.sleep(1.)
    for i in range(N_h):
        # Iter log
        robot.display(ddp.xs[i][:nq])
        # Display force
        gui.resizeArrow('world/force', 0.02, 0.02*np.linalg.norm(f[i]))
        gui.applyConfiguration('world/force', tf_contact_aligned )
        viewer.gui.refresh()
        # if(i%log_rate==0):
        print("Display config n°"+str(i))
        time.sleep(pause)

#  Plot
ddp_data = data_utils.extract_ddp_data(ddp)

fig, ax = plot_utils.plot_ddp_results(ddp_data, which_plots=['all'], SHOW=False)

# # Jacobian, Inertia, NL terms
import pinocchio as pin
q = np.array(ddp.xs)[:,:nq]
v = np.array(ddp.xs)[:,nq:] 
u = np.array(ddp.us)
f = pin_utils.get_f_(q, v, u, robot.model, id_endeff, REG=0.)
import matplotlib.pyplot as plt
for i in range(3):
    ax['f'][i,0].plot(np.linspace(0,N_h*dt, N_h), f[:,i], label="Computed")
    ax['f'][i,1].plot(np.linspace(0,N_h*dt, N_h), f[:,3+i], label="Computed")
# for i in range(N_h):
#     print("Self : ", f[i,:3])
#     print("Pin  : ", ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.angular)
plt.show()