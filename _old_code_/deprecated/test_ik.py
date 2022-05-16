"""
@package force_feedback
@file test_ik.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
Inverse kinematics sampling test
'''

import numpy as np  
from core_mpc import ocp, path_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig
import time
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, linewidth=300)
import pinocchio as pin

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
M_des = M_ee.copy()
# p_des = np.asarray(config['p_des'])
M_des.translation = np.asarray(config['p_des'])
print("Initial EE position , velocity: \n")
print("p0_EE   = ", M_ee.translation)
print("v0_EE   = ", np.zeros(3))
print("Desired EE position : \n")
print("pdes_EE = ", M_des.translation)
print("vdes_EE = ", np.zeros(3))
# Init display
print("Initial joint configuration, joint velocity :\n")
print("q0     = ", q0)
print("v0     = ", v0)
robot.initDisplay(loadModel=True)
robot.display(q0)




TEST_IK = False
SAMPLE_IK = False
SAMPLE_IK_UNIFORM = True
DISPLAY_SAMPLING = True




if(TEST_IK):
    # Solve IK
    q1, v1, errs1 = pin_utils.IK_position(robot, q0, id_endeff, M_des.translation,
                                          DT=1e-1, IT_MAX=1000, EPS=1e-6)
    print("qdes = ", q1)
    print("vdes = ", v1)
    # Check that the solution works
    print("Reached p_EE   = ", robot.data.oMf[id_endeff].translation)
    J = pin.computeFrameJacobian(robot.model, robot.data, q1, id_endeff)
    print("Jacobian(qdes) = \n", J)
    print("J(qdes) * vdes = ", J.dot(v1))





if(SAMPLE_IK):
    # Sample several states 
    N_SAMPLES = 100
    TSK_SPACE_SAMPLES = []
    JNT_SPACE_SAMPLES = []
    # Define bounds in cartesian space to sample (p_EE,v_EE) around (p_des,0)
    p_des = np.asarray(config['p_des'])
    v_des = np.zeros(3)
    eps_p = 0.1  # +/- 10   cm
    eps_v = 0.1 # +/- 0.1 rad/s

    # Sampling uniform IK
    p_min = p_des - eps_p; p_max = p_des + eps_p
    v_min = v_des - eps_v; v_max = v_des + eps_v
    y_min = np.concatenate([p_min, v_min])
    y_max = np.concatenate([p_max, v_max])
    # Generate samples (uniform)
    for i in range(N_SAMPLES):
        # Sample
        y_EE = np.random.uniform(low=y_min, high=y_max, size=(6,))
        TSK_SPACE_SAMPLES.append( y_EE )
        # print(" Task sample  = ", y_EE, " \n")
        # IK
        q, _, _ = pin_utils.IK_position(robot, q0, id_endeff, y_EE[:3],
                                        DISPLAY=False, LOGS=False, DT=1e-1, IT_MAX=1000, EPS=1e-6)
        pin.computeJointJacobians(robot.model, robot.data, q)
        robot.framesForwardKinematics(q)
        J_q = pin.getFrameJacobian(robot.model, robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
        vq = np.linalg.pinv(J_q)[:,:3].dot(y_EE[3:]) 
        J_orthogonal = (np.eye(7) - np.linalg.pinv(J_q).dot(J_q))
        # Check that ortho space doesnt explode
        # vq_o = J_orthogonal[:,:3].dot(y_EE[3:]) 
        # print(vq_o)
        x = np.concatenate([q, vq])
        JNT_SPACE_SAMPLES.append( x )
        # print(" Joint sample = ", x, " \n")

    # Display EE target + box in which we sample p
    if(DISPLAY_SAMPLING):
        viewer = robot.viz.viewer
        gui = viewer.gui
        gui.addSphere('world/p_des', .02, [1. ,0 ,0, 1.])  
        gui.addBox('world/p_bounds',   2*eps_p, 2*eps_p, 2*eps_p,  [1., 1., 1., 0.3]) # depth(x),length(y),height(z), color
        tf_des = pin.utils.se3ToXYZQUAT(M_des)
        gui.applyConfiguration('world/p_des', tf_des)
        gui.applyConfiguration('world/p_bounds', tf_des)
        # Check samples
        for k,sample in enumerate(JNT_SPACE_SAMPLES):
            # q = sample[:nq]
            robot.display(sample[:nq])
            # Update model and display sample
            robot.framesForwardKinematics(sample[:nq])
            robot.computeJointJacobians(sample[:nq])
            M_ = robot.data.oMf[id_endeff]
            gui.addSphere('world/sample'+str(k), .01, [0. ,0 ,1., .8])  
            tf_ = pin.utils.se3ToXYZQUAT(M_)
            gui.applyConfiguration('world/sample'+str(k), tf_)
            gui.refresh()
            time.sleep(0.5)




# Sampling uniform IK + ADAPTIVE 
if(SAMPLE_IK_UNIFORM):
    N_SAMPLES = 15
    TSK_SPACE_SAMPLES = []
    JNT_SPACE_SAMPLES = []
    p_des = np.asarray(config['p_des'])
    v_des = np.zeros(3)
    eps_p = [0.05, 0.15, 0.25]   
    eps_v = [0.005, 0.01, 0.015] 
    p_min = [p_des - np.ones(3)*eps for eps in eps_p]; p_max = [p_des + np.ones(3)*eps for eps in eps_p]
    v_min = [v_des - np.ones(3)*eps for eps in eps_v]; v_max = [v_des + np.ones(3)*eps for eps in eps_v]
    y_min = [np.concatenate([p_min[i], v_min[i]]) for i in range(3)]
    y_max = [np.concatenate([p_max[i], v_max[i]]) for i in range(3)]
    print("Sampling "+str(N_SAMPLES)+" states...")
    # Generate samples (uniform)
    for box in range(3):
        for i in range(N_SAMPLES//3):
            # Task space sample
            y_EE = np.random.uniform(low=y_min[box], high=y_max[box], size=(6,))
            TSK_SPACE_SAMPLES.append( y_EE )
            # Inverse kinematics
            q, _, _ = pin_utils.IK_position(robot, q0, id_endeff, y_EE[:3],
                                            DISPLAY=False, LOGS=False, DT=1e-1, IT_MAX=1000, EPS=1e-6)
            pin.computeJointJacobians(robot.model, robot.data, q)
            robot.framesForwardKinematics(q)
            J_q = pin.getFrameJacobian(robot.model, robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
            vq = np.linalg.pinv(J_q)[:,:3].dot(y_EE[3:]) 
            x = np.concatenate([q, vq])
            JNT_SPACE_SAMPLES.append( x )

    # Display EE target + box in which we sample p
    if(DISPLAY_SAMPLING):
        viewer = robot.viz.viewer
        gui = viewer.gui
        tf_des = pin.utils.se3ToXYZQUAT(M_des)
        gui.addSphere('world/p_des', .02, [1. ,0 ,0, 1.])  
        gui.applyConfiguration('world/p_des', tf_des)
        colors = [[1., 0., 0., 0.5], [0., 1., 0., 0.3], [0., 0., 1., 0.1]]
        for i in range(3):
            gui.addBox('world/p_bounds_'+str(i),   2*eps_p[i], 2*eps_p[i], 2*eps_p[i],  [1., 1./float(i+1), 1.-1./float(i+1), 0.3]) # depth(x),length(y),height(z), color
            gui.applyConfiguration('world/p_bounds_'+str(i), tf_des)
        # Check samples
        for k,sample in enumerate(JNT_SPACE_SAMPLES):
            # q = sample[:nq]
            robot.display(sample[:nq])
            # Update model and display sample
            robot.framesForwardKinematics(sample[:nq])
            robot.computeJointJacobians(sample[:nq])
            M_ = robot.data.oMf[id_endeff]
            gui.addSphere('world/sample'+str(k), .01, [0. ,0 ,1., .8])  
            tf_ = pin.utils.se3ToXYZQUAT(M_)
            gui.applyConfiguration('world/sample'+str(k), tf_)
            gui.refresh()
            time.sleep(0.5)



# Compare sampled points in task space with FK(IK)
fig, ax = plt.subplots(3, 2, sharex='col')   
# Get FK of samples
pEE = np.array(TSK_SPACE_SAMPLES)[:,:3]
vEE = np.array(TSK_SPACE_SAMPLES)[:,3:]
q = np.array(JNT_SPACE_SAMPLES)[:,:nq]
v = np.array(JNT_SPACE_SAMPLES)[:,nv:]
pEE_FK = pin_utils.get_p_(q, robot.model, id_endeff)
vEE_FK = pin_utils.get_v_(q, v, robot.model, id_endeff) 
for i in range(3):
    # Positions
    ax[i,0].plot(np.linspace(0., 1., N_SAMPLES), pEE[:,i], 'bo', color='b', label='task space sample ')
    ax[i,0].plot(np.linspace(0., 1., N_SAMPLES), pEE_FK[:,i], 'gx', label='FK reconstruct ')
    ax[i,1].plot(np.linspace(0., 1., N_SAMPLES), vEE[:,i], 'bo',  label='task space sample ')
    ax[i,1].plot(np.linspace(0., 1., N_SAMPLES), vEE_FK[:,i], 'gx', label='FK reconstruct ')
# Legend
handles, labels = ax[i,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 16})
fig.align_ylabels()
fig.suptitle('Sampled points in task space', size=16)
plt.show()

# Plot 
# viewer.gui.crre
# viewer.gui.addLandmark('p_des', .5)

# viewer.gui.refresh()
# Check velocities 


# plt.plot(errs1)
# plt.grid()
# plt.show()

# q2, v2, errs2 = pin_utils.IK_placement(robot, q0, id_endeff, M_des, DT=1e-2, IT_MAX=1000)
# print("q2 = \n")
# print(q2)
# print("v2 = \n")
# print(v2)
# robot.display(q2)
# plt.plot(errs2)
# plt.grid()
# plt.show()