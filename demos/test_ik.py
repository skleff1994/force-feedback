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
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
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
# Add marker for desired position in Gepetto

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
# Find joint vel corresponding to some small 

# Sample several states 
N_SAMPLES = 100

TSK_SPACE_SAMPLES = []
JNT_SPACE_SAMPLES = []
p_des = np.asarray(config['p_des'])
v_des = np.zeros(3)
eps_p = 0.2 # 20cm
eps_v = 0.1 # +/- 0.1 rad/s
p_min = 
px_des = p_des[0]; py_des = p_des[1]; pz_des = p_des[2]
px_min = 0.9*np.asarray(config['p_des'])
p_max = np.asarray(config['p_des'])
q_max = 0.85*np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
v_max = 0.1*np.ones(nv) #np.array([1.4835, 1.4835, 1.7453, 1.309 , 2.2689, 2.3562, 2.3562])  #np.zeros(nv) 
x_max = np.concatenate([q_max, v_max])   
for i in range(N_SAMPLES):
    samples.append( np.random.uniform(low=-x_max, high=+x_max, size=(nx,)))

# Display robot in the right config
# robot.display(q1)

# Check velocities 


plt.plot(errs1)
plt.grid()
plt.show()

# q2, v2, errs2 = pin_utils.IK_placement(robot, q0, id_endeff, M_des, DT=1e-2, IT_MAX=1000)
# print("q2 = \n")
# print(q2)
# print("v2 = \n")
# print(v2)
# robot.display(q2)
# plt.plot(errs2)
# plt.grid()
# plt.show()