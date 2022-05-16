"""
@package force_feedback
@file iiwa_raisim_test.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
Just to test out the Pinocchio-Rai wrapper for Iiwa robot in raisim simulator
'''


import numpy as np
import time
from core_mpc import raisim_utils

# Load Kuka config from URDF
urdf_path = "/home/skleff/robot_properties_kuka_RAISIM/iiwa_test.urdf"
mesh_path = "/home/skleff/robot_properties_kuka_RAISIM"
iiwa_config = raisim_utils.IiwaMinimalConfig(urdf_path, mesh_path)

# Load Raisim environment
LICENSE_PATH = '/home/skleff/.raisim/activation.raisim'
env = raisim_utils.RaiEnv(LICENSE_PATH)
robot = env.add_robot(iiwa_config, init_config=None)
env.launch_server()
# Raisim parameters for forward prediction
env.world.setTimeStep(1e-3)
q,v = np.zeros(7), np.zeros(7)
robot.reset_state(q,v)
print(robot.get_state())
robot.forward_robot(q,v)
print(robot.get_state())
env.step()
print(robot.get_state())
time.sleep(10)
env.server.killServer()
# for i in range(10):
#     robot.send_joint_command(tau)
#     q,v, = robot.get_state()
#     robot.forward_robot(q,v)
#     env.step()


# Add stuff in environments

# iiwa = server.addVisualArticulatedSystem("v_iiwa", ) #path_utils.get_urdf_path('iiwa'))
# x0 = np.zeros(14)
# iiwa.setGeneralizedCoordinate(x0)
# iiwa.setColor(0.5, 0.0, 0.0, 0.5)

# counter = 0

# for i in range(500000):
#     counter = counter + 1
#     x0[2]+=.1
#     iiwa.setGeneralizedCoordinate(x0)
#     # visBox.setColor(1, 1, (counter % 255 + 1) / 256., 1)
#     # visSphere.setColor(1, (counter % 255 + 1) / 256., 1, 1)
#     # lines.setColor(1 - (counter % 255 + 1) / 256., 1, (counter % 255 + 1) / 256., 1)
#     # visBox.setBoxSize((counter % 255 + 1) / 256. + 0.01, 1, 1)
#     time.sleep(world.getTimeStep())

# server.killServer()
