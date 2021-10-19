import os

from robot_properties_kuka.config import IiwaConfig
import numpy as np
import raisimpy as raisim
import math
import time
from utils import raisim_utils

# Load Kuka config from URDF
urdf_path = "/home/skleff/robot_properties_kuka_RAISIM/iiwa.urdf"
mesh_path = "/home/skleff/robot_properties_kuka_RAISIM"
iiwa_config = raisim_utils.IiwaMinimalConfig(urdf_path, mesh_path)

# Load Raisim environment
LICENSE_PATH = '/home/skleff/.raisim/activation.raisim'
env = raisim_utils.RaiEnv(LICENSE_PATH)
robot = env.add_robot(iiwa_config, urdf_path, init_config=None, vis_ghost=True)
env.step()
env.launch_server()

tau = np.zeros(7)

#Raisim parameters for forward prediction
sim_dt = 0.001
world = raisim.World()
world.setTimeStep(sim_dt)

while(1):
    robot.send_joint_command(tau)
    q,v, = robot.get_state()
    robot.forward_robot(q,v)
    env.step()


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
