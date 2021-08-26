import os
import numpy as np
import raisimpy as raisim
import math
import time
from utils import path_utils

raisim.World.setLicenseFile("/home/skleff/.raisim/activation.raisim")
world = raisim.World()
ground = world.addGround()

# launch raisim server
server = raisim.RaisimServer(world)
server.launchServer(8080)

# visSphere = server.addVisualSphere("v_sphere", 1, 1, 1, 1, 1)
# visBox = server.addVisualBox("v_box", 1, 1, 1, 1, 1, 1, 1)
# visCylinder = server.addVisualCylinder("v_cylinder", 1, 1, 0, 1, 0, 1)
# visCapsule = server.addVisualCapsule("v_capsule", 1, 0.5, 0, 0, 1, 1)
iiwa = server.addVisualArticulatedSystem("v_iiwa", "/home/skleff/robot_properties_kuka_1/urdf/iiwa.urdf") #path_utils.get_urdf_path('iiwa'))
x0 = np.zeros(14)
iiwa.setGeneralizedCoordinate(x0)
iiwa.setColor(0.5, 0.0, 0.0, 0.5)

# visSphere.setPosition(np.array([2, 0, 0]))
# visCylinder.setPosition(np.array([0, 2, 0]))
# visCapsule.setPosition(np.array([2, 2, 0]))

# lines = server.addVisualPolyLine("lines")
# lines.setColor(0, 0, 1, 1)
# for i in range(0, 100):
#     lines.addPoint(np.array([math.sin(i * 0.1), math.cos(i * 0.1), i * 0.01]))

counter = 0

for i in range(500000):
    counter = counter + 1
    x0[2]+=.1
    iiwa.setGeneralizedCoordinate(x0)
    # visBox.setColor(1, 1, (counter % 255 + 1) / 256., 1)
    # visSphere.setColor(1, (counter % 255 + 1) / 256., 1, 1)
    # lines.setColor(1 - (counter % 255 + 1) / 256., 1, (counter % 255 + 1) / 256., 1)
    # visBox.setBoxSize((counter % 255 + 1) / 256. + 0.01, 1, 1)
    time.sleep(world.getTimeStep())

server.killServer()