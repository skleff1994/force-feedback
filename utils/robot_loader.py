"""
@package force_feedback
@file robot_loader.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Load pinocchio RobotWrapper for iiwa robot in PyB environment 
"""

import pybullet as p
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import numpy as np

# Load KUKA arm in PyBullet environment
def init_kuka_simulator(dt=1e3, x0=None):
    '''
    Loads KUKA LBR iiwa model in PyBullet using the 
    Pinocchio-PyBullet wrapper to simplify interactions
    '''
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    pybullet_simulator = IiwaRobot()
    env.add_robot(pybullet_simulator)
    # Initialize
    if(x0 is None):
        q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
        dq0 = np.zeros(pybullet_simulator.robot.model.nv)
    else:
        q0 = x0[:pybullet_simulator.pin_robot.model.nq]
        dq0 = x0[pybullet_simulator.pin_robot.model.nv:]
    pybullet_simulator.reset_state(q0, dq0)
    pybullet_simulator.forward_robot(q0, dq0)
    return pybullet_simulator

# # Load simulator
# def load_simulator(config, simulator='PYBULLET'):
#     # Load robot (pinocchio RobotWrapper object)
#     robot = IiwaConfig.buildRobotWrapper()
#     # Load simulator 
#     if(simulator=='PYBULLET'):
#         from bullet_utils.env import BulletEnvWithGround
#         from robot_properties_kuka.iiwaWrapper import IiwaRobot
#         env = BulletEnvWithGround()
#         simu = env.add_robot(IiwaRobot)
#     elif(simulator=='CONSIM'):
#         from consim_py.simulator import RobotSimulator
#         from robot_properties_kuka.iiwaWrapper import IiwaConfig
#         simu = RobotSimulator(config, robot)
#     return robot, simu


# ############################################
# ### ROBOT MODEL & SIMULATION ENVIRONMENT ###
# ############################################
#   # ROBOT 
#     # Create a Pybullet simulation environment
# env = BulletEnvWithGround()
#     # Create a robot instance. This initializes the simulator as well.
# robot = env.add_robot(IiwaRobot)
# id_endeff = robot.pin_robot.model.getFrameId('contact')
# nq = robot.pin_robot.model.nq 
# nv = robot.pin_robot.model.nv
#     # Initial state 
# q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
# dq0 = pin.utils.zero(nv)
#     # Reset robot to initial state in PyBullet
# robot.reset_state(q0, dq0)
#     # Update pinocchio data accordingly 
# robot.forward_robot(q0, dq0)
#     # Get initial frame placement
# M_ee = robot.pin_robot.data.oMf[id_endeff]
# print("[PyBullet] Created robot (id = "+str(robot.robotId)+")")
# print("Initial placement in WORLD frame : ")
# print(M_ee)
#   # CONTACT
#     # Set contact placement = M_ee with offset (cf. below)
# M_ct = pin.SE3.Identity()
# M_ct.rotation = M_ee.rotation 
# offset = 0.1 + 0.003499998807875214 
# M_ct.translation = M_ee.act(np.array([0., 0., offset])) 
# print("Contact placement in WORLD frame : ")
# print(M_ct)

# # Measure distance EE to contact surface using p.getContactPoints() 
# # in order to avoid PyB repulsion due to penetration 
# # Result = 0.03 + 0.003499998807875214. Problem : smaller than ball radius (changed urdf?) . 
# contactId = utils.display_contact_surface(M_ct, robot.robotId, with_collision=True)
# print("[PyBullet] Created contact plane (id = "+str(contactId)+")")
# print("[PyBullet]   >> Detect contact points : ")
# p.stepSimulation()
# contact_points = p.getContactPoints(1, 2)
# for k,i in enumerate(contact_points):
#   print("      Contact point n°"+str(k)+" : distance = "+str(i[8])+" (m) | force = "+str(i[9])+" (N)")
# # time.sleep(100)
