import numpy as np  
import pinocchio as pin
from py_robot_properties_iiwa.robot import IiwaRobot
from py_ddp_iiwa.mpc_controller import MPCController
from py_ddp_iiwa.ddp_planner import DDPPlanner


############################################
### ROBOT MODEL & SIMULATION ENVIRONMENT ###
############################################
# Create a robot instance. This initializes the simulator as well.
robot = IiwaRobot()
id_endeff = robot.pin_robot.model.getFrameId('contact')
nq = robot.pin_robot.model.nq 
nv = robot.pin_robot.model.nv
# Hard-coded initial state of StartDGM application in KUKA sunrise control panel
q0 = np.array([3.0020535764625e-05, 0.3491614109215945, -6.5913231790875e-05, -0.8727514773831355, -7.1713659020325e-05, -5.794373118614063e-05, 0.00010090719233645313])
dq0 = pin.utils.zero(nv)
# Reset PyBullet to that state
robot.reset_state(q0, dq0)
# Update pinocchio model with forward kinematics and get frame initial placemnent + desired frame placement
pin.forwardKinematics(robot.pin_robot.model, robot.pin_robot.data, q0)
pin.updateFramePlacements(robot.pin_robot.model, robot.pin_robot.data)
M0 = robot.pin_robot.data.oMf[id_endeff]
# p_target = np.array([-0.7, -0.6, 0.5]) #np.array([0.5, 0.5, 0.5]) #M0.translation #np.array([0.522418, 0.0448216, 0.824988])
p_target = np.array([-0.5, -0.5, 0.5])
