import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from utils import pin_utils

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

# Check installed pkg
import importlib
FOUND_PYBULLET_PKG            = importlib.util.find_spec("pybullet")               is not None
FOUND_ROB_PROP_KUKA_PKG       = importlib.util.find_spec("robot_properties_kuka")  is not None
FOUND_ROB_PROP_TALOS_PKG      = importlib.util.find_spec("robot_properties_talos") is not None
FOUND_BULLET_UTILS_PKG        = importlib.util.find_spec("bullet_utils")           is not None
# FOUND_EXAMPLE_ROBOT_DATA_PKG  = importlib.util.find_spec("example_robot_data")     is not None

if(FOUND_PYBULLET_PKG):
    import pybullet as p
else:
    logger.error('You need to install PyBullet ( https://pypi.org/project/pybullet/ )')

if(FOUND_ROB_PROP_KUKA_PKG):
    from robot_properties_kuka.iiwaWrapper import IiwaRobot as IiwaRobot
else:
    logger.error('You need to install robot_properties_kuka ( https://github.com/machines-in-motion/robot_properties_kuka )')

if(FOUND_ROB_PROP_TALOS_PKG):
    from robot_properties_talos.talosArmWrapper import TalosArmRobot
    from robot_properties_talos.talosReducedWrapper import TalosReducedRobot
    from robot_properties_talos.talosFullWrapper import TalosFullRobot
else:
    logger.error('You need to install robot_properties_talos ( https://github.com/machines-in-motion/robot_properties_talos )')

if(FOUND_BULLET_UTILS_PKG):
    from bullet_utils.env import BulletEnvWithGround
    from bullet_utils.wrapper import PinBulletWrapper
else:
    logger.error('You need to install bullet_utils ( https://github.com/machines-in-motion/bullet_utils )')

# if(FOUND_EXAMPLE_ROBOT_DATA_PKG):
#     import example_robot_data
# else:
#     logger.error('You need to install example_robot_data !')

# Global & default settings (change CAREFULLY)
SUPPORTED_ROBOTS         = ['iiwa', 'talos_arm', 'talos_reduced']

TALOS_DEFAULT_MESH_PATH  = '/opt/openrobots/share'

IIWA_DEFAULT_BASE_POS = [0, 0, 0]
IIWA_DEFAULT_BASE_RPY = [0, 0, 0]

TALOS_ARM_DEFAULT_BASE_POS = [0, 0, 0]
TALOS_ARM_DEFAULT_BASE_RPY = [0, 0, 0]

TALOS_REDUCED_DEFAULT_BASE_POS = [0, 0, 1.03]
TALOS_REDUCED_DEFAULT_BASE_RPY = [0, 0, 0]



# Load robot in PyBullet environment 
def init_bullet_simulation(robot_name, dt=1e3, x0=None):
    '''
    Initialize a PyBullet simulation environment with robot SUPPORTED_ROBOTS
    '''
    if(robot_name not in SUPPORTED_ROBOTS):
        logger.error("Specified robot not supported ! Select a robot in "+str(SUPPORTED_ROBOTS))
    else:
        if(robot_name == 'iiwa'):
            return init_iiwa_bullet(dt=dt, x0=x0)
        elif(robot_name == 'talos_arm'):
            return init_talos_arm_bullet(dt=dt, x0=x0)
        elif(robot_name == 'talos_reduced'):
            return init_talos_reduced_bullet(dt=dt, x0=x0)


# Load KUKA arm in PyBullet environment
def init_iiwa_bullet(dt=1e3, x0=None, pos=IIWA_DEFAULT_BASE_POS, orn=IIWA_DEFAULT_BASE_RPY):
    '''
    Loads KUKA LBR iiwa model in PyBullet simulator
    using the PinBullet wrapper to simplify interactions
      INPUT:
        dt        : simulator time step
        x0        : initial robot state (pos and vel)
    '''
    # Info log
    print("")
    logger.info("Initializing KUKA iiwa in PyBullet simulator...")
    print("")
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    orn_quat = p.getQuaternionFromEuler(orn)
    base_placement = pin.XYZQUATToSE3(pos + list(orn_quat))
    robot_simulator = env.add_robot(IiwaRobot(pos, orn_quat))
    # Initialize
    if(x0 is None):
        q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
        dq0 = np.zeros(robot_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:robot_simulator.pin_robot.model.nq]
        dq0 = x0[robot_simulator.pin_robot.model.nv:]
    robot_simulator.reset_state(q0, dq0)
    robot_simulator.forward_robot(q0, dq0)
    return env, robot_simulator, base_placement


# Load TALOS arm in PyBullet environment
def init_talos_arm_bullet(dt=1e3, x0=None, pos=TALOS_ARM_DEFAULT_BASE_POS, orn=TALOS_ARM_DEFAULT_BASE_RPY):
    '''
    Loads TALOS left arm model in PyBullet simulator
    using the PinBullet wrapper to simplify interactions
      INPUT:
        dt        : simulator time step
        x0        : initial robot state (pos and vel)
    '''
    # Info log
    print("")
    logger.info("Initializing TALOS left arm in PyBullet simulator...")
    print("")
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    orn_quat = p.getQuaternionFromEuler(orn)
    base_placement = pin.XYZQUATToSE3(pos + list(orn_quat)) 
    robot_simulator = env.add_robot(TalosArmRobot(pos, orn_quat))
    # Initialize
    if(x0 is None):
        q0 = np.array([2., 0., 0., 0., 0., 0., 0.])
        dq0 = np.zeros(robot_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:robot_simulator.pin_robot.model.nq]
        dq0 = x0[robot_simulator.pin_robot.model.nv:]
    robot_simulator.reset_state(q0, dq0)
    robot_simulator.forward_robot(q0, dq0)
    return env, robot_simulator, base_placement


# Load TALOS arm in PyBullet environment
def init_talos_reduced_bullet(dt=1e3, x0=None, pos=TALOS_REDUCED_DEFAULT_BASE_POS, orn=TALOS_REDUCED_DEFAULT_BASE_RPY):
    '''
    Loads TALOS left arm model in PyBullet simulator
    using the PinBullet wrapper to simplify interactions
      INPUT:
        dt        : simulator time step
        x0        : initial robot state (pos and vel)
    '''
    # Info log
    print("")
    logger.info("Initializing TALOS left arm in PyBulletsimulator...")
    print("")
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    orn_quat = p.getQuaternionFromEuler(orn)
    base_placement = pin.XYZQUATToSE3(pos + list(orn_quat)) 
    robot_simulator = env.add_robot(TalosReducedRobot(pos, orn_quat))
    # Initialize
    if(x0 is None):
        q0 = np.array([2., 0., 0., 0., 0., 0., 0.])
        dq0 = np.zeros(robot_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:robot_simulator.pin_robot.model.nq]
        dq0 = x0[robot_simulator.pin_robot.model.nv:]
    robot_simulator.reset_state(q0, dq0)
    robot_simulator.forward_robot(q0, dq0)
    return env, robot_simulator, base_placement




# Get contact wrench from robot simulator
def get_contact_wrench(pybullet_simulator, id_endeff):
    '''
    Get contact wrench in LOCAL contact frame
    '''
    contact_points = p.getContactPoints()
    force = np.zeros(6)
    for ci in reversed(contact_points):
        p_ct = np.array(ci[6])
        contact_normal = ci[7]
        normal_force = ci[9]
        lateral_friction_direction_1 = ci[11]
        lateral_friction_force_1 = ci[10]
        lateral_friction_direction_2 = ci[13]
        lateral_friction_force_2 = ci[12]
        # Wrench in LOCAL contact frame
        linear_LOCAL = np.array([normal_force, lateral_friction_force_1, lateral_friction_force_2])
        wrench_LOCAL = np.concatenate([linear_LOCAL, np.zeros(3)])
        # LOCAL contact placement
        R_ct = np.vstack([np.array(contact_normal), np.array(lateral_friction_direction_1), np.array(lateral_friction_direction_2)]).T
        M_ct = pin.SE3(R_ct, p_ct) 
        # wrench LOCAL(p)-->WORLD
        wrench_WORLD = M_ct.act(pin.Force(wrench_LOCAL))
        # wrench WORLD-->LOCAL(EE)
        wrench_croco = -pybullet_simulator.pin_robot.data.oMf[id_endeff].actInv(wrench_WORLD)
        force =+ wrench_croco.vector
        return force


# Get joint torques from robot simulator
def get_contact_joint_torques(pybullet_simulator, id_endeff):
    '''
    Get contact wrench in LOCAL contact frame
    '''
    wrench = get_contact_wrench(pybullet_simulator, id_endeff)
    jac = pybullet_simulator.pin_robot.data.J
    joint_torques = jac.T @ wrench
    return joint_torques




# Display
def display_ball(p_des, robot_base_pose=pin.SE3.Identity(), RADIUS=.05, COLOR=[1.,1.,1.,1.]):
    '''
    Create a sphere visual object in PyBullet (no collision)
    Transformed because reference p_des is in pinocchio WORLD frame, which is different
    than PyBullet WORLD frame if the base placement in the simulator is not (eye(3), zeros(3))
    INPUT: 
        p_des           : desired position of the ball in pinocchio.WORLD
        robot_base_pose : initial pose of the robot BASE in bullet.WORLD
        RADIUS          : radius of the ball
        COLOR           : color of the ball
    '''
    logger.debug("Creating PyBullet sphere visual...")
    # pose of the sphere in bullet WORLD
    M = pin.SE3(np.eye(3), p_des)  # ok for talos reduced since pin.W = bullet.W but careful with talos_arm if base is moved
    quat = pin.SE3ToXYZQUAT(M)     
    visualBallId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                       radius=RADIUS,
                                       rgbaColor=COLOR,
                                       visualFramePosition=quat[:3],
                                       visualFrameOrientation=quat[3:])
    ballId = p.createMultiBody(baseMass=0.,
                               baseInertialFramePosition=[0.,0.,0.],
                               baseVisualShapeIndex=visualBallId,
                               basePosition=[0.,0.,0.],
                               useMaximalCoordinates=True)

    return ballId


# Load contact surface in PyBullet for contact experiments
def display_contact_surface(M, robotId=1, radius=.25, length=0.0, with_collision=False, TILT=[0., 0., 0.]):
    '''
    Creates contact surface object in PyBullet as a flat cylinder 
      M              : contact placement expressed in simulator WORLD frame
      robotId        : id of the robot in simulator
      radius         : radius of cylinder
      length         : length of cylinder
      with_collision : robot can collide with surface if True, cannot otherwise
      TILT           : RPY tilt of the surface
    '''
    logger.info("Creating PyBullet contact surface...")
    # Tilt contact surface (default 0)
    TILT_rotation = pin.utils.rpyToMatrix(TILT[0], TILT[1], TILT[2])
    M.rotation = TILT_rotation.dot(M.rotation)
    # Get quaternion
    quat = pin.SE3ToXYZQUAT(M)
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                        radius=radius,
                                        length=length,
                                        rgbaColor=[.1, .8, .1, .5],
                                        visualFramePosition=quat[:3],
                                        visualFrameOrientation=quat[3:])
    # With collision
    if(with_collision):
      collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                radius=radius,
                                                height=length,
                                                collisionFramePosition=quat[:3],
                                                collisionFrameOrientation=quat[3:])
      contactId = p.createMultiBody(baseMass=0.,
                                    baseInertialFramePosition=[0.,0.,0.],
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=[0.,0.,0.],
                                    useMaximalCoordinates=True)
                    
      # Desactivate collisions for all links except end-effector of robot
      # TODO: do not hard-code the PyBullet EE id
      for i in range(p.getNumJoints(robotId)):
            # ji = p.getJointInfo(robotId, i)
            # print(ji)
            p.setCollisionFilterPair(contactId, robotId, -1, i, 1)
            # p.setCollisionFilterPair(contactId, robotId, -1, i, 0)
    #   p.setCollisionFilterPair(contactId, robotId, -1,  8, 1)

      return contactId
    # Without collisions
    else:
      contactId = p.createMultiBody(baseMass=0.,
                        baseInertialFramePosition=[0.,0.,0.],
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[0.,0.,0.],
                        useMaximalCoordinates=True)
      return contactId


# Load contact surface in PyBullet for contact experiments
def remove_body_from_sim(bodyId):
    '''
    Removes bodyfrom sim env
    '''
    logger.info("Removing body "+str(bodyId)+" from simulation !")
    p.removeBody(int(bodyId))


# Set lateral friction coefficient to PyBullet body
def set_friction_coef(bodyId, coef):
  '''
  Set lateral friction coefficient to PyBullet body
  Input :
    bodyId : PyBullet body unique id
    coef   : friction coefficient in (0,1)
  '''
  p.changeDynamics(bodyId, -1, lateralFriction=0.5) 
  logger.info("Set friction of body n°"+str(bodyId)+" to "+str(coef)) 
  # print(p.getDynamicsInfo(id, -1))




# def rotationMatrixFromTwoVectors(a, b):
#     a_copy = a / np.linalg.norm(a)
#     b_copy = b / np.linalg.norm(b)
#     a_cross_b = np.cross(a_copy, b_copy, axis=0)
#     s = np.linalg.norm(a_cross_b)
#     if s == 0:
#         return np.eye(3)
#     c = a_copy.dot(b_copy) 
#     ab_skew = pin.skew(a_cross_b)
#     return np.eye(3) + ab_skew + ( (1 - c) / (s**2) ) * ab_skew.dot(ab_skew) 

# def weighted_moving_average(series, lookback = None):
#     if not lookback:
#         lookback = len(series)
#     if len(series) == 0:
#         return 0
#     assert 0 < lookback <= len(series)

#     wma = 0
#     lookback_offset = len(series) - lookback
#     for index in range(lookback + lookback_offset - 1, lookback_offset - 1, -1):
#         weight = index - lookback_offset + 1
#         wma += series[index] * weight
#     return wma / ((lookback ** 2 + lookback) / 2)

# def hull_moving_average(series, lookback):
#     assert lookback > 0
#     hma_series = []
#     for k in range(int(lookback ** 0.5), -1, -1):
#         s = series[:-k or None]
#         wma_half = weighted_moving_average(s, min(lookback // 2, len(s)))
#         wma_full = weighted_moving_average(s, min(lookback, len(s)))
#         hma_series.append(wma_half * 2 - wma_full)
#     return weighted_moving_average(hma_series)

# N = 500
# X = np.linspace(-10,10,N)
# Y = np.vstack([np.sin(X), np.cos(X)]).T
# W = Y + np.vstack([np.random.normal(0., .2, N), np.random.normal(0, .2, N)]).T
# Z = Y.copy()
# lookback=50
# for i in range(N):
#     if(i==0):
#         pass
#     else:
#         Z[i,:] = hull_moving_average(W[:i,:], min(lookback,i))
# fig, ax = plt.subplots(1,2)
# ax[0].plot(X, Y[:,0], 'b-', label='ground truth')
# ax[0].plot(X, W[:,0], 'g-', label='noised data')
# ax[0].plot(X, Z[:,0], 'r-', label='HMA') 
# ax[1].plot(X, Y[:,1], 'b-', label='ground truth')
# ax[1].plot(X, W[:,1], 'g-', label='noised data')
# ax[1].plot(X, Z[:,1], 'r-', label='HMA') 
# ax[0].legend()
# plt.show()


