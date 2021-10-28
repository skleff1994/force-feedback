from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import numpy as np
import pinocchio as pin

# Load KUKA arm in PyBullet environment
def init_kuka_simulator(dt=1e3, x0=None):
    '''
    Loads KUKA LBR iiwa model in PyBullet using the 
    Pinocchio-PyBullet wrapper to simplify interactions
    '''
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    pybullet_simulator = env.add_robot(IiwaRobot())
    # Initialize
    if(x0 is None):
        q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
        dq0 = np.zeros(pybullet_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:pybullet_simulator.pin_robot.model.nq]
        dq0 = x0[pybullet_simulator.pin_robot.model.nv:]
    pybullet_simulator.reset_state(q0, dq0)
    pybullet_simulator.forward_robot(q0, dq0)
    return env, pybullet_simulator


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
        # Linear force in LOCAL frame
        linear_LOCAL = np.array([normal_force, lateral_friction_force_1, lateral_friction_force_2])
        wrench_LOCAL = np.concatenate([linear_LOCAL, np.zeros(3)])
        # LOCAL contact placement
        R_ct = np.vstack([np.array(contact_normal), np.array(lateral_friction_direction_1), np.array(lateral_friction_direction_2)]).T
        M_ct = pin.SE3(R_ct, p_ct) # LOCAL --> WORLD
        # wrench_WORLD
        wrench_WORLD = M_ct.act(pin.Force(wrench_LOCAL))
        # wrench Croco frame
        wrench_croco = pybullet_simulator.pin_robot.data.oMf[id_endeff].actInv(wrench_WORLD)
        force =+ wrench_croco.vector
        return force

# Load contact surface in PyBullet for contact experiments
def display_contact_surface(M, robotId=1, radius=.5, length=0.0, with_collision=False):
    '''
    Create contact surface object in pybullet and display it
      M       : contact placement
      robotId : id of the robot 

    '''

    quat = pin.SE3ToXYZQUAT(M)
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                        radius=radius,
                                        length=length,
                                        rgbaColor=[.8, .1, .1, .7],
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
      for i in range(p.getNumJoints(robotId)):
        p.setCollisionFilterPair(contactId, robotId, -1, i, 0)
      p.setCollisionFilterPair(contactId, robotId, -1, 8, 1)

      return contactId
    # Without collisions
    else:
      contactId = p.createMultiBody(baseMass=0.,
                        baseInertialFramePosition=[0.,0.,0.],
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[0.,0.,0.],
                        useMaximalCoordinates=True)
      return contactId



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


    # contact_points = p.getContactPoints(1, 2)
    # for id_pt, pt in enumerate(contact_points):
    #   F_mea_pyb[i, :] += pt[9]
    #   print("      Contact point n°"+str(id_pt)+" : ")
    #   print("             - normal vec   = "+str(pt[7]))
    #   # print("             - m_ct.trans   = "+str(M_ct.actInv(np.array(pt[7]))))
    #   # print("             - distance     = "+str(pt[8])+" (m)")
    #   # print("             - normal force = "+str(pt[9])  +" (N)")
    #   # print("             - lat1 force   = "+str(pt[10]) +" (N)")
    #   # print("             - lat2 force   = "+str(pt[12]) +" (N)")

# def animateCartpole(xs, sleep=50):
#     print("processing the animation ... ")
#     cart_size = 1.
#     pole_length = 5.
#     fig = plt.figure()
#     ax = plt.axes(xlim=(-8, 8), ylim=(-6, 6))
#     patch = plt.Rectangle((0., 0.), cart_size, cart_size, fc='b')
#     line, = ax.plot([], [], 'k-', lw=2)
#     time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

#     def init():
#         ax.add_patch(patch)
#         line.set_data([], [])
#         time_text.set_text('')
#         return patch, line, time_text

#     def animate(i):
#         x_cart = np.asscalar(xs[i][0])
#         y_cart = 0.
#         theta = np.asscalar(xs[i][1])
#         patch.set_xy([x_cart - cart_size / 2, y_cart - cart_size / 2])
#         x_pole = np.cumsum([x_cart, -pole_length * sin(theta)])
#         y_pole = np.cumsum([y_cart, pole_length * cos(theta)])
#         line.set_data(x_pole, y_pole)
#         time = i * sleep / 1000.
#         time_text.set_text('time = %.1f sec' % time)
#         return patch, line, time_text

#     anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True)
#     print("... processing done")
#     plt.show()
#     return anim