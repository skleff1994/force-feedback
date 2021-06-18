# Title : utils.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Utilities

from math import cos, sin
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

import time
from numpy import random
from numpy.core.numeric import normalize_axis_tuple

import pinocchio as pin

import pybullet as p
import pybullet_data

def animatePointMass(xs, sleep=1):
    '''
    Animate the point mass system with state trajectory xs
    '''
    # Check which model is used
    print(len(xs[0]))
    if(len(xs[0])>2):
        with_contact = True
    else:
        with_contact = False

    print("processing the animation ... ")
    cart_size = 1.
    fig = plt.figure()
    ax = plt.axes(xlim=(-7, 7), ylim=(-5, 5))
    patch = plt.Rectangle((0., 0.), cart_size, cart_size, fc='b')
    line, = ax.plot([], [], 'k-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        ax.add_patch(patch)
        line.set_data([], [])
        time_text.set_text('')
        return patch, line, time_text

    def animate(i):
        px = np.asscalar(xs[i][0])
        vx = np.asscalar(xs[i][1])
        patch.set_xy([px - cart_size / 2, 0])
        time = i * sleep / 1000.
        time_text.set_text('time = %.1f sec' % time)
        return patch, line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True)
    print("... processing done")
    plt.grid()
    plt.show()
    return anim


def plotPointMass(xs, us, dt=1e-2, ref=None):
    '''
    Plots state-control trajectories  (xs,us) of point mass or point mass in contact
    '''
    # Check which model is used
    if(len(xs[0])>2):
        with_contact = True
    else:
        with_contact = False
    # Extract trajectories from croco sequences
    T = len(us)
    tspan = np.linspace(0,T*dt,T+1)
        # control traj
    u = np.zeros(len(us))  # control
    for i in range(len(us)):
        u[i] = us[i]
        # State traj
    x1 = np.zeros(len(xs)) # position
    x2 = np.zeros(len(xs)) # velocity
    for i in range(len(xs)):
        x1[i] = xs[i][0]
        x2[i] = xs[i][1]
        # Add force if using augmented model 
    if(with_contact):
        x3 = np.zeros(len(xs)) 
        for i in range(len(xs)):
            x3[i] = xs[i][2]
    # Is there a reference to plot as well?
    if(ref is not None):
        with_ref = True
    else:
        with_ref = False
    # Create figs
    if(with_contact):
        fig, ax = plt.subplots(4,1)
    else:
        fig, ax = plt.subplots(3,1)
    # Plot position
    ax[0].plot(tspan, x1, 'b-', linewidth=3, label='p')
    if(with_ref):
        ax[0].plot(tspan, [ref[0]]*(T+1), 'k-.', linewidth=2, label='ref')
    ax[0].set_title('Position p', size=16)
    ax[0].set(xlabel='time (s)', ylabel='p (m)')
    ax[0].grid()
    # Plot velocity
    ax[1].plot(tspan, x2, 'b-', linewidth=3, label='v')
    if(with_ref):
        ax[1].plot(tspan, [ref[1]]*(T+1), 'k-.', linewidth=2, label='ref')
    ax[1].set_title('Velocity v', size=16)
    ax[1].set(xlabel='time (s)', ylabel='v (m/s)')
    ax[1].grid()
    # Plot force if necessary 
    if(with_contact):
        # Contact
        ax[2].plot(tspan, x3, 'b-', linewidth=3, label='lambda')
        if(with_ref):
            ax[2].plot(tspan, [ref[2]]*(T+1), 'k-.', linewidth=2, label='ref')
        ax[2].set_title('Contact force lambda', size=16)
        ax[2].set(xlabel='time (s)', ylabel='lmb (N)')
        ax[2].grid()
    # Plot control 
    ax[-1].plot(tspan[:T], u, 'k-', linewidth=3, label='u')
    ax[-1].set_title('Input force u', size=16)
    ax[-1].set(xlabel='time (s)', ylabel='u (N)')
    ax[-1].grid()
    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Point mass trajectory', size=16)
    plt.show()


def plotFiltered(Y_mea, X_hat, X_real, dt=1e-2):
    '''
    Plot point mass filtering using custom Kalman 
      Y_mea  : measurements
      X_hat  : estimates 
      X_real : ground truth
    '''
    # Extract trajectories and reshape
    T = len(Y_mea)
    ny = len(Y_mea[0])
    nx = len(X_real[0])
    tspan = np.linspace(0, T*dt, T+1)
    Y_mea = np.array(Y_mea).reshape((T, ny))
    X_hat = np.array(X_hat).reshape((T+1, nx))
    X_real = np.array(X_real).reshape((T+1, nx))
    # Create fig
    fig, ax = plt.subplots(2,1)
    # Plot position
    ax[0].plot(tspan[:T], Y_mea[:,0], 'b-', linewidth=2, alpha=.5, label='Measured')
    ax[0].plot(tspan, X_hat[:,0], 'r-', linewidth=3, alpha=.8, label='Filtered')
    ax[0].plot(tspan, X_real[:,0], 'k-.', linewidth=2, label='Ground truth')
    ax[0].set_title('Position p', size=16)
    ax[0].set(xlabel='time (s)', ylabel='p (m)')
    ax[0].grid()
    # Plot velocities
    ax[1].plot(tspan[:T], Y_mea[:,1], 'b-', linewidth=2, alpha=.5, label='Measured')
    ax[1].plot(tspan, X_hat[:,1], 'r-', linewidth=3, alpha=.8, label='Filtered')
    ax[1].plot(tspan, X_real[:,1], 'k-.', linewidth=2, label='Ground truth')
    ax[1].set_title('Velocities p', size=16)
    ax[1].set(xlabel='time (s)', ylabel='v (m/s)')
    ax[1].grid()
    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Kalman-filtered point mass trajectory', size=16)
    plt.show()


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


def get_p(q, pin_robot, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        robot     : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    p = np.empty((N,3))
    for i in range(N):
        pin.forwardKinematics(pin_robot.model, pin_robot.data, q[i])
        pin.updateFramePlacements(pin_robot.model, pin_robot.data)
        p[i,:] = pin_robot.data.oMf[id_endeff].translation.T
    return p


def get_v(q, dq, pin_robot, id_endeff):
    '''
    Returns end-effector velocities given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    v = np.empty((N,3))
    jac = np.zeros((6,pin_robot.model.nv))
    for i in range(N):
        # Get jacobian
        pin.computeJointJacobians(pin_robot.model, pin_robot.data, q[i,:])
        jac = pin.getFrameJacobian(pin_robot.model, pin_robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
        # Get EE velocity
        v[i,:] = jac.dot(dq[i])[:3]
    return v


def get_f(q, v, tau, pin_robot, id_endeff, dt=1e-2):
    '''
    Returns contact force in LOCAL frame based on FD estimate of joint acc
        q         : joint positions
        v         : joint velocities
        a         : joint acceleration
        tau       : joint torques
        pin_robot : Pinocchio wrapper
        id_endeff : id of EE frame
        dt        : step size for FD estimate of joint acceleration
    '''
    # Estimate joint accelerations with finite differences on v
    a = np.zeros(q.shape)
    for i in range(q.shape[0]):
        if i>0:
            a[i,:] = (v[i,:] - v[i-1,:])/dt
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Jacobian (in LOCAL coord)
        pin.computeJointJacobians(pin_robot.model, pin_robot.data, q[i,:])
        jac = pin.getFrameJacobian(pin_robot.model, pin_robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
        # Joint space inertia and its inverse + NL terms
        pin.crba(pin_robot.model, pin_robot.data, q[i,:])
        pin.computeMinverse(pin_robot.model, pin_robot.data, q[i,:])
        M = pin_robot.data.M
        Minv = pin_robot.data.Minv
        h = pin.nonLinearEffects(pin_robot.model, pin_robot.data, q[i,:], v[i,:])
        # Contact force
        f[i,:] = np.linalg.inv( jac.dot(Minv).dot(jac.T) ).dot( jac.dot(Minv).dot( h - tau[i,:] + M.dot(a[i,:]) ) )
    return f


import importlib_resources
import yaml
# Load config file
def load_config_file(config_file):
    '''
    Load config file (yaml)
    '''
    with open(config_file) as f:
        data = yaml.load(f)
    return data 

def get_urdf_path(robot_name, robot_family='kuka'):
    # Get config file
    with importlib_resources.path("robot_properties_"+robot_family, "config.py") as p:
        pkg_dir = p.parent.absolute()
    urdf_path = pkg_dir/"robot_properties_kuka"/(robot_name + ".urdf")
    return str(urdf_path)

class Data:
    ''' Sim data class '''

    def __init__(self, X_pred, ):
        pass

def save_data_to_yaml(data):
    '''Saves data to a yaml file'''
    pass

def load_data_from_yaml(file):
    '''Loads data from yaml file'''
    pass

def plot_sim_data(data):
    '''Plot sim data'''
    pass


# def weighted_moving_average(series, lookback = None):
#     if not lookback:
#         lookback = series.shape[0]
#     if series.shape[0] == 0:
#         return 0
#     assert 0 < lookback <= series.shape[0]
#     wma = np.zeros(series.shape[1])
#     # print("lookback = ", lookback)
#     # print("series = ", series)
#     lookback_offset = series.shape[0] - lookback
#     for index in range(lookback + lookback_offset - 1, lookback_offset - 1, -1):
#         weight = index - lookback_offset + 1
#         wma += series[index, :] * weight
#     return wma / ((lookback ** 2 + lookback) / 2)


# def hull_moving_average(series, lookback):
#     # assert lookback > 0
#     n = int(lookback ** 0.5)
#     hma_series = np.zeros((n, series.shape[1]))

#     for k in range(n, -1, -1):
#         s = series[:-k, :]
#         wma_half = weighted_moving_average(s, min(lookback // 2, s.shape[0]))
#         wma_full = weighted_moving_average(s, min(lookback, s.shape[0]))
#         hma_series[n-1-k,:] = wma_half*2 - wma_full
#     return weighted_moving_average(hma_series)

def weighted_moving_average(series, lookback = None):
    if not lookback:
        lookback = len(series)
    if len(series) == 0:
        return 0
    assert 0 < lookback <= len(series)

    wma = 0
    lookback_offset = len(series) - lookback
    for index in range(lookback + lookback_offset - 1, lookback_offset - 1, -1):
        weight = index - lookback_offset + 1
        wma += series[index] * weight
    return wma / ((lookback ** 2 + lookback) / 2)


def hull_moving_average(series, lookback):
    assert lookback > 0
    hma_series = []
    for k in range(int(lookback ** 0.5), -1, -1):
        s = series[:-k or None]
        wma_half = weighted_moving_average(s, min(lookback // 2, len(s)))
        wma_full = weighted_moving_average(s, min(lookback, len(s)))
        hma_series.append(wma_half * 2 - wma_full)
    return weighted_moving_average(hma_series)

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

# def get_f(self, q, v, a, tau, pin_robot, id_endeff):
#     '''
#     Returns contact force in LOCAL frame based on FD estimate of joint acc
#         q         : joint positions
#         v         : joint velocities
#         a         : joint acceleration
#         tau       : joint torques
#         pin_robot : Pinocchio wrapper
#         id_endeff : id of EE frame
#     '''
#     # Calculate contact force from (q, v, a, tau)
#     f = np.empty((u.shape[0], 6))
#     for i in range(u.shape[0]):
#         # Jacobian (in LOCAL coord)
#         pin.computeJointJacobians(pin_robot.model, pin_robot.data, q[i,:])
#         jac = pin.getFrameJacobian(pin_robot.model, pin_robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
#         # Joint space inertia and its inverse + NL terms
#         pin.crba(pin_robot.model, pin_robot.data, q[i,:])
#         pin.computeMinverse(pin_robot.model, pin_robot.data, q[i,:])
#         M = pin_robot.data.M
#         Minv = pin_robot.data.Minv
#         h = pin.nonLinearEffects(pin_robot.model, pin_robot.data, q[i,:], v[i,:])
#         # Contact force
#         f[i,:] = np.linalg.inv( jac.dot(Minv).dot(jac.T) ).dot( jac.dot(Minv).dot( h - u[i,:] + M.dot(a[i,:]) ) )
#     return f


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