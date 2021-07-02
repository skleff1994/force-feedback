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

# Animate and plot point mass from X,U trajs 
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

# Plots Kalman filtered trajs : measured, estimated, ground truth
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


# Post-process trajectories with pinocchio
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


# Cost weights profiles, useful for reaching tasks/cost design
def cost_weight_tanh(i, N, max_weight=1., alpha=1., alpha_cut=0.25):
    '''
    Monotonically increasing weight profile over [0,...,N]
    based on a custom scaled hyperbolic tangent 
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : value of the weight at the end of the window (must be >0)
       alpha      : controls the sharpness of the tanh (alpha high <=> very sharp)
       alpha_cut  : shifts tanh over the time window (i.e. time of inflexion point)
     OUPUT:
       Cost weight at step i : it tarts at weight=0 (when i=0) and
       ends at weight<= max_weight (at i=N). As alpha --> inf, we tend
       toward max_weight
    '''
    return 0.5*max_weight*( np.tanh(alpha*(i/N) -alpha*alpha_cut) + np.tanh(alpha*alpha_cut) )

def cost_weight_linear(i, N, min_weight=0., max_weight=1.):
    '''
    Linear cost weight profile over [0,...,N]
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : value of the weight at the end of the window (must be >=min_weight)
       min_weight : value of the weight at the start of the window (must be >=0)
     OUPUT:
       Cost weight at step i
    '''
    return (max_weight-min_weight)/N * i + min_weight


import importlib_resources
import yaml
import os

# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f)
    return data 

# Returns urdf path of a kuka robot 
def get_urdf_path(robot_name, robot_family='kuka'):
    # Get config file
    with importlib_resources.path("robot_properties_"+robot_family, "config.py") as p:
        pkg_dir = p.parent.absolute()
    urdf_path = pkg_dir/"robot_properties_kuka"/(robot_name + ".urdf")
    return str(urdf_path)


# Save data (dict) into compressed npz
def save_data(sim_data, save_name=None, save_dir=None):
    '''
    Saves data to a compressed npz file (binary)
    '''
    print('Compressing & saving data...')
    if(save_name is None):
        save_name = 'sim_data_NO_NAME'+str(time.time())
    if(save_dir is None):
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
    save_path = save_dir+'/'+save_name+'.npz'
    np.savez_compressed(save_path, data=sim_data)
    print("Saved data to "+str(save_path)+" !")

# Loads dict from compressed npz
def load_data(npz_file):
    '''
    Loads a npz archive of sim_data into a dict
    '''
    print('Loading data...')
    d = np.load(npz_file, allow_pickle=True)
    return d['data'][()]

# Extract MPC simu-specific plotting data from sim data
def extract_plot_data_from_sim_data(sim_data):
    '''
    Extract plot data from simu data
    '''
    print('Extracting plotting data from simulation data...')
    plot_data = {}
    nx = sim_data['X_mea'].shape[1]
    nq = nx//2
    nv = nx-nq
    # state predictions
    plot_data['q_pred'] = sim_data['X_pred'][:,:,:nq]
    plot_data['v_pred'] = sim_data['X_pred'][:,:,nv:]
    # measured state
    plot_data['q_mea'] = sim_data['X_mea'][:,:nq]
    plot_data['v_mea'] = sim_data['X_mea'][:,nv:]
    plot_data['q_mea_no_noise'] = sim_data['X_mea_no_noise'][:,:nq]
    plot_data['v_mea_no_noise'] = sim_data['X_mea_no_noise'][:,nv:]
    # desired state (append 1st state at start)
    plot_data['q_des'] = np.vstack([sim_data['X_mea'][0,:nq], sim_data['X_pred'][:,1,:nq]])
    plot_data['v_des'] = np.vstack([sim_data['X_mea'][0,nv:], sim_data['X_pred'][:,1,nv:]])
    # end-eff position
    plot_data['p_mea'] = sim_data['P_mea']
    plot_data['p_mea_no_noise'] = sim_data['P_mea_no_noise']
    plot_data['p_pred'] = sim_data['P_pred']
    plot_data['p_des'] = sim_data['P_des'] #np.vstack([sim_data['p0'], sim_data['P_pred'][:,10,:]])
    # control
    plot_data['u_pred'] = sim_data['U_pred']
    plot_data['u_des'] = sim_data['U_pred'][:,0,:]
    plot_data['u_mea'] = sim_data['U_mea']
    # acc error
    plot_data['a_err'] = sim_data['A_err']
    # Misc. params
    plot_data['T_tot'] = sim_data['T_tot']
    plot_data['N_simu'] = sim_data['N_simu']
    plot_data['N_ctrl'] = sim_data['N_ctrl']
    plot_data['N_plan'] = sim_data['N_plan']
    plot_data['dt_plan'] = sim_data['dt_plan']
    plot_data['dt_ctrl'] = sim_data['dt_ctrl']
    plot_data['dt_simu'] = sim_data['dt_simu']
    plot_data['nq'] = sim_data['nq']
    plot_data['nv'] = sim_data['nv']
    plot_data['T_h'] = sim_data['T_h']
    plot_data['N_h'] = sim_data['N_h']
    plot_data['p_ref'] = sim_data['p_ref']
    plot_data['alpha'] = sim_data['alpha']
    plot_data['beta'] = sim_data['beta']
    plot_data['K'] = sim_data['K']
    return plot_data

# Same giving npz path OR dict as argument
def extract_plot_data(input_data):
    '''
    Extract plot data from npz archive or sim_data
    '''
    if(type(input_data)==str):
        sim_data = load_yaml_file(input_data)
    elif(type(input_data)==dict):
        sim_data = input_data
    else:
        TypeError("Input data must be a Python dict or a path to .npz archive")
    return extract_plot_data_from_sim_data(sim_data)


from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib

# Plot data
def plot_results(plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True,
                            AUTOSCALE=False):
    '''
    Plot sim data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
      AUTOSCALE                 : rescale y-axis of endeff plot 
                                  based on maximum value taken
    '''
    print('Plotting data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    dt_ctrl = plot_data['dt_ctrl']
    dt_simu = plot_data['dt_simu']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    p_ref = plot_data['p_ref']

    # Create time spans for X and U + Create figs and subplots
    t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
    t_span_simu_u = np.linspace(0, T_tot-dt_simu, N_simu)
    t_span_ctrl_x = np.linspace(0, T_tot, N_ctrl+1)
    t_span_ctrl_u = np.linspace(0, T_tot-dt_ctrl, N_ctrl)
    t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8))
    fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8))
    fig_p, ax_p = plt.subplots(3,1, figsize=(19.2,10.8)) 
    fig_a, ax_a = plt.subplots(nq,2, figsize=(19.2,10.8))
    fig_K, ax_K = plt.subplots(nq,2, figsize=(19.2,10.8))

    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            q_pred_i = plot_data['q_pred'][:,:,i]
            v_pred_i = plot_data['v_pred'][:,:,i]
            u_pred_i = plot_data['u_pred'][:,:,i]

            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
                lc_q.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred) 
                lc_u.set_array(tspan_u_pred)
                # Customize
                lc_q.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_u.set_linestyle('-')
                lc_q.set_linewidth(1)
                lc_v.set_linewidth(1)
                lc_u.set_linewidth(1)
                # Plot collections
                ax_x[i,0].add_collection(lc_q)
                ax_x[i,1].add_collection(lc_v)
                ax_u[i].add_collection(lc_u)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
                ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

        # Joint position
          # Desired
        ax_x[i,0].plot(t_span_plan_x, plot_data['q_des'][:,i], 'b-', label='Desired')
          # Measured
        ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)', linewidth=2)
        ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{}$ (rad)'.format(i))
        ax_x[i,0].grid()
        
        # Joint velocity 
          # Desired 
        ax_x[i,1].plot(t_span_plan_x, plot_data['v_des'][:,i], 'b-', label='Desired')
          # Measured 
        ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)')
        ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{}$ (rad/s)'.format(i))
        ax_x[i,1].grid()
        
        # Joint torques
          # Desired  
        ax_u[i].plot(t_span_plan_u, plot_data['u_des'][:,i], 'b-', label='Desired')
          # Measured
        ax_u[i].plot(t_span_simu_u, plot_data['u_mea'][:,i], 'r-', label='Measured') 
        ax_u[i].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
        ax_u[i].grid()

        # Joint velocity error (avg over 1 control cycle)
        ax_a[i,0].plot(t_span_ctrl_u, plot_data['a_err'][:,i], 'b-', label='Velocity error (average)')
        ax_a[i,0].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
        ax_a[i,0].grid()
        # Joint acceleration error (avg over 1 control cycle)
        ax_a[i,1].plot(t_span_ctrl_u, plot_data['a_err'][:,nq+i], 'b-', label='Acceleration error (average)')
        ax_a[i,1].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
        ax_a[i,1].grid()

        # Ricatti gains on q
        ax_K[i,0].plot(t_span_plan_u, plot_data['K'][:,i,i], 'b-', label='Ricatti gain Kq')
        ax_K[i,0].set(xlabel='t (s)', ylabel='$Kq_{}$ (Nm)'.format(i,i))
        ax_K[i,0].grid()
        # Ricatti gains on v
        ax_K[i,1].plot(t_span_plan_u, plot_data['K'][:,i,nq+i], 'b-', label='Ricatti gain Kv')
        ax_K[i,1].set(xlabel='t (s)', ylabel='$Kv_{}$ (Nm)'.format(i,i))
        ax_K[i,1].grid()

        # Legend
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

        handles_u, labels_u = ax_u[i].get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

    # Plot endeff
    # x
    ax_p[0].plot(t_span_plan_x, plot_data['p_des'][:,0]-p_ref[0], 'b-', label='p_des - p_ref', alpha=0.5)
    ax_p[0].plot(t_span_simu_x, plot_data['p_mea'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (WITH noise)', linewidth=1, alpha=0.3)
    ax_p[0].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (NO noise)', linewidth=2)
    ax_p[0].set_title('x-position-ERROR')
    ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
    # 
    ax_p[0].grid()
    # y
    ax_p[1].plot(t_span_plan_x, plot_data['p_des'][:,1]-p_ref[1], 'b-', label='py_des - py_ref', alpha=0.5)
    ax_p[1].plot(t_span_simu_x, plot_data['p_mea'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (WITH noise)', linewidth=1, alpha=0.3)
    ax_p[1].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (NO noise)', linewidth=2)
    ax_p[1].set_title('y-position-ERROR')
    ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
    ax_p[1].grid()
    # z
    ax_p[2].plot(t_span_plan_x, plot_data['p_des'][:,2]-p_ref[2], 'b-', label='pz_des - pz_ref', alpha=0.5)
    ax_p[2].plot(t_span_simu_x, plot_data['p_mea'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (WITH noise)', linewidth=1, alpha=0.3)
    ax_p[2].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (NO noise)', linewidth=2)
    ax_p[2].set_title('z-position-ERROR')
    ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
    ax_p[2].grid()
    # Add frame ref if any
    ax_p[0].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., label='err=0', alpha=0.5)
    ax_p[1].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., label='err=0', alpha=0.5)
    ax_p[2].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., label='err=0', alpha=0.5)
    # Set ylim if any
    if(AUTOSCALE):
        ax_p_ylim = np.max(np.abs(plot_data['p_mea']-plot_data['p_ref']))
        ax_p[0].set_ylim(-ax_p_ylim, ax_p_ylim) 
        ax_p[1].set_ylim(-ax_p_ylim, ax_p_ylim) 
        ax_p[2].set_ylim(-ax_p_ylim, ax_p_ylim) 

    if(PLOT_PREDICTIONS):
        # For each component (x,y,z)
        for i in range(3):
            p_pred_i = plot_data['p_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, p_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                lc_p.set_array(tspan_x_pred)
                # Customize
                lc_p.set_linestyle('-')
                lc_p.set_linewidth(1)
                # Plot collections
                ax_p[i].add_collection(lc_p)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_p[i].scatter(tspan_x_pred, p_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    handles_p, labels_p = ax_p[0].get_legend_handles_labels()
    fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

    # Titles
    fig_x.suptitle('State = joint positions, velocities', size=16)
    fig_u.suptitle('Control = joint torques', size=16)
    fig_p.suptitle('End-effector trajectories errors', size=16)
    fig_a.suptitle('Average tracking errors over control cycles (1ms)', size=16)
    fig_K.suptitle('Diagonal Ricatti feedback gains (Kq,Kv)', size=16)

    # Save figs
    if(SAVE):
        figs = {'x': fig_x, 
                'u': fig_u,
                'a': fig_a,
                'p': fig_p,
                'K': fig_K}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    plt.close('all')


# ### DEPRECATED (using YAML)
# def save_data_to_yaml(sim_data, save_name=None, save_dir=None):
#     '''
#     Saves data to a yaml file
#     '''
#     print('Saving data...')
#     if(save_name is None):
#         save_name = 'sim_data_NO_NAME'+str(time.time())
#     if(save_dir is None):
#         save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
#     yaml_save_path = save_dir+'/'+save_name+'.yml'
#     with open(yaml_save_path, 'w') as f:
#         yaml.dump(sim_data, f)
#     print("Saved data to "+str(yaml_save_path)+" !")

# def extract_plot_data_from_sim_data(sim_data):
#     '''
#     Extract plot data from simu data
#     '''
#     print('Extracting plotting data from simulation data...')
#     plot_data = {}
#     nx = sim_data['X_mea'].shape[1]
#     nq = nx//2
#     nv = nx-nq
#     # state predictions
#     plot_data['q_pred'] = sim_data['X_pred'][:,:,:nq]
#     plot_data['v_pred'] = sim_data['X_pred'][:,:,nv:]
#     # measured state
#     plot_data['q_mea'] = sim_data['X_mea'][:,:nq]
#     plot_data['v_mea'] = sim_data['X_mea'][:,nv:]
#     plot_data['q_mea_no_noise'] = sim_data['X_mea_no_noise'][:,:nq]
#     plot_data['v_mea_no_noise'] = sim_data['X_mea_no_noise'][:,nv:]
#     # desired state (append 1st state at start)
#     plot_data['q_des'] = np.vstack([sim_data['X_mea'][0,:nq], sim_data['X_pred'][:,1,:nq]])
#     plot_data['v_des'] = np.vstack([sim_data['X_mea'][0,nv:], sim_data['X_pred'][:,1,nv:]])
#     # end-eff position
#     plot_data['p_mea'] = sim_data['P_mea']
#     plot_data['p_mea_no_noise'] = sim_data['P_mea_no_noise']
#     plot_data['p_pred'] = sim_data['P_pred']
#     plot_data['p_des'] = sim_data['P_des'] #np.vstack([sim_data['p0'], sim_data['P_pred'][:,10,:]])
#     # control
#     plot_data['u_pred'] = sim_data['U_pred']
#     plot_data['u_des'] = sim_data['U_pred'][:,0,:]
#     plot_data['u_mea'] = sim_data['U_mea']
#     # acc error
#     plot_data['a_err'] = sim_data['A_err']
#     # Misc. params
#     plot_data['T_tot'] = sim_data['T_tot']
#     plot_data['N_simu'] = sim_data['N_simu']
#     plot_data['N_ctrl'] = sim_data['N_ctrl']
#     plot_data['N_plan'] = sim_data['N_plan']
#     plot_data['dt_plan'] = sim_data['dt_plan']
#     plot_data['dt_ctrl'] = sim_data['dt_ctrl']
#     plot_data['dt_simu'] = sim_data['dt_simu']
#     plot_data['nq'] = sim_data['nq']
#     plot_data['nv'] = sim_data['nv']
#     plot_data['T_h'] = sim_data['T_h']
#     plot_data['N_h'] = sim_data['N_h']
#     plot_data['p_ref'] = sim_data['p_ref']
#     return plot_data

# def extract_plot_data_from_yaml(sim_yaml_file):
#     '''
#     Extract plot data from yaml file (in which simu data was dumped)
#     '''
#     print("  [1] Loading sim data from YAML file...")
#     sim_data = load_yaml_file(sim_yaml_file)
#     print("  [2] Extracting plot data...")
#     plot_data = extract_plot_data_from_sim_data(sim_data)
#     return plot_data

# def plot_results_from_plot_data(plot_data, PLOT_PREDICTIONS=False, 
#                                            pred_plot_sampling=100, 
#                                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                                            SHOW=True):
#     '''
#     Plot sim data
#      Input:
#       plot_data          : plotting data
#       PLOT_PREDICTIONS   : True or False
#       pred_plot_sampling : plot every pred_plot_sampling prediction 
#                            to avoid huge amount of plotted data 
#                            ("1" = plot all)
#     '''
#     print('Plotting data...')
#     T_tot = plot_data['T_tot']
#     N_simu = plot_data['N_simu']
#     N_ctrl = plot_data['N_ctrl']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']
#     dt_ctrl = plot_data['dt_ctrl']
#     dt_simu = plot_data['dt_simu']
#     nq = plot_data['nq']
#     T_h = plot_data['T_h']
#     N_h = plot_data['N_h']
#     p_ref = plot_data['p_ref']

#     # Create time spans for X and U + Create figs and subplots
#     t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
#     t_span_simu_u = np.linspace(0, T_tot-dt_simu, N_simu)
#     t_span_ctrl_x = np.linspace(0, T_tot, N_ctrl+1)
#     t_span_ctrl_u = np.linspace(0, T_tot-dt_ctrl, N_ctrl)
#     t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
#     t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
#     fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8))
#     fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8))
#     fig_p, ax_p = plt.subplots(3,1, figsize=(19.2,10.8)) 
#     fig_a, ax_a = plt.subplots(nq,2, figsize=(19.2,10.8))

#     # For each joint
#     for i in range(nq):

#         if(PLOT_PREDICTIONS):

#             # Extract state predictions of i^th joint
#             q_pred_i = plot_data['q_pred'][:,:,i]
#             v_pred_i = plot_data['v_pred'][:,:,i]
#             u_pred_i = plot_data['u_pred'][:,:,i]

#             # For each planning step in the trajectory
#             for j in range(0, N_plan, pred_plot_sampling):
#                 # Receding horizon = [j,j+N_h]
#                 t0_horizon = j*dt_plan
#                 tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
#                 tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
#                 # Set up lists of (x,y) points for predicted positions and velocities
#                 points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 # Set up lists of segments
#                 segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
#                 segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
#                 segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
#                 # Make collections segments
#                 cm = plt.get_cmap('Greys_r') 
#                 lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
#                 lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
#                 lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
#                 lc_q.set_array(tspan_x_pred)
#                 lc_v.set_array(tspan_x_pred) 
#                 lc_u.set_array(tspan_u_pred)
#                 # Customize
#                 lc_q.set_linestyle('-')
#                 lc_v.set_linestyle('-')
#                 lc_u.set_linestyle('-')
#                 lc_q.set_linewidth(1)
#                 lc_v.set_linewidth(1)
#                 lc_u.set_linewidth(1)
#                 # Plot collections
#                 ax_x[i,0].add_collection(lc_q)
#                 ax_x[i,1].add_collection(lc_v)
#                 ax_u[i].add_collection(lc_u)
#                 # Scatter to highlight points
#                 colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
#                 my_colors = cm(colors)
#                 ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
#                 ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
#                 ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

#         # Joint position
#           # Desired
#         ax_x[i,0].plot(t_span_plan_x, plot_data['q_des'][:,i], 'b-', label='Desired')
#           # Measured
#         ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
#         ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)', linewidth=2)
#         ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{}$ (rad)')
#         ax_x[i,0].grid()
#         # ax_x[i,0].set_ylim(x0[i]-0.1, x0[i]+0.1)
        
#         # Joint velocity 
#           # Desired 
#         ax_x[i,1].plot(t_span_plan_x, plot_data['v_des'][:,i], 'b-', label='Desired')
#           # Measured 
#         ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
#         ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)')
#         ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{}$ (rad/s)')
#         ax_x[i,1].grid()
#         # ax_x[i,1].set_ylim(x0[nq+i]-0.1, x0[nq+i]+0.1)
        
#         # Joint torques
#           # Desired  
#         ax_u[i].plot(t_span_plan_u, plot_data['u_des'][:,i], 'b-', label='Desired')
#           # Measured
#         ax_u[i].plot(t_span_simu_u, plot_data['u_mea'][:,i], 'r-', label='Measured') 
#         ax_u[i].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)')
#         ax_u[i].grid()
#         # ax_u[i].set_ylim(u_min[i], u_max[i])

#         # Desired joint torque (interpolated feedforward)
#         ax_a[i,0].plot(t_span_ctrl_u, plot_data['a_err'][:,i], 'b-', label='Velocity error (average)')
#         # Total
#         ax_a[i,0].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)')
#         ax_a[i,0].grid()

#         # Desired joint torque (interpolated feedforward)
#         ax_a[i,1].plot(t_span_ctrl_u, plot_data['a_err'][:,nq+i], 'b-', label='Acceleration error (average)')
#         # Total
#         ax_a[i,1].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)')
#         ax_a[i,1].grid()

#         # Legend
#         handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
#         fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

#         handles_u, labels_u = ax_u[i].get_legend_handles_labels()
#         fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})

#     # Plot endeff
#     # x
#     ax_p[0].plot(t_span_plan_x, plot_data['p_des'][:,0]-p_ref[0], 'b-', label='p_des - p_ref', alpha=0.5)
#     ax_p[0].plot(t_span_simu_x, plot_data['p_mea'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (WITH noise)', linewidth=1, alpha=0.3)
#     ax_p[0].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (NO noise)', linewidth=2)
#     ax_p[0].set_title('x-position-ERROR')
#     # ax_p[0].set_ylim(-ax_p_ylim, ax_p_ylim) #delta_px, p_ref[0]+delta_px
#     ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
#     # 
#     ax_p[0].grid()
#     # y
#     ax_p[1].plot(t_span_plan_x, plot_data['p_des'][:,1]-p_ref[1], 'b-', label='py_des - py_ref', alpha=0.5)
#     ax_p[1].plot(t_span_simu_x, plot_data['p_mea'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (WITH noise)', linewidth=1, alpha=0.3)
#     ax_p[1].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (NO noise)', linewidth=2)
#     ax_p[1].set_title('y-position-ERROR')
#     # ax_p[1].set_ylim(-ax_p_ylim, ax_p_ylim)
#     ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
#     ax_p[1].grid()
#     # z
#     ax_p[2].plot(t_span_plan_x, plot_data['p_des'][:,2]-p_ref[2], 'b-', label='pz_des - pz_ref', alpha=0.5)
#     ax_p[2].plot(t_span_simu_x, plot_data['p_mea'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (WITH noise)', linewidth=1, alpha=0.3)
#     ax_p[2].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (NO noise)', linewidth=2)
#     ax_p[2].set_title('z-position-ERROR')
#     # ax_p[2].set_ylim(-ax_p_ylim, ax_p_ylim)
#     ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
#     ax_p[2].grid()
#     # Add frame ref if any
#     ax_p[0].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', label='err=0', alpha=0.4)
#     ax_p[1].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', label='err=0', alpha=0.4)
#     ax_p[2].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', label='err=0', alpha=0.4)
#     # Set ylim if any
#     if 'ax_p_ylim' in plot_data:
#         ax_p_ylim = plot_data['ax_p_ylim']
#         ax_p[0].set_ylim(-ax_p_ylim, ax_p_ylim) #delta_px, p_ref[0]+delta_px)
#         ax_p[1].set_ylim(-ax_p_ylim, ax_p_ylim) #p_ref[1]-delta_py, p_ref[1]+delta_py)
#         ax_p[2].set_ylim(-ax_p_ylim, ax_p_ylim) #p_ref[2]-delta_pz, p_ref[2]+delta_pz)

#     if(PLOT_PREDICTIONS):
#         # For each component (x,y,z)
#         for i in range(3):
#             p_pred_i = plot_data['p_pred'][:, :, i]
#             # For each planning step in the trajectory
#             for j in range(0, N_plan, pred_plot_sampling):
#                 # Receding horizon = [j,j+N_h]
#                 t0_horizon = j*dt_plan
#                 tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
#                 # Set up lists of (x,y) points for predicted positions
#                 points_p = np.array([tspan_x_pred, p_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 # Set up lists of segments
#                 segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
#                 # Make collections segments
#                 cm = plt.get_cmap('Greys_r') 
#                 lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
#                 lc_p.set_array(tspan_x_pred)
#                 # Customize
#                 lc_p.set_linestyle('-')
#                 lc_p.set_linewidth(1)
#                 # Plot collections
#                 ax_p[i].add_collection(lc_p)
#                 # Scatter to highlight points
#                 colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
#                 my_colors = cm(colors)
#                 ax_p[i].scatter(tspan_x_pred, p_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

#     handles_p, labels_p = ax_p[0].get_legend_handles_labels()
#     fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

#     # Titles
#     fig_x.suptitle('State = joint positions, velocities', size=16)
#     fig_u.suptitle('Control = joint torques', size=16)
#     fig_p.suptitle('End-effector trajectories errors', size=16)
#     fig_a.suptitle('Average tracking errors over control cycles (1ms)', size=16)
    
#     # Save figs
#     if(SAVE):
#         figs = {'x': fig_x, 
#                 'u': fig_u,
#                 'a': fig_a,
#                 'p': fig_p}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
#     plt.close('all')

# def plot_results_from_sim_data(sim_data, PLOT_PREDICTIONS=False, pred_plot_sampling=100):
#     plot_data = extract_plot_data_from_sim_data(sim_data)
#     plot_results_from_plot_data(plot_data, PLOT_PREDICTIONS, pred_plot_sampling)

# def plot_results_from_yaml(yaml_file, PLOT_PREDICTIONS=False, pred_plot_sampling=100):
#     plot_data = extract_plot_data_from_yaml(yaml_file)
#     plot_results_from_plot_data(plot_data, PLOT_PREDICTIONS, pred_plot_sampling)

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