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

# Returns urdf path of a kuka robot 
def get_mesh_dir(robot_family='kuka'):
    # Get config file
    with importlib_resources.path("robot_properties_"+robot_family, "config.py") as p:
        pkg_dir = p.parent.absolute()
    urdf_dir = pkg_dir
    return str(urdf_dir)

# Load config file
def load_config_file(config_name):
    '''
    Loads YAML config file in demos/config as a dict
    '''
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../demos', 'config/'))
    config_file = config_path+"/"+config_name+".yml"
    config = load_yaml_file(config_file)
    return config

# # Save data (dict) into compressed npz
# def save_data(sim_data, save_name=None, save_dir=None):
#     '''
#     Saves data to a compressed npz file (binary)
#     '''
#     print('Compressing & saving data...')
#     if(save_name is None):
#         save_name = 'sim_data_NO_NAME'+str(time.time())
#     if(save_dir is None):
#         save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
#     save_path = save_dir+'/'+save_name+'.npz'
#     np.savez_compressed(save_path, data=sim_data)
#     print("Saved data to "+str(save_path)+" !")

# # Loads dict from compressed npz
# def load_data(npz_file):
#     '''
#     Loads a npz archive of sim_data into a dict
#     '''
#     print('Loading data...')
#     d = np.load(npz_file, allow_pickle=True)
#     return d['data'][()]

# # Extract MPC simu-specific plotting data from sim data
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
#     plot_data['nx'] = sim_data['nx']
#     plot_data['T_h'] = sim_data['T_h']
#     plot_data['N_h'] = sim_data['N_h']
#     plot_data['p_ref'] = sim_data['p_ref']
#     plot_data['alpha'] = sim_data['alpha']
#     plot_data['beta'] = sim_data['beta']
#     # Solver stuff
#     plot_data['K'] = sim_data['K']
#     plot_data['K_svd'] = sim_data['K_svd']
#     plot_data['Vxx_diag'] = sim_data['Vxx_diag']
#     plot_data['Vxx_eigval'] = sim_data['Vxx_eigval']
#     plot_data['J_rank'] = sim_data['J_rank']
#     plot_data['xreg'] = sim_data['xreg']
#     plot_data['ureg'] = sim_data['ureg']
#     return plot_data

# # Same giving npz path OR dict as argument
# def extract_plot_data(input_data):
#     '''
#     Extract plot data from npz archive or sim_data
#     '''
#     if(type(input_data)==str):
#         sim_data = load_yaml_file(input_data)
#     elif(type(input_data)==dict):
#         sim_data = input_data
#     else:
#         TypeError("Input data must be a Python dict or a path to .npz archive")
#     return extract_plot_data_from_sim_data(sim_data)


# from matplotlib.collections import LineCollection
# import matplotlib.pyplot as plt
# import matplotlib


# # Plot state data
# def plot_state(plot_data, PLOT_PREDICTIONS=False, 
#                           pred_plot_sampling=100, 
#                           SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                           SHOW=True):
#     '''
#     Plot state data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#     '''
#     print('Plotting state data...')
#     T_tot = plot_data['T_tot']
#     N_simu = plot_data['N_simu']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']
#     nq = plot_data['nq']
#     T_h = plot_data['T_h']
#     N_h = plot_data['N_h']
#     # Create time spans for X and U + Create figs and subplots
#     t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
#     t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
#     fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8))
#     # For each joint
#     for i in range(nq):

#         if(PLOT_PREDICTIONS):

#             # Extract state predictions of i^th joint
#             q_pred_i = plot_data['q_pred'][:,:,i]
#             v_pred_i = plot_data['v_pred'][:,:,i]

#             # For each planning step in the trajectory
#             for j in range(0, N_plan, pred_plot_sampling):
#                 # Receding horizon = [j,j+N_h]
#                 t0_horizon = j*dt_plan
#                 tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
#                 tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
#                 # Set up lists of (x,y) points for predicted positions and velocities
#                 points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 # Set up lists of segments
#                 segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
#                 segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
#                 # Make collections segments
#                 cm = plt.get_cmap('Greys_r') 
#                 lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
#                 lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
#                 lc_q.set_array(tspan_x_pred)
#                 lc_v.set_array(tspan_x_pred) 
#                 # Customize
#                 lc_q.set_linestyle('-')
#                 lc_v.set_linestyle('-')
#                 lc_q.set_linewidth(1)
#                 lc_v.set_linewidth(1)
#                 # Plot collections
#                 ax_x[i,0].add_collection(lc_q)
#                 ax_x[i,1].add_collection(lc_v)
#                 # Scatter to highlight points
#                 colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
#                 my_colors = cm(colors)
#                 ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
#                 ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',

#         # Joint position
#           # Desired
#         ax_x[i,0].plot(t_span_plan_x, plot_data['q_des'][:,i], 'b-', label='Desired')
#           # Measured
#         ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
#         ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)', linewidth=2)
#         ax_x[i,0].set(xlabel='t (s)', ylabel='$q_{}$ (rad)'.format(i))
#         ax_x[i,0].grid()
        
#         # Joint velocity 
#           # Desired 
#         ax_x[i,1].plot(t_span_plan_x, plot_data['v_des'][:,i], 'b-', label='Desired')
#           # Measured 
#         ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
#         ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)')
#         ax_x[i,1].set(xlabel='t (s)', ylabel='$v_{}$ (rad/s)'.format(i))
#         ax_x[i,1].grid()
        
#         # Legend
#         handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
#         fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
#     # Titles
#     fig_x.suptitle('State = joint positions, velocities', size=16)
#     # Save fig
#     if(SAVE):
#         figs = {'x': fig_x}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
    
#     return fig_x


# # Plot control data
# def plot_control(plot_data, PLOT_PREDICTIONS=False, 
#                             pred_plot_sampling=100, 
#                             SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                             SHOW=True,
#                             AUTOSCALE=False):
#     '''
#     Plot control data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#     '''
#     print('Plotting control data...')
#     T_tot = plot_data['T_tot']
#     N_simu = plot_data['N_simu']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']
#     dt_simu = plot_data['dt_simu']
#     nq = plot_data['nq']
#     T_h = plot_data['T_h']
#     N_h = plot_data['N_h']
#     # Create time spans for X and U + Create figs and subplots
#     t_span_simu_u = np.linspace(0, T_tot-dt_simu, N_simu)
#     t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
#     fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8))
#     # For each joint
#     for i in range(nq):

#         if(PLOT_PREDICTIONS):

#             # Extract state predictions of i^th joint
#             u_pred_i = plot_data['u_pred'][:,:,i]

#             # For each planning step in the trajectory
#             for j in range(0, N_plan, pred_plot_sampling):
#                 # Receding horizon = [j,j+N_h]
#                 t0_horizon = j*dt_plan
#                 tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
#                 # Set up lists of (x,y) points for predicted positions and velocities
#                 points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
#                 # Set up lists of segments
#                 segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
#                 # Make collections segments
#                 cm = plt.get_cmap('Greys_r') 
#                 lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
#                 lc_u.set_array(tspan_u_pred)
#                 # Customize
#                 lc_u.set_linestyle('-')
#                 lc_u.set_linewidth(1)
#                 # Plot collections
#                 ax_u[i].add_collection(lc_u)
#                 # Scatter to highlight points
#                 colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
#                 my_colors = cm(colors)
#                 ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

#         # Joint torques
#           # Desired  
#         ax_u[i].plot(t_span_plan_u, plot_data['u_des'][:,i], 'b-', label='Desired')
#           # Measured
#         ax_u[i].plot(t_span_simu_u, plot_data['u_mea'][:,i], 'r-', label='Measured') 
#         ax_u[i].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
#         ax_u[i].grid()

#         handles_u, labels_u = ax_u[i].get_legend_handles_labels()
#         fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
#     # Titles
#     fig_u.suptitle('Control = joint torques', size=16)
#     # Save figs
#     if(SAVE):
#         figs = {'u': fig_u}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 

#     return fig_u


# # Plot end-eff data
# def plot_endeff(plot_data, PLOT_PREDICTIONS=False, 
#                            pred_plot_sampling=100, 
#                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                            SHOW=True,
#                            AUTOSCALE=False):
#     '''
#     Plot endeff data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#       AUTOSCALE                 : rescale y-axis of endeff plot 
#                                   based on maximum value taken
#     '''
#     print('Plotting end-eff data...')
#     T_tot = plot_data['T_tot']
#     N_simu = plot_data['N_simu']
#     N_ctrl = plot_data['N_ctrl']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']
#     T_h = plot_data['T_h']
#     N_h = plot_data['N_h']
#     p_ref = plot_data['p_ref']
#     # Create time spans for X and U + Create figs and subplots
#     t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
#     t_span_ctrl_x = np.linspace(0, T_tot, N_ctrl+1)
#     t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
#     fig_p, ax_p = plt.subplots(3,1, figsize=(19.2,10.8)) 
#     # Plot endeff
#     # x
#     ax_p[0].plot(t_span_plan_x, plot_data['p_des'][:,0]-p_ref[0], 'b-', label='p_des - p_ref', alpha=0.5)
#     ax_p[0].plot(t_span_simu_x, plot_data['p_mea'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (WITH noise)', linewidth=1, alpha=0.3)
#     ax_p[0].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (NO noise)', linewidth=2)
#     ax_p[0].set_title('x-position-ERROR')
#     ax_p[0].set(xlabel='t (s)', ylabel='x (m)')
#     # 
#     ax_p[0].grid()
#     # y
#     ax_p[1].plot(t_span_plan_x, plot_data['p_des'][:,1]-p_ref[1], 'b-', label='py_des - py_ref', alpha=0.5)
#     ax_p[1].plot(t_span_simu_x, plot_data['p_mea'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (WITH noise)', linewidth=1, alpha=0.3)
#     ax_p[1].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (NO noise)', linewidth=2)
#     ax_p[1].set_title('y-position-ERROR')
#     ax_p[1].set(xlabel='t (s)', ylabel='y (m)')
#     ax_p[1].grid()
#     # z
#     ax_p[2].plot(t_span_plan_x, plot_data['p_des'][:,2]-p_ref[2], 'b-', label='pz_des - pz_ref', alpha=0.5)
#     ax_p[2].plot(t_span_simu_x, plot_data['p_mea'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (WITH noise)', linewidth=1, alpha=0.3)
#     ax_p[2].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (NO noise)', linewidth=2)
#     ax_p[2].set_title('z-position-ERROR')
#     ax_p[2].set(xlabel='t (s)', ylabel='z (m)')
#     ax_p[2].grid()
#     # Add frame ref if any
#     ax_p[0].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., label='err=0', alpha=0.5)
#     ax_p[1].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., label='err=0', alpha=0.5)
#     ax_p[2].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., label='err=0', alpha=0.5)
#     # Set ylim if any
#     if(AUTOSCALE):
#         ax_p_ylim = np.max(np.abs(plot_data['p_mea']-plot_data['p_ref']))
#         ax_p[0].set_ylim(-ax_p_ylim, ax_p_ylim) 
#         ax_p[1].set_ylim(-ax_p_ylim, ax_p_ylim) 
#         ax_p[2].set_ylim(-ax_p_ylim, ax_p_ylim) 

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
#     fig_p.suptitle('End-effector trajectories errors', size=16)

#     # Save figs
#     if(SAVE):
#         figs = {'p': fig_p}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
    
#     return fig_p


# # Plot acceleration error data
# def plot_acc_err(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                             SHOW=True):
#     '''
#     Plot acc err data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#     '''
#     print('Plotting acc error data...')
#     T_tot = plot_data['T_tot']
#     N_ctrl = plot_data['N_ctrl']
#     dt_ctrl = plot_data['dt_ctrl']
#     nq = plot_data['nq']
#     # Create time spans for X and U + Create figs and subplots
#     t_span_ctrl_u = np.linspace(0, T_tot-dt_ctrl, N_ctrl)
#     fig_a, ax_a = plt.subplots(nq,2, figsize=(19.2,10.8))
#     # For each joint
#     for i in range(nq):

#         # Joint velocity error (avg over 1 control cycle)
#         ax_a[i,0].plot(t_span_ctrl_u, plot_data['a_err'][:,i], 'b-', label='Velocity error (average)')
#         ax_a[i,0].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
#         ax_a[i,0].grid()
#         # Joint acceleration error (avg over 1 control cycle)
#         ax_a[i,1].plot(t_span_ctrl_u, plot_data['a_err'][:,nq+i], 'b-', label='Acceleration error (average)')
#         ax_a[i,1].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
#         ax_a[i,1].grid()
#     # title
#     fig_a.suptitle('Average tracking errors over control cycles (1ms)', size=16)
#     # Save figs
#     if(SAVE):
#         figs = {'a': fig_a}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
    
#     return fig_a


# # Plot Ricatti
# def plot_ricatti(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                             SHOW=True):
#     '''
#     Plot ricatti data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#     '''
#     print('Plotting Ricatti data...')
#     T_tot = plot_data['T_tot']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']
#     nq = plot_data['nq']

#     # Create time spans for X and U + Create figs and subplots
#     t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
#     fig_K, ax_K = plt.subplots(nq, 2, figsize=(19.2,10.8))
#     # For each joint
#     for i in range(nq):
#         # Ricatti gains diag
#         ax_K[i,0].plot(t_span_plan_u, plot_data['K'][:,i,i], 'b-', label='Diag of Ricatti')
#         ax_K[i,0].set(xlabel='t (s)', ylabel='$diag [K]_{}$'.format(i))
#         ax_K[i,0].grid()
#         # Ricatti gains singular values
#         ax_K[i,1].plot(t_span_plan_u, plot_data['K_svd'][:,i], 'b-', label='Singular Value of Ricatti')
#         ax_K[i,1].set(xlabel='t (s)', ylabel='$\sigma [K]_{}$'.format(i))
#         ax_K[i,1].grid()
#     # Titles
#     fig_K.suptitle('Singular Values of Ricatti feedback gains K', size=16)
#     # Save figs
#     if(SAVE):
#         figs = {'K': fig_K}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
    
#     return fig_K


# # Plot Vxx
# def plot_Vxx(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                         SHOW=True):
#     '''
#     Plot Vxx data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#     '''
#     print('Plotting Vxx data...')
#     T_tot = plot_data['T_tot']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']
#     nq = plot_data['nq']

#     # Create time spans for X and U + Create figs and subplots
#     t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
#     fig_V, ax_V = plt.subplots(nq, 2, figsize=(19.2,10.8))
#     # For each state
#     for i in range(nq):
#         # Vxx diag
#         # ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_diag'][:,i], 'b-', label='Vxx diagonal')
#         # ax_V[i,0].set(xlabel='t (s)', ylabel='$Diag[Vxx_{}]$'.format(i,i))
#         # ax_V[i,0].grid()
#         # Vxx eigenvals
#         ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_eigval'][:,i], 'b-', label='Vxx eigenvalue')
#         ax_V[i,0].set(xlabel='t (s)', ylabel='$\lambda_{}$'.format(i)+'(Vxx)')
#         ax_V[i,0].grid()
#         # Vxx eigenvals
#         ax_V[i,1].plot(t_span_plan_u, plot_data['Vxx_eigval'][:,nq+i], 'b-', label='Vxx eigenvalue')
#         ax_V[i,1].set(xlabel='t (s)', ylabel='$\lambda_{}$'.format(nq+i)+'(Vxx)')
#         ax_V[i,1].grid()
#     # Titles
#     fig_V.suptitle('Eigenvalues of Value Function Hessian Vxx', size=16)
#     # Save figs
#     if(SAVE):
#         figs = {'V': fig_V}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
    
#     return fig_V


# # Plot Solver regs
# def plot_solver(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                            SHOW=True):
#     '''
#     Plot solver data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#     '''
#     print('Plotting solver data...')
#     T_tot = plot_data['T_tot']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']

#     # Create time spans for X and U + Create figs and subplots
#     t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
#     fig_S, ax_S = plt.subplots(2, 1, figsize=(19.2,10.8))
#     # Xreg
#     ax_S[0].plot(t_span_plan_u, plot_data['xreg'], 'b-', label='xreg')
#     ax_S[0].set(xlabel='t (s)', ylabel='$xreg$')
#     ax_S[0].grid()
#     # Ureg
#     ax_S[1].plot(t_span_plan_u, plot_data['ureg'], 'r-', label='ureg')
#     ax_S[1].set(xlabel='t (s)', ylabel='$ureg$')
#     ax_S[1].grid()
#     # Titles
#     fig_S.suptitle('FDDP solver regularization on x (Vxx diag) and u (Quu diag)', size=16)
#     # Save figs
#     if(SAVE):
#         figs = {'S': fig_S}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
    
#     return fig_S


# # Plot rank of Jacobian
# def plot_jacobian(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                              SHOW=True):
#     '''
#     Plot jacobian data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#     '''
#     print('Plotting solver data...')
#     T_tot = plot_data['T_tot']
#     N_plan = plot_data['N_plan']
#     dt_plan = plot_data['dt_plan']

#     # Create time spans for X and U + Create figs and subplots
#     t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
#     fig_J, ax_J = plt.subplots(1, 1, figsize=(19.2,10.8))
#     # Rank of Jacobian
#     ax_J.plot(t_span_plan_u, plot_data['J_rank'], 'b-', label='rank')
#     ax_J.set(xlabel='t (s)', ylabel='rank')
#     ax_J.grid()
#     # Titles
#     fig_J.suptitle('Rank of Jacobian J(q)', size=16)
#     # Save figs
#     if(SAVE):
#         figs = {'J': fig_J}
#         if(SAVE_DIR is None):
#             SAVE_DIR = '/home/skleff/force-feedback/data'
#         if(SAVE_NAME is None):
#             SAVE_NAME = 'testfig'
#         for name, fig in figs.items():
#             fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
#     if(SHOW):
#         plt.show() 
    
#     return fig_J


# # Plot data
# def plot_results(plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
#                                               pred_plot_sampling=100, 
#                                               SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
#                                               SHOW=True,
#                                               AUTOSCALE=False):
#     '''
#     Plot sim data
#      Input:
#       plot_data                 : plotting data
#       PLOT_PREDICTIONS          : True or False
#       pred_plot_sampling        : plot every pred_plot_sampling prediction 
#                                   to avoid huge amount of plotted data 
#                                   ("1" = plot all)
#       SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
#       SHOW                      : show plots
#       AUTOSCALE                 : rescale y-axis of endeff plot 
#                                   based on maximum value taken
#     '''

#     plots = {}

#     if('x' in which_plots or which_plots is None or which_plots =='all'):
#         plots['x'] = plot_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
#                                            pred_plot_sampling=pred_plot_sampling, 
#                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                            SHOW=False)
    
#     if('u' in which_plots or which_plots is None or which_plots =='all'):
#         plots['u'] = plot_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
#                                              pred_plot_sampling=pred_plot_sampling, 
#                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                              SHOW=False)
    
#     if('a' in which_plots or which_plots is None or which_plots =='all'):
#         plots['a'] = plot_acc_err(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                              SHOW=SHOW)

#     if('p' in which_plots or which_plots is None or which_plots =='all'):
#         plots['p'] = plot_endeff(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
#                                             pred_plot_sampling=pred_plot_sampling, 
#                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                             SHOW=False, AUTOSCALE=AUTOSCALE)

#     if('K' in which_plots or which_plots is None or which_plots =='all'):
#         plots['K'] = plot_ricatti(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                              SHOW=False)

#     if('V' in which_plots or which_plots is None or which_plots =='all'):
#         plots['V'] = plot_Vxx(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                          SHOW=False)

#     if('S' in which_plots or which_plots is None or which_plots =='all'):
#         plots['S'] = plot_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                             SHOW=False)

#     if('J' in which_plots or which_plots is None or which_plots =='all'):
#         plots['J'] = plot_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
#                                               SHOW=False)

#     if(SHOW):
#         plt.show() 
#     plt.close('all')



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