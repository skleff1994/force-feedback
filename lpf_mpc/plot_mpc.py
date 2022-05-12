"""
@package force_feedback
@file lpf_mpc/plot_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Plot MPC solution
"""

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from classical_mpc import *

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


### Plot from MPC simulation (LPF)

# Plot data
def plot_mpc_results_LPF(plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
                                              pred_plot_sampling=100, 
                                              SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                              SHOW=True,
                                              AUTOSCALE=False):
    '''
    Plot sim data (MPC simulation using LPF, i.e. state y = (q,v,tau))
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

    figs = {}; axes = {}

    if('y' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        figs['y'], axes['y'] = plot_mpc_state_LPF(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                           pred_plot_sampling=pred_plot_sampling, 
                                           SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                           SHOW=False)
    
    if('w' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        figs['w'], axes['w'] = plot_mpc_control_LPF(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                             pred_plot_sampling=pred_plot_sampling, 
                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False)

    if('ee' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        figs['ee_lin'], axes['ee_lin'] = plot_mpc_endeff_linear_LPF(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)
        figs['ee_ang'], axes['ee_ang'] = plot_mpc_endeff_angular_LPF(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)

    if('f' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        figs['f'], axes['f'] = plot_mpc_force_LPF(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)

    if('K' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('K_diag' in plot_data.keys()):
            figs['K_diag'], axes['K_diag'] = plot_mpc_ricatti_diag_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)
        if('K_svd' in plot_data.keys()):
            figs['K_svd'], axes['K_svd'] = plot_mpc_ricatti_svd_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('V' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('V_diag' in plot_data.keys()):
            figs['V_diag'], axes['V_diag'] = plot_mpc_Vxx_diag_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('V_eig' in plot_data.keys()):
            figs['V_eig'], axes['V_eig'] = plot_mpc_Vxx_eig_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)

    if('S' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('S' in plot_data.keys()):
            figs['S'], axes['S'] = plot_mpc_solver_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('J' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('J' in plot_data.keys()):
            figs['J'], axes['J'] = plot_mpc_jacobian_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('Q' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('Q_diag' in plot_data.keys()):
            figs['Q_diag'], axes['Q_diag'] = plot_mpc_Quu_diag_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('Q_eig' in plot_data.keys()):
            figs['Q_eig'], axes['Q_eig'] = plot_mpc_Quu_eig_LPF(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
    
    if(SHOW):
        plt.show() 
    
    return figs, axes
    # plt.close('all')

# Plot state data
def plot_mpc_state_LPF(plot_data, PLOT_PREDICTIONS=False, 
                                  pred_plot_sampling=100, 
                                  SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                  SHOW=True):
    '''
    Plot state data (MPC simulation using LPF, i.e. state x = (q,v,tau))
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    logger.info('Plotting state data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']
    nv = plot_data['nv']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot, N_simu+1)
    t_span_ctrl = np.linspace(0, T_tot, N_ctrl+1)
    t_span_plan = np.linspace(0, T_tot, N_plan+1)
    fig, ax = plt.subplots(nq, 3, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            q_pred_i = plot_data['q_pred'][:,:,i]
            v_pred_i = plot_data['v_pred'][:,:,i]
            tau_pred_i = plot_data['tau_pred'][:,:,i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_tau = np.array([tspan_x_pred, tau_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                segs_tau= np.concatenate([points_tau[:-1], points_tau[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_tau = LineCollection(segs_tau, cmap=cm, zorder=-1)
                lc_q.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred) 
                lc_tau.set_array(tspan_x_pred)
                # Customize
                lc_q.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_tau.set_linestyle('-')
                lc_q.set_linewidth(1)
                lc_v.set_linewidth(1)
                lc_tau.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_q)
                ax[i,1].add_collection(lc_v)
                ax[i,2].add_collection(lc_tau)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                ax[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
                ax[i,2].scatter(tspan_x_pred, tau_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 

        # Joint position
        ax[i,0].plot(t_span_plan, plot_data['q_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,0].plot(t_span_ctrl, plot_data['q_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL rate)', alpha=0.3)
        # ax[i,0].plot(t_span_simu, plot_data['q_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,0].plot(t_span_simu, plot_data['q_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        if('stateReg' in plot_data['WHICH_COSTS']):
            ax[i,0].plot(t_span_plan[:-1], plot_data['state_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', marker=None, label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        
        # Joint velocity 
        ax[i,1].plot(t_span_plan, plot_data['v_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,1].plot(t_span_ctrl, plot_data['v_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL)', alpha=0.3)
        # ax[i,1].plot(t_span_simu, plot_data['v_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,1].plot(t_span_simu, plot_data['v_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        if('stateReg' in plot_data['WHICH_COSTS']):
            ax[i,1].plot(t_span_plan[:-1], plot_data['state_ref'][:,i+nq], color=[0.,1.,0.,0.], linestyle='-.', marker=None, label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)

        # Joint torques
        ax[i,2].plot(t_span_plan, plot_data['tau_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,2].plot(t_span_ctrl, plot_data['tau_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL rate)', alpha=0.3)
        # ax[i,2].plot(t_span_simu, plot_data['tau_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,2].plot(t_span_simu, plot_data['tau_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,2].plot(t_span_simu, plot_data['tau_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        if('ctrlReg' or 'ctrlRegGrav' in plot_data['WHICH_COSTS']):
            ax[i,2].plot(t_span_plan[:-1], plot_data['ctrl_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', marker=None, label='Reference', alpha=0.9)
        # ax[i,2].plot(t_span_simu, plot_data['grav'][:,i], color='k', marker=None, linestyle='-.', label='Reg (grav)', alpha=0.6)
        ax[i,2].set_ylabel('$\\tau{}$'.format(i), fontsize=12)
        ax[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,2].grid(True)

        # Add xlabel on bottom plot of each column
        if(i == nq-1):
            ax[i,0].set_xlabel('t(s)', fontsize=16)
            ax[i,1].set_xlabel('t(s)', fontsize=16)
            ax[i,2].set_xlabel('t(s)', fontsize=16)
        # Legend
        handles_x, labels_x = ax[i,0].get_legend_handles_labels()
        fig.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
    TOL = 1e-5; 
    for i in range(nq):
        ax_q_ylim = 1.1*max(np.max(np.abs(plot_data['q_mea_no_noise'][:,i])), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['v_mea_no_noise'][:,i])), TOL)
        ax_tau_ylim = 1.1*max(np.max(np.abs(plot_data['tau_mea_no_noise'][:,i])), TOL)
        ax[i,0].set_ylim(-ax_q_ylim, ax_q_ylim) 
        ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 
        ax[i,2].set_ylim(-ax_tau_ylim, ax_tau_ylim) 

    # y axis labels
    fig.text(0.06, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.345, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.625, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=12)
    fig.subplots_adjust(wspace=0.37)
    # Titles
    fig.suptitle('State = joint position ($q$), velocity ($v$), torque ($\\tau$)', size=18)
    # Save fig
    if(SAVE):
        figs = {'x': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig, ax

# Plot control data
def plot_mpc_control_LPF(plot_data, PLOT_PREDICTIONS=False, 
                                    pred_plot_sampling=100, 
                                    SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                    SHOW=True):
    '''
    Plot control data (MPC simulation using LPF, i.e. control u = unfiltered torque)
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    logger.info('Plotting control data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    dt_simu = plot_data['dt_simu']
    dt_ctrl = plot_data['dt_ctrl']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot-dt_simu, N_simu)
    t_span_ctrl = np.linspace(0, T_tot-dt_ctrl, N_ctrl)
    t_span_plan = np.linspace(0, T_tot-dt_plan, N_plan)
    fig, ax = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            u_pred_i = plot_data['w_pred'][:,:,i]

            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
                lc_u.set_array(tspan_u_pred)
                # Customize
                lc_u.set_linestyle('-')
                lc_u.set_linewidth(1)
                # Plot collections
                ax[i].add_collection(lc_u)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

        # Joint torques
        ax[i].plot(t_span_plan, plot_data['w_pred'][:,0,i], color='r', marker=None, linestyle='-', label='Optimal control w0*', alpha=0.6)
        ax[i].plot(t_span_plan, plot_data['w_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.1)
        # ax[i].plot(t_span_ctrl, plot_data['w_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Prediction (CTRL)', alpha=0.6)
        # ax[i].plot(t_span_simu, plot_data['w_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Prediction (SIMU)', alpha=0.6)
        ax[i].plot(t_span_simu, plot_data['grav'][:-1,i], color=[0.,1.,0.,0.], marker=None, linestyle='-.', label='Reg reference (grav)', alpha=0.9)
        ax[i].set_ylabel('$u_{}$'.format(i), fontsize=12)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i].grid(True)
        # Last x axis label
        if(i == nq-1):
            ax[i].set_xlabel('t (s)', fontsize=16)
        # LEgend
        handles_u, labels_u = ax[i].get_legend_handles_labels()
        fig.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
    TOL = 1e-5
    for i in range(nq):
        ax_u_ylim = 1.1*max(np.max(np.abs(plot_data['w_pred'][:,0,i])), TOL)
        ax[i].set_ylim(-ax_u_ylim, ax_u_ylim) 
    # Sup-y label
    fig.text(0.04, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=16)
    # Titles
    fig.suptitle('Control = unfiltered joint torques', size=18)
    # Save figs
    if(SAVE):
        figs = {'u': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 

    return fig, ax

# Plot end-eff data (linear)
def plot_mpc_endeff_linear_LPF(plot_data, PLOT_PREDICTIONS=False, 
                                   pred_plot_sampling=100, 
                                   SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                   SHOW=True,
                                   AUTOSCALE=False):
    '''
    Plot endeff data (linear)
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
    return plot_mpc_endeff_linear(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                      pred_plot_sampling=pred_plot_sampling, 
                                      SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                      SHOW=SHOW, AUTOSCALE=AUTOSCALE)

# Plot end-eff data (angular)
def plot_mpc_endeff_angular_LPF(plot_data, PLOT_PREDICTIONS=False, 
                                   pred_plot_sampling=100, 
                                   SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                   SHOW=True,
                                   AUTOSCALE=False):
    '''
    Plot endeff data (angular)
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
    return plot_mpc_endeff_angular(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                      pred_plot_sampling=pred_plot_sampling, 
                                      SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                      SHOW=SHOW, AUTOSCALE=AUTOSCALE)

# Plot end-eff data
def plot_mpc_force_LPF(plot_data, PLOT_PREDICTIONS=False, 
                                   pred_plot_sampling=100, 
                                   SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                   SHOW=True,
                                   AUTOSCALE=False):
    '''
    Plot force data
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
    return plot_mpc_force(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                     pred_plot_sampling=pred_plot_sampling, 
                                     SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                     SHOW=SHOW, AUTOSCALE=AUTOSCALE)

# Plot Ricatti SVD
def plot_mpc_ricatti_svd_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True):
    '''
    Plot ricatti data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    return plot_mpc_ricatti_svd(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME, SHOW=SHOW)

# Plot Ricatti Diagonal
def plot_mpc_ricatti_diag_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True):
    '''
    Plot ricatti data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    logger.info('Plotting Ricatti diagonal...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_K, ax_K = plt.subplots(nq, 3, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):
        # Diagonal terms
        ax_K[i,0].plot(t_span_plan_u, plot_data['Kp_diag'][:, 0, i], 'b-', label='Diag of Ricatti (Kp)')
        ax_K[i,0].set_ylabel('$Kp_{}$'.format(i)+"$_{}$".format(i), fontsize=12)
        ax_K[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_K[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_K[i,0].grid(True)
        # Diagonal terms
        ax_K[i,1].plot(t_span_plan_u, plot_data['Kv_diag'][:, 0, i], 'b-', label='Diag of Ricatti (Kv)')
        ax_K[i,1].set_ylabel('$Kv_{}$'.format(i)+"$_{}$".format(i), fontsize=12)
        ax_K[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_K[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_K[i,1].grid(True)
        # Diagonal terms
        ax_K[i,2].plot(t_span_plan_u, plot_data['Ktau_diag'][:, 0, i], 'b-', label='Diag of Ricatti (K\\tau)')
        ax_K[i,2].set_ylabel('$K\\tau_{}$'.format(i)+"$_{}$".format(i), fontsize=12)
        ax_K[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_K[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_K[i,2].grid(True)

    # labels and stuff
    ax_K[-1,0].set_xlabel('t (s)', fontsize=16)
    ax_K[-1,1].set_xlabel('t (s)', fontsize=16)
    ax_K[-1,2].set_xlabel('t (s)', fontsize=16)
    ax_K[0,0].set_title('$K_p$', fontsize=16)
    ax_K[0,1].set_title('$K_v$', fontsize=16)
    ax_K[0,2].set_title('$K_\\tau$', fontsize=16)
    # Titles
    fig_K.suptitle('Diagonal Ricatti feedback gains K', size=16)
    # Save figs
    if(SAVE):
        figs = {'K_diag': fig_K}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_K

# Plot Vxx eig
def plot_mpc_Vxx_eig_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                        SHOW=True):
    '''
    Plot Vxx eigenvalues
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    logger.info('Plotting Vxx eigenvalues...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_V, ax_V = plt.subplots(nq, 3, figsize=(19.2,10.8), sharex='col') 
    # For each state
    for i in range(nq):
        # Vxx eigenvals
        ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_eig'][:, 0, i], 'b-', label='Vxx eigenvalue')
        ax_V[i,0].set_ylabel('$\lambda_{}$'.format(i), fontsize=12)
        ax_V[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_V[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_V[i,0].grid(True)
        # Vxx eigenvals
        ax_V[i,1].plot(t_span_plan_u, plot_data['Vxx_eig'][:, 0, nq+i], 'b-', label='Vxx eigenvalue')
        ax_V[i,1].set_ylabel('$\lambda_{%s}$'%str(nq+i), fontsize=12)
        ax_V[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_V[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_V[i,1].grid(True)
        # Vxx eigenvals
        ax_V[i,2].plot(t_span_plan_u, plot_data['Vxx_eig'][:, 0, nq+nq+i], 'b-', label='Vxx eigenvalue')
        ax_V[i,2].set_ylabel('$\lambda_{%s}$'%str(nq+nq+i), fontsize=12)
        ax_V[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_V[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_V[i,2].grid(True)
    # labels and stuff
    ax_V[-1,0].set_xlabel('t (s)', fontsize=16)
    ax_V[-1,1].set_xlabel('t (s)', fontsize=16)
    ax_V[-1,2].set_xlabel('t (s)', fontsize=16)
    ax_V[0,0].set_title('$Vxx_q$', fontsize=16)
    ax_V[0,1].set_title('$Vxx_v$', fontsize=16)
    ax_V[0,2].set_title('$Vxx_\\tau$', fontsize=16)
    fig_V.suptitle('Eigenvalues of Value Function Hessian Vxx', size=16)
    # Save figs
    if(SAVE):
        figs = {'V_eig': fig_V}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_V

# Plot Vxx diag
def plot_mpc_Vxx_diag_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                        SHOW=True):
    '''
    Plot Vxx diagonal terms
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    logger.info('Plotting Vxx diagonal...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_V, ax_V = plt.subplots(nq, 3, figsize=(19.2,10.8), sharex='col') 
    # For each state
    for i in range(nq):
        # Vxx diag
        ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_diag'][:, 0, i], 'b-', label='Vxx diagonal')
        ax_V[i,0].set_ylabel('$Vxx_{}$'.format(i), fontsize=12)
        ax_V[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_V[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_V[i,0].grid(True)
        # Vxx diag
        ax_V[i,1].plot(t_span_plan_u, plot_data['Vxx_diag'][:, 0, nq+i], 'b-', label='Vxx diagonal')
        ax_V[i,1].set_ylabel('$Vxx_{%s}$'%str(nq+i), fontsize=12)
        ax_V[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_V[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_V[i,1].grid(True)
        # Vxx diag
        ax_V[i,2].plot(t_span_plan_u, plot_data['Vxx_diag'][:, 0, nq+nq+i], 'b-', label='Vxx diagonal')
        ax_V[i,2].set_ylabel('$Vxx_{%s}$'%str(nq+nq+i), fontsize=12)
        ax_V[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_V[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_V[i,2].grid(True)
    # labels and stuff
    ax_V[-1,0].set_xlabel('t (s)', fontsize=16)
    ax_V[-1,1].set_xlabel('t (s)', fontsize=16)
    ax_V[-1,2].set_xlabel('t (s)', fontsize=16)
    ax_V[0,0].set_title('$Diag Vxx_q$', fontsize=16)
    ax_V[0,1].set_title('$Diag Vxx_v$', fontsize=16)
    ax_V[0,2].set_title('$Diag Vxx_\\tau$', fontsize=16) 
    # Titles
    fig_V.suptitle('Diagonal of Value Function Hessian Vxx', size=16)
    # Save figs
    if(SAVE):
        figs = {'V_diag': fig_V}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_V

# Plot Quu eig
def plot_mpc_Quu_eig_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                        SHOW=True):
    '''
    Plot Quu eigenvalues
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    return plot_mpc_Quu_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME, SHOW=SHOW)

# Plot Quu diag
def plot_mpc_Quu_diag_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                        SHOW=True):
    '''
    Plot Quu diagonal terms
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    return plot_mpc_Quu_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME, SHOW=SHOW)

# Plot Solver regs
def plot_mpc_solver_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                           SHOW=True):
    '''
    Plot solver data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    return plot_mpc_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME, SHOW=SHOW)

# Plot rank of Jacobian
def plot_mpc_jacobian_LPF(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                             SHOW=True):
    '''
    Plot jacobian data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    return plot_mpc_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME, SHOW=SHOW)
