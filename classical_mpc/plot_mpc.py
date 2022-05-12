"""
@package force_feedback
@file classical_mpc/plot_mpc.py
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

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


### Plot from MPC simulation (regular i.e. 'impedance_mpc' repo)
# Plot data
def plot_mpc_results(plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
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

    plots = {}

    if('x' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['x'] = plot_mpc_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                           pred_plot_sampling=pred_plot_sampling, 
                                           SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                           SHOW=False)
    
    if('u' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['u'] = plot_mpc_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                             pred_plot_sampling=pred_plot_sampling, 
                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False)

    if('ee' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['ee_lin'] = plot_mpc_endeff_linear(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)
        plots['ee_ang'] = plot_mpc_endeff_angular(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)

    if('f' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['f'] = plot_mpc_force(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)


    if('K' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('K_diag' in plot_data.keys()):
            plots['K_diag'] = plot_mpc_ricatti_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)
        if('K_svd' in plot_data.keys()):
            plots['K_svd'] = plot_mpc_ricatti_svd(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('V' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('V_diag' in plot_data.keys()):
            plots['V_diag'] = plot_mpc_Vxx_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('V_eig' in plot_data.keys()):
            plots['V_eig'] = plot_mpc_Vxx_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)

    if('S' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('S' in plot_data.keys()):
            plots['S'] = plot_mpc_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('J' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('J' in plot_data.keys()):
            plots['J'] = plot_mpc_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('Q' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        if('Q_diag' in plot_data.keys()):
            plots['Q_diag'] = plot_mpc_Quu_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('Q_eig' in plot_data.keys()):
            plots['Q_eig'] = plot_mpc_Quu_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
    
    if(SHOW):
        plt.show() 
    plt.close('all')

# Plot state data
def plot_mpc_state(plot_data, PLOT_PREDICTIONS=False, 
                          pred_plot_sampling=100, 
                          SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                          SHOW=True):
    '''
    Plot state data
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
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']
    nx = plot_data['nx']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot, N_simu+1)
    t_span_plan = np.linspace(0, T_tot, N_plan+1)
    fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            q_pred_i = plot_data['q_pred'][:,:,i]
            v_pred_i = plot_data['v_pred'][:,:,i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_q.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred) 
                # Customize
                lc_q.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_q.set_linewidth(1)
                lc_v.set_linewidth(1)
                # Plot collections
                ax_x[i,0].add_collection(lc_q)
                ax_x[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',

        # Joint position
        ax_x[i,0].plot(t_span_plan, plot_data['q_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,0].plot(t_span_ctrl, plot_data['q_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL rate)', alpha=0.3)
        # ax_x[i,0].plot(t_span_simu, plot_data['q_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax_x[i,0].plot(t_span_simu, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,0].plot(t_span_simu, plot_data['q_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        # Plot joint position regularization reference
        if('stateReg' in plot_data['WHICH_COSTS']):
            ax_x[i,0].plot(t_span_plan[:-1], plot_data['state_ref'][:, i], linestyle='-.', color='k', marker=None, label='xReg_ref', alpha=0.5)
        ax_x[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax_x[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,0].grid(True)

        # Joint velocity 
        ax_x[i,1].plot(t_span_plan, plot_data['v_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN)', alpha=0.5)
        # ax[i,1].plot(t_span_ctrl, plot_data['v_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL)', alpha=0.3)
        # ax_x[i,1].plot(t_span_simu, plot_data['v_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU)', alpha=0.5)
        ax_x[i,1].plot(t_span_simu, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,1].plot(t_span_simu, plot_data['v_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        if('stateReg' in plot_data['WHICH_COSTS']):
            ax_x[i,1].plot(t_span_plan[:-1], plot_data['state_ref'][:, i+nq], linestyle='-.', color='k', marker=None, label='xReg_ref', alpha=0.5)
        ax_x[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
        ax_x[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,1].grid(True)

        # Add xlabel on bottom plot of each column
        if(i == nq-1):
            ax_x[i,0].set_xlabel('t(s)', fontsize=16)
            ax_x[i,1].set_xlabel('t(s)', fontsize=16)
        # Legend
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
    # y axis labels
    fig_x.text(0.05, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=16)
    fig_x.text(0.49, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=16)
    fig_x.subplots_adjust(wspace=0.27)
    # Titles
    fig_x.suptitle('State = joint positions, velocities', size=18)
    # Save fig
    if(SAVE):
        figs = {'x': fig_x}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_x

# Plot control data
def plot_mpc_control(plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True,
                            AUTOSCALE=False):
    '''
    Plot control data
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
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    dt_simu = plot_data['dt_simu']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot-dt_simu, N_simu)
    t_span_plan = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            u_pred_i = plot_data['u_pred'][:,:,i]

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
                ax_u[i].add_collection(lc_u)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

        # Joint torques
        ax_u[i].plot(t_span_plan, plot_data['u_pred'][:,0,i], color='r', marker=None, linestyle='-', label='Optimal control u0*', alpha=0.6)
        ax_u[i].plot(t_span_plan, plot_data['u_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.1)
        # ax[i].plot(t_span_ctrl, plot_data['w_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Prediction (CTRL)', alpha=0.6)
        # ax_u[i].plot(t_span_simu, plot_data['u_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Prediction (SIMU)', alpha=0.6)
        ax_u[i].plot(t_span_simu, plot_data['grav'][:-1,i], color=[0.,1.,0.,0.], marker=None, linestyle='-.', label='Reg', alpha=0.9)
        # Plot reference
        if('ctrlReg' or 'ctrlRegGrav' in plot_data['WHICH_COSTS']):
            ax_u[i].plot(t_span_plan, plot_data['ctrl_ref'][:, i], linestyle='-.', color='k', marker=None, label='uReg_ref', alpha=0.5)
        ax_u[i].set_ylabel('$u_{}$'.format(i), fontsize=12)
        ax_u[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_u[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_u[i].grid(True)
        # Last x axis label
        if(i == nq-1):
            ax_u[i].set_xlabel('t (s)', fontsize=16)
        # LEgend
        handles_u, labels_u = ax_u[i].get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
    # Sup-y label
    fig_u.text(0.04, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=16)
    # Titles
    fig_u.suptitle('Control = joint torques', size=18)
    # Save figs
    if(SAVE):
        figs = {'u': fig_u}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 

    return fig_u

# Plot end-eff data
def plot_mpc_endeff_linear(plot_data, PLOT_PREDICTIONS=False, 
                               pred_plot_sampling=100, 
                               SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                               SHOW=True,
                               AUTOSCALE=False):
    '''
    Plot endeff data (linear position and velocity)
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
    logger.info('Plotting end-eff data (linear)...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot, N_simu+1)
    t_span_ctrl = np.linspace(0, T_tot, N_ctrl+1)
    t_span_plan = np.linspace(0, T_tot, N_plan+1)
    fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
    # Plot endeff
    xyz = ['x', 'y', 'z']
    for i in range(3):

        if(PLOT_PREDICTIONS):
            lin_pos_ee_pred_i = plot_data['lin_pos_ee_pred'][:, :, i]
            lin_vel_ee_pred_i = plot_data['lin_vel_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, lin_pos_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, lin_vel_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_p.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred)
                # Customize
                lc_p.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_p.set_linewidth(1)
                lc_v.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_p)
                ax[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, lin_pos_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                ax[i,1].scatter(tspan_x_pred, lin_vel_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

        # EE position
        ax[i,0].plot(t_span_plan, plot_data['lin_pos_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,0].plot(t_span_ctrl, plot_data['lin_pos_ee_des_CTRL'][:,i] , 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax[i,0].plot(t_span_simu, plot_data['lin_pos_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['lin_pos_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,0].plot(t_span_simu, plot_data['lin_pos_ee_mea_no_noise'][:,i], 'r-', label='measured', linewidth=2)
        # Plot reference
        if('translation' in plot_data['WHICH_COSTS']):
            ax[i,0].plot(t_span_plan[:-1], plot_data['lin_pos_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$P^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)
        
        # EE velocity
        ax[i,1].plot(t_span_plan, plot_data['lin_vel_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,1].plot(t_span_ctrl, plot_data['lin_vel_ee_des_CTRL'][:,i]-plot_data['lin_vel_ee_ref'][i], 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax[i,1].plot(t_span_simu, plot_data['lin_vel_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['lin_vel_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,1].plot(t_span_simu, plot_data['lin_vel_ee_mea_no_noise'][:,i], 'r-', label='Measured', linewidth=2)
        # Plot reference 
        if('velocity' in plot_data['WHICH_COSTS']):
            ax[i,1].plot(t_span_plan, [0.]*(N_plan+1), color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$V^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,1].grid(True)


    # Align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Set ylim if any
    TOL = 1e-3
    if(AUTOSCALE):
        ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['lin_pos_ee_mea'])), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['lin_vel_ee_mea'])), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector trajectories', size=18)
    # Save figs
    if(SAVE):
        figs = {'ee_lin': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig, ax

# Plot end-eff data
def plot_mpc_endeff_angular(plot_data, PLOT_PREDICTIONS=False, 
                               pred_plot_sampling=100, 
                               SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                               SHOW=True,
                               AUTOSCALE=False):
    '''
    Plot endeff data (angular position and velocity)
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
    logger.info('Plotting end-eff data (angular)...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot, N_simu+1)
    t_span_ctrl = np.linspace(0, T_tot, N_ctrl+1)
    t_span_plan = np.linspace(0, T_tot, N_plan+1)
    fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
    # Plot endeff
    xyz = ['x', 'y', 'z']
    for i in range(3):

        if(PLOT_PREDICTIONS):
            ang_pos_ee_pred_i = plot_data['ang_pos_ee_pred'][:, :, i]
            ang_vel_ee_pred_i = plot_data['ang_vel_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, ang_pos_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, ang_vel_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_p.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred)
                # Customize
                lc_p.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_p.set_linewidth(1)
                lc_v.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_p)
                ax[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, ang_pos_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                ax[i,1].scatter(tspan_x_pred, ang_vel_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

        # EE position
        ax[i,0].plot(t_span_plan, plot_data['ang_pos_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,0].plot(t_span_ctrl, plot_data['ang_pos_ee_des_CTRL'][:,i]-plot_data['ang_pos_ee_ref'][i], 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax[i,0].plot(t_span_simu, plot_data['ang_pos_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['ang_pos_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,0].plot(t_span_simu, plot_data['ang_pos_ee_mea_no_noise'][:,i], 'r-', label='measured', linewidth=2)
        # Plot reference
        if('rotation' in plot_data['WHICH_COSTS']):
            ax[i,0].plot(t_span_plan[:-1], plot_data['ang_pos_ee_ref'][:,i], 'm-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$RPY^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)
        
        # EE velocity
        ax[i,1].plot(t_span_plan, plot_data['ang_vel_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,1].plot(t_span_ctrl, plot_data['ang_vel_ee_des_CTRL'][:,i]-plot_data['lin_vel_ee_ref'][i], 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax[i,1].plot(t_span_simu, plot_data['ang_vel_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['ang_vel_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,1].plot(t_span_simu, plot_data['ang_vel_ee_mea_no_noise'][:,i], 'r-', label='Measured', linewidth=2)
        # Plot reference 
        if('velocity' in plot_data['WHICH_COSTS']):
            ax[i,1].plot(t_span_plan, [0.]*(N_plan+1), 'm-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$W^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,1].grid(True)


    # Align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Set ylim if any
    TOL = 1e-3
    if(AUTOSCALE):
        ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['ang_pos_ee_mea'])), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['ang_vel_ee_mea'])), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector frame orientation (RPY) and angular velocity', size=18)
    # Save figs
    if(SAVE):
        figs = {'ee_ang': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig, ax

# Plot end-eff data
def plot_mpc_force(plot_data, PLOT_PREDICTIONS=False, 
                           pred_plot_sampling=100, 
                           SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                           SHOW=True,
                           AUTOSCALE=False):
    '''
    Plot EE force data
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
    logger.info('Plotting force data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    dt_simu = plot_data['dt_simu']
    dt_ctrl = plot_data['dt_ctrl']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot - dt_simu, N_simu)
    t_span_ctrl = np.linspace(0, T_tot - dt_ctrl, N_ctrl)
    t_span_plan = np.linspace(0, T_tot - dt_plan, N_plan)
    fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
    # Plot endeff
    xyz = ['x', 'y', 'z']
    for i in range(3):

        if(PLOT_PREDICTIONS):
            f_ee_pred_i = plot_data['f_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions
                points_f = np.array([tspan_x_pred, f_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
                lc_f.set_array(tspan_x_pred)
                # Customize
                lc_f.set_linestyle('-')
                lc_f.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_f)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h-1), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, f_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
       
        # EE linear force
        ax[i,0].plot(t_span_plan, plot_data['f_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,0].plot(t_span_ctrl, plot_data['f_ee_des_CTRL'][:,i], 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax[i,0].plot(t_span_simu, plot_data['f_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['f_ee_mea'][:,i], 'r-', label='Measured', linewidth=2, alpha=0.6)
        # ax[i,0].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,i], 'r-', label='measured', linewidth=2)
        # Plot reference
        if('force' in plot_data['WHICH_COSTS']):
            ax[i,0].plot(t_span_plan, plot_data['f_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)

        # EE angular force 
        ax[i,1].plot(t_span_plan, plot_data['f_ee_des_PLAN'][:,3+i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,1].plot(t_span_ctrl, plot_data['f_ee_des_CTRL'][:,i], 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax[i,1].plot(t_span_simu, plot_data['f_ee_des_SIMU'][:,3+i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['f_ee_mea'][:,3+i], 'r-', label='Measured', linewidth=2, alpha=0.6)
        # ax[i,1].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,3+i]-[plot_data['f_ee_ref'][3+i]]*(N_simu+1), 'r-', label='Measured', linewidth=2)
        # Plot reference
        if('force' in plot_data['WHICH_COSTS']):
            ax[i,1].plot(t_span_plan, plot_data['f_ee_ref'][:,3+i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$\\tau^{EE}_%s$  (Nm)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,1].grid(True)
    
    # Align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Set ylim if any
    TOL = 1e-3
    if(AUTOSCALE):
        ax_ylim = 1.1*max( np.nanmax(np.abs(plot_data['f_ee_mea'])), TOL )
        ax_ylim = 1.1*max( np.nanmax(np.abs(plot_data['f_ee_mea'])), TOL )
        for i in range(3):
            ax[i,0].set_ylim(-ax_ylim, ax_ylim) 
            # ax[i,0].set_ylim(-30, 10) 
            ax[i,1].set_ylim(-ax_ylim, ax_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector forces (LOCAL)', size=18)
    # Save figs
    if(SAVE):
        figs = {'f': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig, ax

# Plot Ricatti SVD
def plot_mpc_ricatti_svd(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    logger.info('Plotting Ricatti singular values...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_K, ax_K = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):
        # Ricatti gains singular values
        ax_K[i].plot(t_span_plan_u, plot_data['K_svd'][:, 0, i], 'b-', label='Singular Values of Ricatti gain K')
        ax_K[i].set_ylabel('$\sigma_{}$'.format(i), fontsize=12)
        ax_K[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_K[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_K[i].grid(True)
        # Set xlabel on bottom plot
        if(i == nq-1):
            ax_K[i].set_xlabel('t (s)', fontsize=16)
    # y axis labels
    # fig_K.text(0.04, 0.5, 'Singular values', va='center', rotation='vertical', fontsize=16)
    # Titles
    fig_K.suptitle('Singular Values of Ricatti feedback gains K', size=16)
    # Save figs
    if(SAVE):
        figs = {'K_svd': fig_K}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_K

# Plot Ricatti Diagonal
def plot_mpc_ricatti_diag(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    fig_K, ax_K = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
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
        if(i == nq-1):
            ax_K[i,0].set_xlabel('t (s)', fontsize=16)
            ax_K[i,1].set_xlabel('t (s)', fontsize=16)
    # y axis labels
    fig_K.text(0.05, 0.5, '$K_p$', va='center', rotation='vertical', fontsize=16)
    fig_K.text(0.48, 0.5, '$K_v', va='center', rotation='vertical', fontsize=16)
    fig_K.subplots_adjust(wspace=0.27)  
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
def plot_mpc_Vxx_eig(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    fig_V, ax_V = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
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
        # Set xlabel on bottom plot
        if(i == nq-1):
            ax_V[i,0].set_xlabel('t (s)', fontsize=16)
            ax_V[i,1].set_xlabel('t (s)', fontsize=16)
    # # y axis labels
    # fig_V.text(0.05, 0.5, 'Eigenvalue', va='center', rotation='vertical', fontsize=16)
    # fig_V.text(0.49, 0.5, 'Eigenvalue', va='center', rotation='vertical', fontsize=16)
    # fig_V.subplots_adjust(wspace=0.27)  
    # Titles
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
def plot_mpc_Vxx_diag(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    fig_V, ax_V = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
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
        # Set xlabel on bottom plot
        if(i == nq-1):
            ax_V[i,0].set_xlabel('t (s)', fontsize=16)
            ax_V[i,1].set_xlabel('t (s)', fontsize=16)
    # y axis labels
    # fig_V.text(0.05, 0.5, 'Diagonal Vxx', va='center', rotation='vertical', fontsize=16)
    # fig_V.text(0.49, 0.5, 'Diagonal Vxx', va='center', rotation='vertical', fontsize=16)
    # fig_V.subplots_adjust(wspace=0.27)  
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
def plot_mpc_Quu_eig(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    logger.info('Plotting Quu eigenvalues...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_Q, ax_Q = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
    # For each state
    for i in range(nq):
        # Quu eigenvals
        ax_Q[i].plot(t_span_plan_u, plot_data['Quu_eig'][:, 0, i], 'b-', label='Quu eigenvalue')
        ax_Q[i].set_ylabel('$\lambda_{}$'.format(i), fontsize=12)
        ax_Q[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_Q[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_Q[i].grid(True)
        # Set xlabel on bottom plot
        if(i == nq-1):
            ax_Q[i].set_xlabel('t (s)', fontsize=16)
    # Titles
    fig_Q.suptitle('Eigenvalues of Hamiltonian Hessian Quu', size=16)
    # Save figs
    if(SAVE):
        figs = {'Q_eig': fig_Q}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_Q

# Plot Quu diag
def plot_mpc_Quu_diag(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    logger.info('Plotting Quu diagonal...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_Q, ax_Q = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
    # For each state
    for i in range(nq):
        # Quu diag
        ax_Q[i].plot(t_span_plan_u, plot_data['Quu_diag'][:, 0, i], 'b-', label='Quu diagonal')
        ax_Q[i].set_ylabel('$Quu_{}$'.format(i), fontsize=12)
        ax_Q[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_Q[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_Q[i].grid(True)
        # Set xlabel on bottom plot
        if(i == nq-1):
            ax_Q[i].set_xlabel('t (s)', fontsize=16)
    # Titles
    fig_Q.suptitle('Diagonal of Hamiltonian Hessian Quu', size=16)
    # Save figs
    if(SAVE):
        figs = {'Q_diag': fig_Q}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_Q

# Plot Solver regs
def plot_mpc_solver(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    logger.info('Plotting solver data...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_S, ax_S = plt.subplots(2, 1, figsize=(19.2,10.8), sharex='col') 
    # Xreg
    ax_S[0].plot(t_span_plan_u, plot_data['xreg'], 'b-', label='xreg')
    ax_S[0].set(xlabel='t (s)', ylabel='$xreg$')
    ax_S[0].grid(True)
    # Ureg
    ax_S[1].plot(t_span_plan_u, plot_data['ureg'], 'r-', label='ureg')
    ax_S[1].set(xlabel='t (s)', ylabel='$ureg$')
    ax_S[1].grid(True)

    # Titles
    fig_S.suptitle('FDDP solver regularization on x (Vxx diag) and u (Quu diag)', size=16)
    # Save figs
    if(SAVE):
        figs = {'S': fig_S}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_S

# Plot rank of Jacobian
def plot_mpc_jacobian(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    logger.info('Plotting solver data...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_J, ax_J = plt.subplots(1, 1, figsize=(19.2,10.8), sharex='col') 
    # Rank of Jacobian
    ax_J.plot(t_span_plan_u, plot_data['J_rank'], 'b-', label='rank')
    ax_J.set(xlabel='t (s)', ylabel='rank')
    ax_J.grid(True)

    # Titles
    fig_J.suptitle('Rank of Jacobian J(q)', size=16)
    # Save figs
    if(SAVE):
        figs = {'J': fig_J}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_J
