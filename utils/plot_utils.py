from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import pin_utils

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    if('p' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        figs['p'], axes['p'] = plot_mpc_endeff_LPF(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
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
        ax[i,0].plot(t_span_plan, plot_data['q_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.5)
        # ax[i,0].plot(t_span_ctrl, plot_data['q_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Predicted (CTRL)', alpha=0.3)
        ax[i,0].plot(t_span_simu, plot_data['q_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,0].plot(t_span_simu, plot_data['q_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        ax[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        
        # Joint velocity 
        ax[i,1].plot(t_span_plan, plot_data['v_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN)', alpha=0.5)
        # ax[i,1].plot(t_span_ctrl, plot_data['v_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL)', alpha=0.3)
        ax[i,1].plot(t_span_simu, plot_data['v_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,1].plot(t_span_simu, plot_data['v_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        ax[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)

        # Joint torques
        ax[i,2].plot(t_span_plan, plot_data['tau_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.5)
        # ax[i,2].plot(t_span_ctrl, plot_data['tau_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Predicted (CTRL)', alpha=0.3)
        ax[i,2].plot(t_span_simu, plot_data['tau_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax[i,2].plot(t_span_simu, plot_data['tau_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,2].plot(t_span_simu, plot_data['tau_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        ax[i,2].plot(t_span_simu, plot_data['grav'][:,i], color='k', marker=None, linestyle='-.', label='Reg (grav)', alpha=0.6)
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
        ax[i].plot(t_span_plan, plot_data['w_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Prediction (PLAN)', alpha=0.6)
        # ax[i].plot(t_span_ctrl, plot_data['w_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Prediction (CTRL)', alpha=0.6)
        ax[i].plot(t_span_simu, plot_data['w_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Prediction (SIMU)', alpha=0.6)
        ax[i].plot(t_span_simu, plot_data['grav'][:-1,i], color='k', marker=None, linestyle='-.', label='Reg (grav)', alpha=0.6)
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

# Plot end-eff data
def plot_mpc_endeff_LPF(plot_data, PLOT_PREDICTIONS=False, 
                                   pred_plot_sampling=100, 
                                   SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                   SHOW=True,
                                   AUTOSCALE=False):
    '''
    Plot endeff data
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
    return plot_mpc_endeff(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
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





### Plot from DDP solver (LPF)

def plot_ddp_results_LPF(DDP_DATA, which_plots='all', labels=None, markers=None, colors=None, sampling_plot=1, SHOW=False):
    '''
    Plot ddp results from 1 or several DDP solvers
        X, U, EE trajs
        INPUT 
        DDP_DATA    : DDP data or list of ddp data (cf. data_utils.extract_ddp_data())
    '''
    logger.info("Plotting DDP solver data (LPF)...")
    if(type(DDP_DATA) != list):
        DDP_DATA = [DDP_DATA]
    if(labels==None):
        labels=[None for k in range(len(DDP_DATA))]
    if(markers==None):
        markers=[None for k in range(len(DDP_DATA))]
    if(colors==None):
        colors=[None for k in range(len(DDP_DATA))]
    for k,data in enumerate(DDP_DATA):
        # If last plot, make legend
        make_legend = False
        if(k+sampling_plot > len(DDP_DATA)-1):
            make_legend=True
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('y' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_x, ax_x = plot_ddp_state_LPF(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('w' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('us' in data.keys()):
                    fig_u, ax_u = plot_ddp_control_LPF(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('p' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_p, ax_p = plot_ddp_endeff_LPF(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('fs' in data.keys()):
                    fig_f, ax_f = plot_ddp_force_LPF(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
        else:
            if(k%sampling_plot==0):
                if('y' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_state_LPF(data, fig=fig_x, ax=ax_x, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('w' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('us' in data.keys()):
                        plot_ddp_control_LPF(data, fig=fig_u, ax=ax_u, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('p' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_endeff_LPF(data, fig=fig_p, ax=ax_p, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('fs' in data.keys()):
                        plot_ddp_force_LPF(data, fig=fig_f, ax=ax_f, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
    if(SHOW):
      plt.show()
    
    
    # Record and return if user needs to overlay stuff
    fig = {}
    ax = {}
    if('y' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('xs' in data.keys()):
            fig['y'] = fig_x
            ax['y'] = ax_x
    if('w' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('us' in data.keys()):
            fig['w'] = fig_u
            ax['w'] = ax_u
    if('p' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('xs' in data.keys()):
            fig['p'] = fig_p
            ax['p'] = ax_p
    if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('fs' in data.keys()):
            fig['f'] = fig_f
            ax['f'] = ax_f

    return fig, ax

def plot_ddp_state_LPF(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (state)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nq = ddp_data['nq'] 
    nv = ddp_data['nv'] 
    nu = ddp_data['nu'] 
    # Extract pos, vel trajs
    x = np.array(ddp_data['xs'])
    q = x[:,:nq]
    v = x[:,nq:nq+nv]
    tau = x[:,-nu:]
    # If tau reg cost, compute gravity torque
    if('ctrlReg' in ddp_data['active_costs']):
        ureg_ref  = np.array(ddp_data['ctrlReg_ref']) 
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ureg_grav = np.array(ddp_data['ctrlRegGrav_ref'])
    if('stateReg' in ddp_data['active_costs']):
        x_reg_ref = np.array(ddp_data['stateReg_ref'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 3, sharex='col') 
    if(label is None):
        label='State'
    for i in range(nq):
        # Positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
        if('stateReg' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reg_ref' in labels):
                handles.pop(labels.index('reg_ref'))
                ax[i,0].lines.pop(labels.index('reg_ref'))
                labels.remove('reg_ref')
            ax[i,0].plot(tspan, x_reg_ref[:,i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        # Velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)  
        if('stateReg' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reg_ref' in labels):
                handles.pop(labels.index('reg_ref'))
                ax[i,1].lines.pop(labels.index('reg_ref'))
                labels.remove('reg_ref')
            ax[i,1].plot(tspan, x_reg_ref[:,nq+i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)  
        # Torques
        ax[i,2].plot(tspan, tau[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
        # Plot control regularization reference 
        if('ctrlReg' in ddp_data['active_costs']):
            handles, labels = ax[i,2].get_legend_handles_labels()
            if('u_reg' in labels):
                handles.pop(labels.index('u_reg'))
                ax[i,2].lines.pop(labels.index('u_reg'))
                labels.remove('u_reg')
            ax[i,2].plot(tspan, ureg_ref[:,i], linestyle='-.', color='k', marker=None, label='u_reg', alpha=0.5)
        # Plot gravity compensation torque
        if('ctrlRegGrav' in ddp_data['active_costs']):
            handles, labels = ax[i,2].get_legend_handles_labels()
            if('grav(q)' in labels):
                handles.pop(labels.index('u_grav(q)'))
                ax[i,2].lines.pop(labels.index('u_grav(q)'))
                labels.remove('u_grav(q)')
            ax[i,2].plot(tspan, ureg_grav[:,i], linestyle='-.', color='m', marker=None, label='u_grav(q)', alpha=0.5)
        ax[i,2].set_ylabel('$\\tau_{}$'.format(i), fontsize=16)
        ax[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,2].grid()
    # Common x-labels
    ax[-1,0].set_xlabel('Time (s)', fontsize=16)
    ax[-1,1].set_xlabel('Time (s)', fontsize=16)
    ax[-1,2].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])
    fig.align_ylabels(ax[:, 2])
    # Legend
    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('State trajectories : joint positions and velocities', size=18)
    plt.subplots_adjust(wspace=0.3)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control_LPF(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (control)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nu = ddp_data['nu'] 
    nq = ddp_data['nq'] 
    # Extract pos, vel trajs
    w = np.array(ddp_data['us'])
    x = np.array(ddp_data['xs'])
    q = x[:,:nq]
    # If tau reg cost, compute gravity torque
    w_reg_ref = np.zeros((N,nu))
    for i in range(N):
        w_reg_ref[i,:] = pin_utils.get_u_grav(q[i,:], ddp_data['pin_model'])
    # Plots
    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col') 
    if(label is None):
        label='Control'    
    for i in range(nu):
        # Positions
        ax[i].plot(tspan, w[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
        # If tau reg cost, plot gravity torque
        handles, labels = ax[i].get_legend_handles_labels()
        if('reg_ref' in labels):
            handles.pop(labels.index('reg_ref'))
            ax[i].lines.pop(labels.index('reg_ref'))
            labels.remove('reg_ref')
        ax[i].plot(tspan, w_reg_ref[:,i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        ax[i].set_ylabel('$w_%s$'%i, fontsize=16)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:])
    # Legend
    if(MAKE_LEGEND):
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Control trajectories: unfiltered joint torques', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff_LPF(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (endeff)
    '''
    return plot_ddp_endeff(ddp_data, fig=fig, ax=ax, label=label, marker=marker, color=color, alpha=alpha, MAKE_LEGEND=MAKE_LEGEND, SHOW=SHOW)

def plot_ddp_force_LPF(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (force)
    '''
    return plot_ddp_force(ddp_data, fig=fig, ax=ax, label=label, marker=marker, color=color, alpha=alpha, MAKE_LEGEND=MAKE_LEGEND, SHOW=SHOW)






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

    if('p' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['p'] = plot_mpc_endeff(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
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
        ax_x[i,0].plot(t_span_plan, plot_data['q_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.5)
        # ax[i,0].plot(t_span_ctrl, plot_data['q_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Predicted (CTRL)', alpha=0.3)
        ax_x[i,0].plot(t_span_simu, plot_data['q_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax_x[i,0].plot(t_span_simu, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,0].plot(t_span_simu, plot_data['q_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        ax_x[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax_x[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,0].grid(True)

        # Joint velocity 
        ax_x[i,1].plot(t_span_plan, plot_data['v_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN)', alpha=0.5)
        # ax[i,1].plot(t_span_ctrl, plot_data['v_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL)', alpha=0.3)
        ax_x[i,1].plot(t_span_simu, plot_data['v_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU)', alpha=0.5)
        ax_x[i,1].plot(t_span_simu, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,1].plot(t_span_simu, plot_data['v_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
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
        ax_u[i].plot(t_span_plan, plot_data['u_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Prediction (PLAN)', alpha=0.6)
        # ax[i].plot(t_span_ctrl, plot_data['w_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Prediction (CTRL)', alpha=0.6)
        ax_u[i].plot(t_span_simu, plot_data['u_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Prediction (SIMU)', alpha=0.6)
        ax_u[i].plot(t_span_simu, plot_data['grav'][:-1,i], color='k', marker=None, linestyle='-.', label='Reg', alpha=0.6)
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
def plot_mpc_endeff(plot_data, PLOT_PREDICTIONS=False, 
                               pred_plot_sampling=100, 
                               SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                               SHOW=True,
                               AUTOSCALE=False):
    '''
    Plot endeff data
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
    logger.info('Plotting end-eff data...')
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
            p_ee_pred_i = plot_data['p_ee_pred'][:, :, i]
            v_ee_pred_i = plot_data['v_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, p_ee_pred_i[j,:]-plot_data['p_ee_ref'][i]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_ee_pred_i[j,:]-plot_data['v_ee_ref'][i]]).transpose().reshape(-1,1,2)
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
                ax[i,0].scatter(tspan_x_pred, p_ee_pred_i[j,:]-plot_data['p_ee_ref'][i], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                ax[i,1].scatter(tspan_x_pred, v_ee_pred_i[j,:]-plot_data['v_ee_ref'][i], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
       
        # EE position
        ax[i,0].plot(t_span_plan, plot_data['p_ee_des_PLAN'][:,i]-plot_data['p_ee_ref'][i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.5)
        # ax[i,0].plot(t_span_ctrl, plot_data['p_ee_des_CTRL'][:,i]-plot_data['p_ee_ref'][i], 'g-', label='Predicted (CTRL)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['p_ee_des_SIMU'][:,i]-plot_data['p_ee_ref'][i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['p_ee_mea'][:,i]-plot_data['p_ee_ref'][i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,0].plot(t_span_simu, plot_data['p_ee_mea_no_noise'][:,i]-[plot_data['p_ee_ref'][i]]*(N_simu+1), 'r-', label='measured', linewidth=2)
        ax[i,0].plot(t_span_plan, [0.]*(N_plan+1), 'k-.', linewidth=2., label='err=0', alpha=0.5)
        ax[i,0].set_ylabel('$\\Delta P^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)
        # EE velocity
        ax[i,1].plot(t_span_plan, plot_data['v_ee_des_PLAN'][:,i]-plot_data['v_ee_ref'][i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.5)
        # ax[i,1].plot(t_span_ctrl, plot_data['v_ee_des_CTRL'][:,i]-plot_data['v_ee_ref'][i], 'g-', label='Predicted (CTRL)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['v_ee_des_SIMU'][:,i]-plot_data['v_ee_ref'][i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['v_ee_mea'][:,i]-plot_data['v_ee_ref'][i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,1].plot(t_span_simu, plot_data['v_ee_mea_no_noise'][:,i]-[plot_data['v_ee_ref'][i]]*(N_simu+1), 'r-', label='Measured', linewidth=2)
        ax[i,1].plot(t_span_plan, [0.]*(N_plan+1), 'k-.', linewidth=2., label='err=0', alpha=0.5)
        ax[i,1].set_ylabel('$\\Delta V^{EE}_%s$  (m)'%xyz[i], fontsize=16)
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
        ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['p_ee_mea']-plot_data['p_ee_ref'])), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['v_ee_mea']-plot_data['v_ee_ref'])), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector trajectories errors', size=18)
    # Save figs
    if(SAVE):
        figs = {'p': fig}
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
                points_f = np.array([tspan_x_pred, f_ee_pred_i[j,:]-plot_data['f_ee_ref'][i]]).transpose().reshape(-1,1,2)
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
                ax[i,0].scatter(tspan_x_pred, f_ee_pred_i[j,:]-plot_data['f_ee_ref'][i], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
       
        # EE position
        ax[i,0].plot(t_span_plan, plot_data['f_ee_des_PLAN'][:,i]-plot_data['f_ee_ref'][i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.5)
        # ax[i,0].plot(t_span_ctrl, plot_data['p_ee_des_CTRL'][:,i]-plot_data['p_ee_ref'][i], 'g-', label='Predicted (CTRL)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['f_ee_des_SIMU'][:,i]-plot_data['f_ee_ref'][i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['f_ee_mea'][:,i]-plot_data['f_ee_ref'][i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        # ax[i,0].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,i]-[plot_data['p_ee_ref'][i]]*(N_simu+1), 'r-', label='measured', linewidth=2)
        ax[i,0].plot(t_span_plan, [0.]*(N_plan), 'k-.', linewidth=2., label='err=0', alpha=0.5)
        ax[i,0].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)
        # EE velocity
        ax[i,1].plot(t_span_plan, plot_data['f_ee_des_PLAN'][:,3+i]-plot_data['f_ee_ref'][3+i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.5)
        # ax[i,1].plot(t_span_ctrl, plot_data['v_ee_des_CTRL'][:,i]-plot_data['v_ee_ref'][i], 'g-', label='Predicted (CTRL)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['f_ee_des_SIMU'][:,3+i]-plot_data['f_ee_ref'][3+i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['f_ee_mea'][:,3+i]-plot_data['f_ee_ref'][3+i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        # ax[i,1].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,3+i]-[plot_data['f_ee_ref'][3+i]]*(N_simu+1), 'r-', label='Measured', linewidth=2)
        ax[i,1].plot(t_span_plan, [0.]*(N_plan), 'k-.', linewidth=2., label='err=0', alpha=0.5)
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
        ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['f_ee_mea']-plot_data['f_ee_ref'])), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['f_ee_mea']-plot_data['f_ee_ref'])), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector forces errors', size=18)
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




### Plot from DDP solver 
def plot_ddp_results(DDP_DATA, which_plots='all', labels=None, markers=None, colors=None, sampling_plot=1, SHOW=False):
    '''
    Plot ddp results from 1 or several DDP solvers
        X, U, EE trajs
        INPUT 
        DDP_DATA         : DDP data or list of ddp data (cf. data_utils.extract_ddp_data())
    '''
    logger.info("Plotting DDP solver data...")
    if(type(DDP_DATA) != list):
        DDP_DATA = [DDP_DATA]
    if(labels==None):
        labels=[None for k in range(len(DDP_DATA))]
    if(markers==None):
        markers=[None for k in range(len(DDP_DATA))]
    if(colors==None):
        colors=[None for k in range(len(DDP_DATA))]
    for k,data in enumerate(DDP_DATA):
        # If last plot, make legend
        make_legend = False
        if(k+sampling_plot > len(DDP_DATA)-1):
            make_legend=True
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_x, ax_x = plot_ddp_state(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('us' in data.keys()):
                    fig_u, ax_u = plot_ddp_control(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('p' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_p, ax_p = plot_ddp_endeff(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('fs' in data.keys()):
                    fig_f, ax_f = plot_ddp_force(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
        else:
            if(k%sampling_plot==0):
                if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_state(data, fig=fig_x, ax=ax_x, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('us' in data.keys()):
                        plot_ddp_control(data, fig=fig_u, ax=ax_u, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('p' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_endeff(data, fig=fig_p, ax=ax_p, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('fs' in data.keys()):
                        plot_ddp_force(data, fig=fig_f, ax=ax_f, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
    if(SHOW):
      plt.show()
    
    # Record and return if user needs to overlay stuff
    fig = {}
    ax = {}
    if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('xs' in data.keys()):
            fig['x'] = fig_x
            ax['x'] = ax_x
    if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('us' in data.keys()):
            fig['u'] = fig_u
            ax['u'] = ax_u
    if('p' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('xs' in data.keys()):
            fig['p'] = fig_p
            ax['p'] = ax_p
    if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('fs' in data.keys()):
            fig['f'] = fig_f
            ax['f'] = ax_f

    return fig, ax

def plot_ddp_state(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (state)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nq = ddp_data['nq'] 
    nv = ddp_data['nv'] 
    # Extract trajectories
    x = np.array(ddp_data['xs'])
    q = x[:,:nq]
    v = x[:,nv:]
    # If state reg cost, 
    if('stateReg' in ddp_data['active_costs']):
        x_reg_ref = np.array(ddp_data['stateReg_ref'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 2, sharex='col') 
    if(label is None):
        label='State'
    for i in range(nq):
        # Plot positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot joint position regularization reference
        if('stateReg' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reg_ref' in labels):
                handles.pop(labels.index('reg_ref'))
                ax[i,0].lines.pop(labels.index('reg_ref'))
                labels.remove('reg_ref')
            ax[i,0].plot(tspan, x_reg_ref[:,i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)  

        # Plot joint velocity regularization reference
        if('stateReg' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reg_ref' in labels):
                handles.pop(labels.index('reg_ref'))
                ax[i,1].lines.pop(labels.index('reg_ref'))
                labels.remove('reg_ref')
            ax[i,1].plot(tspan, x_reg_ref[:,nq+i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        
        # Labels, tick labels and grid
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)  

    # Common x-labels + align
    ax[-1,0].set_xlabel('Time (s)', fontsize=16)
    ax[-1,1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])

    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('State trajectories', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (control)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nu = ddp_data['nu'] 
    # Extract trajectory
    u = np.array(ddp_data['us'])
    if('ctrlReg' in ddp_data['active_costs']):
        ureg_ref  = np.array(ddp_data['ctrlReg_ref']) 
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ureg_grav = np.array(ddp_data['ctrlRegGrav_ref'])

    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col') 
    if(label is None):
        label='Control'    

    for i in range(nu):
        # Plot optimal control trajectory
        ax[i].plot(tspan, u[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot control regularization reference 
        if('ctrlReg' in ddp_data['active_costs']):
            handles, labels = ax[i].get_legend_handles_labels()
            if('u_reg' in labels):
                handles.pop(labels.index('u_reg'))
                ax[i].lines.pop(labels.index('u_reg'))
                labels.remove('u_reg')
            ax[i].plot(tspan, ureg_ref[:,i], linestyle='-.', color='k', marker=None, label='u_reg', alpha=0.5)

        # Plot gravity compensation torque
        if('ctrlRegGrav' in ddp_data['active_costs']):
            handles, labels = ax[i].get_legend_handles_labels()
            if('grav(q)' in labels):
                handles.pop(labels.index('u_grav(q)'))
                ax[i].lines.pop(labels.index('u_grav(q)'))
                labels.remove('u_grav(q)')
            ax[i].plot(tspan, ureg_grav[:,i], linestyle='-.', color='m', marker=None, label='u_grav(q)', alpha=0.5)
        
        # Labels, tick labels, grid
        ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i].grid(True)

    # Set x label + align
    ax[-1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:])
    # Legend
    if(MAKE_LEGEND):
        handles, labels = ax[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Control trajectories', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                    MAKE_LEGEND=False, SHOW=True, AUTOSCALE=True):
    '''
    Plot ddp results (endeff)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nq = ddp_data['nq']
    nv = ddp_data['nv'] 
    # Extract EE traj
    x = np.array(ddp_data['xs'])
    q = x[:,:nq]
    v = x[:,nq:nq+nv]
    p_ee = pin_utils.get_p_(q, ddp_data['pin_model'], ddp_data['frame_id'])
    v_ee = pin_utils.get_v_(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    if('translation' in ddp_data['active_costs']):
        p_ee_ref = np.array(ddp_data['translation_ref'])
    else:
        p_ee_ref = np.array([p_ee[0,:] for i in range(N+1)])
    if('velocity' in ddp_data['active_costs']):
        v_ee_ref = np.array(ddp_data['velocity_ref'])
    else:
        v_ee_ref = np.array([v_ee[0,:] for i in range(N+1)])
    if('contact_translation' in ddp_data):
        p_ee_contact = np.array(ddp_data['contact_translation'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col')
    if(label is None):
        label='OCP solution'
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE position in WORLD frame
        ax[i,0].plot(tspan, p_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot EE target frame translation in WORLD frame
        if('translation' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,0].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,0].plot(tspan, p_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        
        # Plot CONTACT reference frame translation in WORLD frame
        if('contact_translation' in ddp_data):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('contact' in labels):
                handles.pop(labels.index('contact'))
                ax[i,0].lines.pop(labels.index('contact'))
                labels.remove('contact')
            ax[i,0].plot(tspan, p_ee_contact[:,i], linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)

        # Labels, tick labels, grid
        ax[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot EE 'linear) velocities in WORLD frame
        ax[i,1].plot(tspan, v_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot EE target frame (linear) velocity in WORLD frame
        if('velocity' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,1].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,1].plot(tspan, v_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        
        # Labels, tick labels, grid
        ax[i,1].set_ylabel('$V^{EE}_%s$ (m/s)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)
    
    #x-label + align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)

    # Set ylim if any
    if(AUTOSCALE):
        TOL = 0.1
        ax_p_ylim = 1.1*max(np.max(np.abs(p_ee)), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(v_ee)), TOL)
        for i in range(3):
            ax[i,0].set_ylim(p_ee_ref[0,i]-ax_p_ylim, p_ee_ref[0,i]+ax_p_ylim) 
            ax[i,1].set_ylim(v_ee_ref[0,i]-ax_v_ylim, v_ee_ref[0,i]+ax_v_ylim)

    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector trajectories: position and velocity', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_force(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                MAKE_LEGEND=False, SHOW=True, AUTOSCALE=True):
    '''
    Plot ddp results (force)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    # Extract EE traj
    f = np.array(ddp_data['fs'])
    f_ee_lin = f[:,:3]
    f_ee_ang = f[:,3:]
    # Get desired contact wrench (linear, angular)
    if('force' in ddp_data['active_costs']):
        f_ee_ref = np.array(ddp_data['force_ref'])
    else:
        f_ee_ref = np.zeros((N,6))
    f_ee_lin_ref = f_ee_ref[:,:3]
    f_ee_ang_ref = f_ee_ref[:,3:]
    # Plots
    tspan = np.linspace(0, N*dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col')
    if(label is None):
        label='End-effector force'
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot contact linear wrench (force) in LOCAL frame
        ax[i,0].plot(tspan, f_ee_lin[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot desired contact linear wrench (force) in LOCAL frame 
        if('force' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,0].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,0].plot(tspan, f_ee_lin_ref[:,i], linestyle='-.', color='k', marker=None, label='reference', alpha=0.5)
        
        # Labels, tick labels+ grid
        ax[i,0].set_ylabel('$\\lambda^{lin}_%s$ (N)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot contact angular wrench (torque) in LOCAL frame 
        ax[i,1].plot(tspan, f_ee_ang[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot desired contact anguler wrench (torque) in LOCAL frame
        if('force' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,1].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,1].plot(tspan, f_ee_ang_ref[:,i], linestyle='-.', color='k', marker=None, label='reference', alpha=0.5)

        # Labels, tick labels+ grid
        ax[i,1].set_ylabel('$\\lambda^{ang}_%s$ (Nm)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)
    
    # x-label + align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)

    # Set ylim if any
    if(AUTOSCALE):
        TOL = 1e-1
        ax_lin_ylim = 1.1*max(np.max(np.abs(f_ee_lin)), TOL)
        ax_ang_ylim = 1.1*max(np.max(np.abs(f_ee_ang)), TOL)
        for i in range(3):
            ax[i,0].set_ylim(f_ee_lin_ref[0,i]-ax_lin_ylim, f_ee_lin_ref[0,i]+ax_lin_ylim) 
            ax[i,1].set_ylim(f_ee_ang_ref[0,i]-ax_ang_ylim, f_ee_ang_ref[0,i]+ax_ang_ylim)

    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector forces: linear and angular', size=18)
    if(SHOW):
        plt.show()
    return fig, ax





# Animate and plot point mass from X,U trajs 
def animatePointMass(xs, sleep=1):
    '''
    Animate the point mass system with state trajectory xs
    '''
    # Check which model is used
    logger.info(len(xs[0]))
    if(len(xs[0])>2):
        with_contact = True
    else:
        with_contact = False

    logger.info("processing the animation ... ")
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
    logger.info("... processing done")
    plt.grid(True)
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
    # Is there a 'reference' to plot as well?
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
    ax[0].grid(True)
    # Plot velocity
    ax[1].plot(tspan, x2, 'b-', linewidth=3, label='v')
    if(with_ref):
        ax[1].plot(tspan, [ref[1]]*(T+1), 'k-.', linewidth=2, label='ref')
    ax[1].set_title('Velocity v', size=16)
    ax[1].set(xlabel='time (s)', ylabel='v (m/s)')
    ax[1].grid(True)
    # Plot force if necessary 
    if(with_contact):
        # Contact
        ax[2].plot(tspan, x3, 'b-', linewidth=3, label='lambda')
        if(with_ref):
            ax[2].plot(tspan, [ref[2]]*(T+1), 'k-.', linewidth=2, label='ref')
        ax[2].set_title('Contact force lambda', size=16)
        ax[2].set(xlabel='time (s)', ylabel='lmb (N)')
        ax[2].grid(True)
    # Plot control 
    ax[-1].plot(tspan[:T], u, 'k-', linewidth=3, label='u')
    ax[-1].set_title('Input force u', size=16)
    ax[-1].set(xlabel='time (s)', ylabel='u (N)')
    ax[-1].grid(True)
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
    ax[0].grid(True)
    # Plot velocities
    ax[1].plot(tspan[:T], Y_mea[:,1], 'b-', linewidth=2, alpha=.5, label='Measured')
    ax[1].plot(tspan, X_hat[:,1], 'r-', linewidth=3, alpha=.8, label='Filtered')
    ax[1].plot(tspan, X_real[:,1], 'k-.', linewidth=2, label='Ground truth')
    ax[1].set_title('Velocities p', size=16)
    ax[1].set(xlabel='time (s)', ylabel='v (m/s)')
    ax[1].grid(True)
    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Kalman-filtered point mass trajectory', size=16)
    plt.show()

