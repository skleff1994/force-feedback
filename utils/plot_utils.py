from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import pin_utils

### Plot from MPC simulation (LPF)
# Plot state data
def plot_mpc_state_lpf(plot_data, PLOT_PREDICTIONS=False, 
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
    print('Plotting state data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
    t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
    fig_x, ax_x = plt.subplots(nq, 3, figsize=(19.2,10.8), sharex='col') 
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
                ax_x[i,0].add_collection(lc_q)
                ax_x[i,1].add_collection(lc_v)
                ax_x[i,2].add_collection(lc_tau)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
                ax_x[i,2].scatter(tspan_x_pred, tau_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 

        # Joint position
        ax_x[i,0].plot(t_span_plan_x, plot_data['q_des'][:,i], 'b-', label='Desired')
        ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)', linewidth=2)
        ax_x[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax_x[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,0].grid(True)
        
        # Joint velocity 
        ax_x[i,1].plot(t_span_plan_x, plot_data['v_des'][:,i], 'b-', label='Desired')
        ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)')
        ax_x[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
        ax_x[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,1].grid(True)

        # Joint torques
        ax_x[i,2].plot(t_span_plan_x, plot_data['tau_des'][:,i], 'b-', label='Desired')
        ax_x[i,2].plot(t_span_simu_x[:-1], plot_data['tau_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,2].plot(t_span_simu_x, plot_data['tau_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)')
        ax_x[i,2].set_ylabel('$\\tau{}$'.format(i), fontsize=12)
        ax_x[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,2].grid(True)

        # Add xlabel on bottom plot of each column
        if(i == nq-1):
            ax_x[i,0].set_xlabel('t(s)', fontsize=16)
            ax_x[i,1].set_xlabel('t(s)', fontsize=16)
            ax_x[i,2].set_xlabel('t(s)', fontsize=16)
        # Legend
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
    # y axis labels
    fig_x.text(0.06, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=12)
    fig_x.text(0.345, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=12)
    fig_x.text(0.625, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=12)
    fig_x.subplots_adjust(wspace=0.37)
    # Titles
    fig_x.suptitle('State = joint position ($q$), velocity ($v$), torque ($\\tau$)', size=16)
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
def plot_mpc_control_lpf(plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True,
                            AUTOSCALE=False):
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
    print('Plotting control data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    dt_simu = plot_data['dt_simu']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_u = np.linspace(0, T_tot-dt_simu, N_simu)
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
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
                ax_u[i].add_collection(lc_u)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

        # Joint torques
        ax_u[i].plot(t_span_plan_u, plot_data['tau_des'][:-1,i], 'b-', label='Desired')
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
    fig_u.suptitle('Control = unfiltered joint torques', size=16)
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

# Plot data
def plot_mpc_results_lpf(plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
                                              pred_plot_sampling=100, 
                                              SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                              SHOW=True,
                                              AUTOSCALE=False):
    '''
    Plot sim data (MPC simulation using LPF, i.e. state x = (q,v,tau))
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

    if('x' in which_plots or which_plots is None or which_plots =='all'):
        plots['x'] = plot_mpc_state_lpf(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                           pred_plot_sampling=pred_plot_sampling, 
                                           SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                           SHOW=False)
    
    if('u' in which_plots or which_plots is None or which_plots =='all'):
        plots['u'] = plot_mpc_control_lpf(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                             pred_plot_sampling=pred_plot_sampling, 
                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False)
    
    if('a' in which_plots or which_plots is None or which_plots =='all'):
        plots['a'] = plot_mpc_acc_err(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=SHOW)

    if('p' in which_plots or which_plots is None or which_plots =='all'):
        plots['p'] = plot_mpc_endeff(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)

    if('K' in which_plots or which_plots is None or which_plots =='all'):
        if('K_diag' in plot_data.keys()):
            plots['K_diag'] = plot_mpc_ricatti_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)
        if('K_svd' in plot_data.keys()):
            plots['K_svd'] = plot_mpc_ricatti_svd(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('V' in which_plots or which_plots is None or which_plots =='all'):
        if('V_diag' in plot_data.keys()):
            plots['V_diag'] = plot_mpc_Vxx_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('V_eig' in plot_data.keys()):
            plots['V_eig'] = plot_mpc_Vxx_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)

    if('S' in which_plots or which_plots is None or which_plots =='all'):
        if('S' in plot_data.keys()):
            plots['S'] = plot_mpc_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('J' in which_plots or which_plots is None or which_plots =='all'):
        if('J' in plot_data.keys()):
            plots['J'] = plot_mpc_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('Q' in which_plots or which_plots is None or which_plots =='all'):
        if('Q_diag' in plot_data.keys()):
            plots['Q_diag'] = plot_mpc_Quu_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('Q_eig' in plot_data.keys()):
            plots['Q_eig'] = plot_mpc_Quu_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
    
    if(SHOW):
        plt.show() 
    plt.close('all')



### Plot from MPC simulation (regular i.e. 'impedance_mpc' repo)
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
    print('Plotting state data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']
    nx = plot_data['nx']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
    t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
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
        ax_x[i,0].plot(t_span_plan_x, plot_data['q_des'][:,i], 'b-', label='Desired')
        ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)', linewidth=2)
        ax_x[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax_x[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,0].grid(True)
        
        # Joint velocity 
        ax_x[i,1].plot(t_span_plan_x, plot_data['v_des'][:,i], 'b-', label='Desired')
        ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea_no_noise'][:,i], 'r-', label='Measured (NO noise)')
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
    fig_x.suptitle('State = joint positions, velocities', size=16)
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
    print('Plotting control data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    dt_simu = plot_data['dt_simu']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_u = np.linspace(0, T_tot-dt_simu, N_simu)
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
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
        ax_u[i].plot(t_span_plan_u, plot_data['u_des'][:,i], 'b-', label='Desired')
        ax_u[i].plot(t_span_simu_u, plot_data['u_mea'][:,i], 'r-', label='Measured') 
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
    fig_u.suptitle('Control = joint torques', size=16)
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
    print('Plotting end-eff data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    p_ref = plot_data['p_ref']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
    t_span_ctrl_x = np.linspace(0, T_tot, N_ctrl+1)
    t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
    fig_p, ax_p = plt.subplots(3,1, figsize=(19.2,10.8), sharex='col') 
    # Plot endeff
    # x
    ax_p[0].plot(t_span_plan_x, plot_data['p_des'][:,0]-p_ref[0], 'b-', label='p_des - p_ref', alpha=0.5)
    ax_p[0].plot(t_span_simu_x, plot_data['p_mea'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (WITH noise)', linewidth=1, alpha=0.3)
    ax_p[0].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,0]-[p_ref[0]]*(N_simu+1), 'r-', label='p_mea - p_ref (NO noise)', linewidth=2)
    ax_p[0].set_title('x-position-ERROR')
    ax_p[0].set_ylabel('x (m)', fontsize=16)
    ax_p[0].yaxis.set_major_locator(plt.MaxNLocator(2))
    ax_p[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
    ax_p[0].grid(True)
    # y
    ax_p[1].plot(t_span_plan_x, plot_data['p_des'][:,1]-p_ref[1], 'b-', label='py_des - py_ref', alpha=0.5)
    ax_p[1].plot(t_span_simu_x, plot_data['p_mea'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (WITH noise)', linewidth=1, alpha=0.3)
    ax_p[1].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,1]-[p_ref[1]]*(N_simu+1), 'r-', label='py_mea - py_ref (NO noise)', linewidth=2)
    ax_p[1].set_title('y-position-ERROR')
    ax_p[1].set_ylabel('y (m)', fontsize=16)
    ax_p[1].yaxis.set_major_locator(plt.MaxNLocator(2))
    ax_p[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
    ax_p[1].grid(True)
    # z
    ax_p[2].plot(t_span_plan_x, plot_data['p_des'][:,2]-p_ref[2], 'b-', label='pz_des - pz_ref', alpha=0.5)
    ax_p[2].plot(t_span_simu_x, plot_data['p_mea'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (WITH noise)', linewidth=1, alpha=0.3)
    ax_p[2].plot(t_span_simu_x, plot_data['p_mea_no_noise'][:,2]-[p_ref[2]]*(N_simu+1), 'r-', label='pz_mea - pz_ref (NO noise)', linewidth=2)
    ax_p[2].set_title('z-position-ERROR')
    ax_p[2].set_ylabel('z (m)', fontsize=16)
    ax_p[2].yaxis.set_major_locator(plt.MaxNLocator(2))
    ax_p[2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
    ax_p[2].set_xlabel('t (s)', fontsize=16)
    ax_p[2].grid(True)
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
    fig_p.suptitle('End-effector trajectories errors', size=16)

    # Save figs
    if(SAVE):
        figs = {'p': fig_p}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_p

# Plot acceleration error data
def plot_mpc_acc_err(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True):
    '''
    Plot acc err data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    print('Plotting acc error data...')
    T_tot = plot_data['T_tot']
    N_ctrl = plot_data['N_ctrl']
    dt_ctrl = plot_data['dt_ctrl']
    nq = plot_data['nq']
    # Create time spans for X and U + Create figs and subplots
    t_span_ctrl_u = np.linspace(0, T_tot-dt_ctrl, N_ctrl)
    fig_a, ax_a = plt.subplots(nq,2, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):
        # Joint velocity error (avg over 1 control cycle)
        ax_a[i,0].plot(t_span_ctrl_u, plot_data['a_err'][:,i], 'b-', label='Velocity error (average)')
        ax_a[i,0].set_ylabel('$verr_{}$'.format(i), fontsize=12)
        ax_a[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_a[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_a[i,0].grid(True)
        # Joint acceleration error (avg over 1 control cycle)
        ax_a[i,1].plot(t_span_ctrl_u, plot_data['a_err'][:,nq+i], 'b-', label='Acceleration error (average)')
        ax_a[i,1].set_ylabel('$aerr_{}$'.format(i), fontsize=12)
        ax_a[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_a[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_a[i,1].grid(True)
        # Set xlabel on bottom plot
        if(i == nq-1):
            ax_a[i,0].set_xlabel('t (s)', fontsize=16)
            ax_a[i,1].set_xlabel('t (s)', fontsize=16)
    # y axis labels
    fig_a.text(0.05, 0.5, 'Vel. error (rad/s)', va='center', rotation='vertical', fontsize=16)
    fig_a.text(0.49, 0.5, 'Acc. error (rad/s^2)', va='center', rotation='vertical', fontsize=16)
    fig_a.subplots_adjust(wspace=0.27)    
    # title
    fig_a.suptitle('Average tracking errors over control cycles (1ms)', size=16)
    # Save figs
    if(SAVE):
        figs = {'a': fig_a}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_a

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
    print('Plotting Ricatti singular values...')
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
    print('Plotting Ricatti diagonal...')
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
    print('Plotting Vxx eigenvalues...')
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
    print('Plotting Vxx diagonal...')
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
    print('Plotting Quu eigenvalues...')
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
    print('Plotting Quu diagonal...')
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
        ax_Q[i].set_ylabel('$Vxx_{}$'.format(i), fontsize=12)
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
    print('Plotting solver data...')
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
    print('Plotting solver data...')
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

    if('x' in which_plots or which_plots is None or which_plots =='all'):
        plots['x'] = plot_mpc_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                           pred_plot_sampling=pred_plot_sampling, 
                                           SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                           SHOW=False)
    
    if('u' in which_plots or which_plots is None or which_plots =='all'):
        plots['u'] = plot_mpc_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                             pred_plot_sampling=pred_plot_sampling, 
                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False)
    
    if('a' in which_plots or which_plots is None or which_plots =='all'):
        plots['a'] = plot_mpc_acc_err(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=SHOW)

    if('p' in which_plots or which_plots is None or which_plots =='all'):
        plots['p'] = plot_mpc_endeff(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)

    if('K' in which_plots or which_plots is None or which_plots =='all'):
        if('K_diag' in plot_data.keys()):
            plots['K_diag'] = plot_mpc_ricatti_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)
        if('K_svd' in plot_data.keys()):
            plots['K_svd'] = plot_mpc_ricatti_svd(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('V' in which_plots or which_plots is None or which_plots =='all'):
        if('V_diag' in plot_data.keys()):
            plots['V_diag'] = plot_mpc_Vxx_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('V_eig' in plot_data.keys()):
            plots['V_eig'] = plot_mpc_Vxx_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)

    if('S' in which_plots or which_plots is None or which_plots =='all'):
        if('S' in plot_data.keys()):
            plots['S'] = plot_mpc_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('J' in which_plots or which_plots is None or which_plots =='all'):
        if('J' in plot_data.keys()):
            plots['J'] = plot_mpc_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                SHOW=False)

    if('Q' in which_plots or which_plots is None or which_plots =='all'):
        if('Q_diag' in plot_data.keys()):
            plots['Q_diag'] = plot_mpc_Quu_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
        if('Q_eig' in plot_data.keys()):
            plots['Q_eig'] = plot_mpc_Quu_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
    
    if(SHOW):
        plt.show() 
    plt.close('all')



### Plot from DDP solver 
def plot_ddp_results(ddp, robot, id_endeff, which_plots='all'):
    '''
    Plot ddp results from 1 or several DDP solvers
    X, U, EE trajs
    INPUT 
      ddp       : DDP solver or list of ddp solvers
      robot     : pinocchio robot wrapper
      id_endeff : frame id of endeffector 
    '''
    if(type(ddp) != list):
        ddp = [ddp]
    for k,d in enumerate(ddp):
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all'):
                fig_x, ax_x = plot_ddp_state(ddp[k], SHOW=False)
            if('u' in which_plots or which_plots =='all'):
                fig_u, ax_u = plot_ddp_control(ddp[k], SHOW=False)
            if('p' in which_plots or which_plots =='all'):
                fig_p, ax_p = plot_ddp_endeff(ddp[k], robot, id_endeff, SHOW=False)
            if('vxx' in which_plots or which_plots =='all'):
                fig_vxx_sv, ax_vxx_sv = plot_ddp_vxx_sv(ddp[k], SHOW=False)
                fig_vxx_eig, ax_vxx_eig = plot_ddp_vxx_eig(ddp[k], SHOW=False)
            if('K' in which_plots or which_plots =='all'):
                fig_K_sv, ax_K_sv = plot_ddp_ricatti_sv(ddp[k], SHOW=False)
                fig_K_eig, ax_K_eig = plot_ddp_ricatti_eig(ddp[k], SHOW=False)
        # Overlay on top of first plot
        else:
            if('x' in which_plots or which_plots =='all'):
                plot_ddp_state(ddp[k], fig=fig_x, ax=ax_x, SHOW=False)
            if('u' in which_plots or which_plots =='all'):
                plot_ddp_control(ddp[k], fig=fig_u, ax=ax_u, SHOW=False)
            if('p' in which_plots or which_plots =='all'):
                plot_ddp_endeff(ddp[k], robot, id_endeff, fig=fig_p, ax=ax_p, SHOW=False)
            if('vxx' in which_plots or which_plots =='all'):
                plot_ddp_vxx_sv(ddp[k], fig=fig_vxx_sv, ax=ax_vxx_sv, SHOW=False)
                plot_ddp_vxx_eig(ddp[k], fig=fig_vxx_eig, ax=ax_vxx_eig, SHOW=False)
            if('K' in which_plots or which_plots =='all'):
                plot_ddp_ricatti_sv(ddp[k], fig=fig_K_sv, ax=ax_K_sv, SHOW=False)
                plot_ddp_ricatti_eig(ddp[k], fig=fig_K_eig, ax=ax_K_eig, SHOW=False)
    
    plt.show()

def plot_ddp_state(ddp, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (state)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    # Extract pos, vel trajs
    x = np.array(ddp.xs)
    q = x[:,:nq]
    v = x[:,nv:]
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 2, sharex='col') 
    if(label is None):
        label='State'
    for i in range(nq):
        # Positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker='o', label=label)
        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].grid(True)
        # Velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker='o', label=label)
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].grid(True)

    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('State trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control(ddp, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (control)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nu = ddp.problem.runningModels[0].nu
    # Extract pos, vel trajs
    u = np.array(ddp.us)
    # Plots
    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col') 
    if(label is None):
        label='Control'    
    for i in range(nu):
        # Positions
        ax[i].plot(tspan, u[:,i], linestyle='-', marker='o', label=label)
        ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax[i].grid(True)
        # Set xlabel on bottom plot
        if(i == nu-1):
            ax[i].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Control trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff(ddp, robot, id_endeff, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (endeff)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    # Extract EE traj
    x = np.array(ddp.xs)
    q = x[:,:nq]
    p = pin_utils.get_p(q, robot, id_endeff)
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 1, sharex='col')
    if(label is None):
        label='End-effector'
    ylabels = ['Px', 'Py', 'Pz']
    for i in range(3):
        # Positions
        ax[i].plot(tspan, q[:,i], linestyle='-', marker='o', label=label)
        ax[i].set_ylabel(ylabel=ylabels[i], fontsize=16)
        ax[i].grid(True)
    handles, labels = ax[i].get_legend_handles_labels()
    ax[i].set_xlabel('t (s)', fontsize=16)
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Endeffector trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_vxx_sv(ddp, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (vxx singular values)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    nx = nq+nv
    nx2 = nx//2
    Vxx_sv = np.zeros((N, nq+nv)) 
    # Extract singular values and eigenvalues of VF Hessian
    for i in range(N):
        _, sv, _ = np.linalg.svd(ddp.Vxx[i])
        Vxx_sv[i, :] = np.sort(sv)[::-1]
    # Plots
    tspan = np.linspace(0, N*dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nx2, 2, sharex='col')
    if(label is None):
        label='Vxx Singular Values'
    for i in range(nx2):
        # Singular values 0 to 6
        ax[i,0].plot(tspan, Vxx_sv[:,i], linestyle='-', marker='o', label=label)
        ax[i,0].set_ylabel('$\sigma_{%s}$'%i, fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        # Eigenvalues 7 to 13
        ax[i,1].plot(tspan, Vxx_sv[:,nx2+i], linestyle='-', marker='o', label=label)
        ax[i,1].set_ylabel('$\sigma_{%s}$'%str(nx2+i), fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)
        # Set xlabel on bottom plot
        if(i == nx2-1):
            ax[i,0].set_xlabel('t (s)', fontsize=16)
            ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Vxx Singular Values', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_vxx_eig(ddp, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (vxx eigenvalues)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    nx = nq+nv
    nx2 = nx//2
    Vxx_eig = np.zeros((N, nx))
    # Extract singular values VF Hessian
    for i in range(N):
        Vxx_eig[i, :] = np.linalg.eigvals(ddp.Vxx[i])
    # Plots
    tspan = np.linspace(0, N*dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nx2, 2, sharex='col')
    if(label is None):
        label='Vxx Eigenvalues'
    # ax_ylim = np.max(Vxx_eig)
    for i in range(nx2):
        # Eigenvalues 0 to 6
        ax[i,0].plot(tspan, Vxx_eig[:,i], linestyle='-', marker='o', label=label)
        ax[i,0].set_ylabel('$\lambda_{%s}$'%i, fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        # Eigenvalues 7 to 13
        ax[i,1].plot(tspan, Vxx_eig[:,nx2+i], linestyle='-', marker='o', label=label)
        ax[i,1].set_ylabel('$\lambda_{%s}$'%str(nx2+i), fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)
        # Set xlabel on bottom plot
        if(i == nx2-1):
            ax[i,0].set_xlabel('t (s)', fontsize=16)
            ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Vxx Eigenvalues', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_ricatti_sv(ddp, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (K sing vals)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    nx = nq+nv
    nx2 = nx//2
    # K_diag = np.zeros((N, nx))
    # K_eig = np.zeros((N, nx))
    K_sv = np.zeros((N, nq))
    # Extract diag , eig and sing val of Ricatti gain
    for i in range(N):
        _, K_sv[i, :], _ = np.linalg.svd(ddp.K[i][:nq,:nq])
    # Plots
    tspan = np.linspace(0, N*dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nx2, 1, sharex='col')
    if(label is None):
        label='K singular values'
    for i in range(nx2):
        # Singular values
        ax[i].plot(tspan, K_sv[:,i], linestyle='-', marker='o', label=label)
        ax[i].set_ylabel('$\sigma_{%s}$'%str(i), fontsize=16)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i].grid(True)
        # Set xlabel on bottom plot
        if(i == nx2-1):
            ax[i].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('K singular values', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_ricatti_eig(ddp, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (K sing vals)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    nx = nq+nv
    nx2 = nx//2
    # K_diag = np.zeros((N, nx))
    K_eig = np.zeros((N, nx))
    # Extract diag , eig and sing val of Ricatti gain
    for i in range(N):
        K_eig[i, :nq] = np.sort(np.linalg.eigvals(ddp.K[i][:nq,:nq]))[::-1]
        K_eig[i, nv:] = np.sort(np.linalg.eigvals(ddp.K[i][:nq,nv:]))[::-1]
    # Plots
    tspan = np.linspace(0, N*dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nx2, 2, sharex='col')
    if(label is None):
        label='K eigenvalues'
    for i in range(nx2):
        # Eigenvalues
        ax[i,0].plot(tspan, K_eig[:,i], linestyle='-', marker='o', label=label)
        ax[i,0].set_ylabel('$\lambda_{%s}$'%str(nx2+i), fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax[i,0].grid(True)
        # Eigenvalues
        ax[i,1].plot(tspan, K_eig[:,nx2+i], linestyle='-', marker='o', label=label)
        ax[i,1].set_ylabel('$\lambda_{%s}$'%str(nx2+i), fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax[i,1].grid(True)
        # Set xlabel on bottom plot
        if(i == nx2-1):
            ax[i,0].set_xlabel('t (s)', fontsize=16)
            ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Ricatti gain eigenvalues', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_ricatti_diag(ddp, fig=None, ax=None, label=None, SHOW=True):
    '''
    Plot ddp results (K diag)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    nx = nq+nv
    nx2 = nx//2
    # K_diag = np.zeros((N, nx))
    K_diag = np.zeros((N, nx))
    # Extract diag , eig and sing val of Ricatti gain
    for i in range(N):
        K_diag[i, :nq] = ddp.K[i][:nq,:nq].diagonal()
        K_diag[i, nv:] = ddp.K[i][:nq,nv:].diagonal()
    # Plots
    tspan = np.linspace(0, N*dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nx2, 2, sharex='col')
    if(label is None):
        label='K diagonal terms'
    for i in range(nx2):
        # Diagonal terms
        ax[i,0].plot(tspan, K_diag[:,i], linestyle='-', marker='o', label=label)
        ax[i,0].set_ylabel('$diag_{%s}$'%i, fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax[i,0].grid(True)
        # Set xlabel on bottom plot
        if(i == nx2-1):
            ax[i,0].set_xlabel('t (s)', fontsize=16)
            ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Ricatti gain diagonal', size=16)
    if(SHOW):
        plt.show()
    return fig, ax



def plot_ddp_results_LPF(ddp, robot, id_endeff):
    '''
    Plot results of DDP solver with stateLPF
    '''
    # Parameters
    N_h = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    nu = nq
    # Extract solution trajectories
    xs = np.array(ddp.xs)
    us = np.array(ddp.us)
    q = np.empty((N_h+1, nq))
    v = np.empty((N_h+1, nv))
    u = np.empty((N_h+1, nq))
    for i in range(N_h+1):
        q[i,:] = xs[i][:nq].T
        v[i,:] = xs[i][nv:nv+nq].T
        u[i,:] = xs[i][nv+nq:].T
    p_ee = pin_utils.get_p(q, robot, id_endeff)
    w = np.empty((N_h, nu))
    for i in range(N_h):
        w[i,:] = us[i].T
    print("Plot results...")
    # Create time spans for X and U + figs and subplots
    tspan_x = np.linspace(0, N_h*dt, N_h+1)
    tspan_w = np.linspace(0, N_h*dt, N_h)
    fig_x, ax_x = plt.subplots(nq, 3)
    fig_w, ax_w = plt.subplots(nq, 1)
    fig_p, ax_p = plt.subplots(3, 1)
    # Plot joints pos, vel , acc, torques
    for i in range(nq):
        # Positions
        ax_x[i,0].plot(tspan_x, q[:,i], 'b.', label='pos_des')
        ax_x[i,0].plot(tspan_x[-1], q[-1,i], 'ro')
        ax_x[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax_x[i,0].grid()
        # Velocities
        ax_x[i,1].plot(tspan_x, v[:,i], 'b.', label='vel_des')
        ax_x[i,1].plot(tspan_x[-1], v[-1,i], 'ro')
        ax_x[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax_x[i,1].grid()
        # Torques
        ax_x[i,2].plot(tspan_x, u[:,i], 'b.', label='torque_des')
        ax_x[i,2].plot(tspan_x[-1], u[-1,i], 'ro')
        ax_x[i,2].set_ylabel('$u_%s$'%i, fontsize=16)
        ax_x[i,2].grid()
        # Input (w)
        ax_w[i].plot(tspan_w, w[:,i], 'b.', label='input_des') 
        ax_w[i].set_ylabel(ylabel='$w_%d$'%i, fontsize=16)
        ax_w[i].grid()
        # Remove xticks labels for clarity 
        if(i != nq-1):
            for j in range(3):
                ax_x[i,j].set_xticklabels([])
            ax_w[i].set_xticklabels([])
        # Set xlabel on bottom plot
        if(i == nq-1):
            for j in range(3):
                ax_x[i,j].set_xlabel('t (s)', fontsize=16)
            ax_w[i].set_xlabel('t (s)', fontsize=16)
        # Legend
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
        handles_w, labels_w = ax_w[i].get_legend_handles_labels()
        fig_w.legend(handles_w, labels_w, loc='upper right', prop={'size': 16})
    # Plot EE
    ylabels_p = ['Px', 'Py', 'Pz']
    for i in range(3):
        ax_p[i].plot(tspan_x, p_ee[:,i], 'b.', label='desired')
        ax_p[i].set_ylabel(ylabel=ylabels_p[i], fontsize=16)
        ax_p[i].grid()
        handles_p, labels_p = ax_p[i].get_legend_handles_labels()
        fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    ax_p[-1].set_xlabel('t (s)', fontsize=16)
    # Align labels + set titles
    fig_x.align_ylabels()
    fig_w.align_ylabels()
    fig_p.align_ylabels()
    fig_x.suptitle('Joint trajectories', size=16)
    fig_w.suptitle('Joint input', size=16)
    fig_p.suptitle('End-effector trajectory', size=16)
    plt.show()


# OLD

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

