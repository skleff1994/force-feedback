from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# Plot state data
def plot_state(plot_data, PLOT_PREDICTIONS=False, 
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
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
    t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
    fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8))
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
        
        # Legend
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
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
def plot_control(plot_data, PLOT_PREDICTIONS=False, 
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
    fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8))
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
          # Desired  
        ax_u[i].plot(t_span_plan_u, plot_data['u_des'][:,i], 'b-', label='Desired')
          # Measured
        ax_u[i].plot(t_span_simu_u, plot_data['u_mea'][:,i], 'r-', label='Measured') 
        ax_u[i].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
        ax_u[i].grid()

        handles_u, labels_u = ax_u[i].get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
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
def plot_endeff(plot_data, PLOT_PREDICTIONS=False, 
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
    fig_p, ax_p = plt.subplots(3,1, figsize=(19.2,10.8)) 
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
def plot_acc_err(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    fig_a, ax_a = plt.subplots(nq,2, figsize=(19.2,10.8))
    # For each joint
    for i in range(nq):

        # Joint velocity error (avg over 1 control cycle)
        ax_a[i,0].plot(t_span_ctrl_u, plot_data['a_err'][:,i], 'b-', label='Velocity error (average)')
        ax_a[i,0].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
        ax_a[i,0].grid()
        # Joint acceleration error (avg over 1 control cycle)
        ax_a[i,1].plot(t_span_ctrl_u, plot_data['a_err'][:,nq+i], 'b-', label='Acceleration error (average)')
        ax_a[i,1].set(xlabel='t (s)', ylabel='$u_{}$ (Nm)'.format(i))
        ax_a[i,1].grid()
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

# Plot Ricatti
def plot_ricatti(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    print('Plotting Ricatti data...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_K, ax_K = plt.subplots(nq, 2, figsize=(19.2,10.8))
    # For each joint
    for i in range(nq):
        # Ricatti gains diag
        ax_K[i,0].plot(t_span_plan_u, plot_data['K'][:,i,i], 'b-', label='Diag of Ricatti')
        ax_K[i,0].set(xlabel='t (s)', ylabel='$diag [K]_{}$'.format(i))
        ax_K[i,0].grid()
        # Ricatti gains singular values
        ax_K[i,1].plot(t_span_plan_u, plot_data['K_svd'][:,i], 'b-', label='Singular Value of Ricatti')
        ax_K[i,1].set(xlabel='t (s)', ylabel='$\sigma [K]_{}$'.format(i))
        ax_K[i,1].grid()
    # Titles
    fig_K.suptitle('Singular Values of Ricatti feedback gains K', size=16)
    # Save figs
    if(SAVE):
        figs = {'K': fig_K}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_K

# Plot Vxx
def plot_Vxx(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                        SHOW=True):
    '''
    Plot Vxx data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    print('Plotting Vxx data...')
    T_tot = plot_data['T_tot']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']

    # Create time spans for X and U + Create figs and subplots
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_V, ax_V = plt.subplots(nq, 2, figsize=(19.2,10.8))
    # For each state
    for i in range(nq):
        # Vxx diag
        # ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_diag'][:,i], 'b-', label='Vxx diagonal')
        # ax_V[i,0].set(xlabel='t (s)', ylabel='$Diag[Vxx_{}]$'.format(i,i))
        # ax_V[i,0].grid()
        # Vxx eigenvals
        ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_eigval'][:,i], 'b-', label='Vxx eigenvalue')
        ax_V[i,0].set(xlabel='t (s)', ylabel='$\lambda_{}$'.format(i)+'(Vxx)')
        ax_V[i,0].grid()
        # Vxx eigenvals
        ax_V[i,1].plot(t_span_plan_u, plot_data['Vxx_eigval'][:,nq+i], 'b-', label='Vxx eigenvalue')
        ax_V[i,1].set(xlabel='t (s)', ylabel='$\lambda_{}$'.format(nq+i)+'(Vxx)')
        ax_V[i,1].grid()
    # Titles
    fig_V.suptitle('Eigenvalues of Value Function Hessian Vxx', size=16)
    # Save figs
    if(SAVE):
        figs = {'V': fig_V}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_V

# Plot Solver regs
def plot_solver(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    fig_S, ax_S = plt.subplots(2, 1, figsize=(19.2,10.8))
    # Xreg
    ax_S[0].plot(t_span_plan_u, plot_data['xreg'], 'b-', label='xreg')
    ax_S[0].set(xlabel='t (s)', ylabel='$xreg$')
    ax_S[0].grid()
    # Ureg
    ax_S[1].plot(t_span_plan_u, plot_data['ureg'], 'r-', label='ureg')
    ax_S[1].set(xlabel='t (s)', ylabel='$ureg$')
    ax_S[1].grid()
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
def plot_jacobian(plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
    fig_J, ax_J = plt.subplots(1, 1, figsize=(19.2,10.8))
    # Rank of Jacobian
    ax_J.plot(t_span_plan_u, plot_data['J_rank'], 'b-', label='rank')
    ax_J.set(xlabel='t (s)', ylabel='rank')
    ax_J.grid()
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
def plot_results(plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
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
        plots['x'] = plot_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                           pred_plot_sampling=pred_plot_sampling, 
                                           SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                           SHOW=False)
    
    if('u' in which_plots or which_plots is None or which_plots =='all'):
        plots['u'] = plot_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                             pred_plot_sampling=pred_plot_sampling, 
                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False)
    
    if('a' in which_plots or which_plots is None or which_plots =='all'):
        plots['a'] = plot_acc_err(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=SHOW)

    if('p' in which_plots or which_plots is None or which_plots =='all'):
        plots['p'] = plot_endeff(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)

    if('K' in which_plots or which_plots is None or which_plots =='all'):
        plots['K'] = plot_ricatti(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False)

    if('V' in which_plots or which_plots is None or which_plots =='all'):
        plots['V'] = plot_Vxx(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                         SHOW=False)

    if('S' in which_plots or which_plots is None or which_plots =='all'):
        plots['S'] = plot_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)

    if('J' in which_plots or which_plots is None or which_plots =='all'):
        plots['J'] = plot_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)

    if(SHOW):
        plt.show() 
    plt.close('all')




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

