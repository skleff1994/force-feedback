"""
@package force_feedback
@file classical_mpc/plot_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Plot OCP solution
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


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
            if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_ee_lin, ax_ee_lin = plot_ddp_endeff_linear(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                    fig_ee_ang, ax_ee_ang = plot_ddp_endeff_angular(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
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
                if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_endeff_linear(data, fig=fig_ee_lin, ax=ax_ee_lin, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
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
    if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('xs' in data.keys()):
            fig['ee_lin'] = fig_ee_lin
            ax['ee_lin'] = ax_ee_lin
            fig['ee_ang'] = fig_ee_ang
            ax['ee_ang'] = ax_ee_ang
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

def plot_ddp_endeff_linear(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                    MAKE_LEGEND=False, SHOW=True, AUTOSCALE=True):
    '''
    Plot ddp results (endeff linear position, velocity)
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
    lin_pos_ee = pin_utils.get_p_(q, ddp_data['pin_model'], ddp_data['frame_id'])
    lin_vel_ee = pin_utils.get_v_(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    # Cost reference frame translation if any, or initial one
    if('translation' in ddp_data['active_costs'] or 'placement' in ddp_data['active_costs']):
        lin_pos_ee_ref = np.array(ddp_data['translation_ref'])
    else:
        lin_pos_ee_ref = np.array([lin_pos_ee[0,:] for i in range(N+1)])
    # Cost reference frame linear velocity if any, or initial one
    if('velocity' in ddp_data['active_costs']):
        lin_vel_ee_ref = np.array(ddp_data['velocity_ref'])[:,:3] # linear part
    else:
        lin_vel_ee_ref = np.array([lin_vel_ee[0,:] for i in range(N+1)])
    # Contact reference translation if CONTACT
    if(ddp_data['CONTACT_TYPE'] is not None):
        lin_pos_ee_contact = np.array(ddp_data['contact_translation'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col')
    if(label is None):
        label='OCP solution'
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE position in WORLD frame
        ax[i,0].plot(tspan, lin_pos_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
        # Plot EE target frame translation in WORLD frame
        if('translation' or 'placement' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,0].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,0].plot(tspan, lin_pos_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        # Plot CONTACT reference frame translation in WORLD frame
        if(ddp_data['CONTACT_TYPE'] is not None):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('Baumgarte stab. ref.' in labels):
                handles.pop(labels.index('Baumgarte stab. ref.'))
                ax[i,0].lines.pop(labels.index('Baumgarte stab. ref.'))
                labels.remove('Baumgarte stab. ref.')
            ax[i,0].plot(tspan, lin_pos_ee_contact[:,i], linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)
        # Labels, tick labels, grid
        ax[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot EE (linear) velocities in WORLD frame
        ax[i,1].plot(tspan, lin_vel_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
        # Plot EE target frame (linear) velocity in WORLD frame
        if('velocity' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,1].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,1].plot(tspan, lin_vel_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
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
        ax_p_ylim = 1  #1.1*max(np.max(np.abs(lin_pos_ee)), TOL)
        ax_v_ylim = 1 #1.1*max(np.max(np.abs(lin_vel_ee)), TOL)
        for i in range(3):
            ax[i,0].set_ylim(lin_pos_ee_ref[0,i]-ax_p_ylim, lin_pos_ee_ref[0,i]+ax_p_ylim) 
            ax[i,1].set_ylim(lin_vel_ee_ref[0,i]-ax_v_ylim, lin_vel_ee_ref[0,i]+ax_v_ylim)

    if(MAKE_LEGEND):
        handles, labels = ax[2,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector frame position and linear velocity', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff_angular(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                    MAKE_LEGEND=False, SHOW=True, AUTOSCALE=False):
    '''
    Plot ddp results (endeff angular position, velocity)
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
    rpy_ee = pin_utils.get_rpy_(q, ddp_data['pin_model'], ddp_data['frame_id'])
    w_ee   = pin_utils.get_w_(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    # Cost reference frame orientation if any, or initial one
    if('rotation' in ddp_data['active_costs'] or 'placement' in ddp_data['active_costs']):
        rpy_ee_ref = np.array([pin.utils.matrixToRpy(np.array(R)) for R in ddp_data['rotation_ref']])
    else:
        rpy_ee_ref = np.array([rpy_ee[0,:] for i in range(N+1)])
    # Cost reference angular velocity if any, or initial one
    if('velocity' in ddp_data['active_costs']):
        w_ee_ref = np.array(ddp_data['velocity_ref'])[:,3:] # angular part
    else:
        w_ee_ref = np.array([w_ee[0,:] for i in range(N+1)])
    # Contact reference orientation (6D)
    if(ddp_data['CONTACT_TYPE']=='6D'):
        rpy_ee_contact = np.array([pin.utils.matrixToRpy(R) for R in ddp_data['contact_rotation']])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col')
    if(label is None):
        label='OCP solution'
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE orientation in WORLD frame
        ax[i,0].plot(tspan, rpy_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot EE target frame orientation in WORLD frame
        if('rotation' or 'placement' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,0].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,0].plot(tspan, rpy_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        
        # Plot CONTACT reference frame rotation in WORLD frame
        if(ddp_data['CONTACT_TYPE']=='6D'):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('contact' in labels):
                handles.pop(labels.index('contact'))
                ax[i,0].lines.pop(labels.index('contact'))
                labels.remove('contact')
            ax[i,0].plot(tspan, rpy_ee_contact[:,i], linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)

        # Labels, tick labels, grid
        ax[i,0].set_ylabel('$RPY^{EE}_%s$ (rad)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot EE 'linear) velocities in WORLD frame
        ax[i,1].plot(tspan, w_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot EE target frame (linear) velocity in WORLD frame
        if('velocity' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,1].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,1].plot(tspan, w_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        
        # Labels, tick labels, grid
        ax[i,1].set_ylabel('$W^{EE}_%s$ (rad/s)'%xyz[i], fontsize=16)
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
        ax_p_ylim = 1.1*max(np.max(np.abs(rpy_ee)), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(w_ee)), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, +ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, +ax_v_ylim)

    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector frame orientation and angular velocity', size=18)
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