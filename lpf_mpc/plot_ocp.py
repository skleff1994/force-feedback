"""
@package force_feedback
@file lpf_mpc/plot_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Plot OCP solution
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import pin_utils

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

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
            if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_ee_lin, ax_ee_lin = plot_ddp_endeff_LPF_linear(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                    fig_ee_ang, ax_ee_ang = plot_ddp_endeff_LPF_angular(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
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
                if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_endeff_LPF_linear(data, fig=fig_ee_lin, ax=ax_ee_lin, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                        plot_ddp_endeff_LPF_angular(data, fig=fig_ee_ang, ax=ax_ee_ang, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
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
            ax[i,2].plot(tspan, ureg_grav[:,i], linestyle='-.', color=[0.,1.,0.,0.], marker=None, label='u_grav(q)', alpha=0.5)
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
        w_reg_ref[i,:] = pin_utils.get_u_grav(q[i,:], ddp_data['pin_model'], ddp_data['armature'])
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

def plot_ddp_endeff_LPF_linear(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (endeff)
    '''
    return plot_ddp_endeff_linear(ddp_data, fig=fig, ax=ax, label=label, marker=marker, color=color, alpha=alpha, MAKE_LEGEND=MAKE_LEGEND, SHOW=SHOW)

def plot_ddp_endeff_LPF_angular(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (endeff)
    '''
    return plot_ddp_endeff_angular(ddp_data, fig=fig, ax=ax, label=label, marker=marker, color=color, alpha=alpha, MAKE_LEGEND=MAKE_LEGEND, SHOW=SHOW)

def plot_ddp_force_LPF(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (force)
    '''
    return plot_ddp_force(ddp_data, fig=fig, ax=ax, label=label, marker=marker, color=color, alpha=alpha, MAKE_LEGEND=MAKE_LEGEND, SHOW=SHOW)
