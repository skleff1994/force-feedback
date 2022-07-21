
"""
@package force_feedback
@file lpf_mpc/data.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initialize / extract data for MPC simulation
"""

import numpy as np
from core_mpc import pin_utils
from core_mpc.data import DDPDataHandlerAbstract, MPCDataHandlerAbstract

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib


import pinocchio as pin

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# Classical OCP data handler : extract data + generate fancy plots
class DDPDataHandlerLPF(DDPDataHandlerAbstract):
  def __init__(self, ddp, n_lpf):
    super().__init__(ddp)
    self.n_lpf = n_lpf

  def extract_data(self, ee_frame_name, ct_frame_name):
    '''
    extract data to plot
    '''
    ddp_data = super().extract_data(ee_frame_name, ct_frame_name)
    # Add terminal regularization references on filtered torques
    if('ctrlReg' in ddp_data['active_costs']):
        ddp_data['ctrlReg_ref'].append(self.ddp.problem.terminalModel.differential.costs.costs['ctrlReg'].cost.residual.reference)
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ddp_data['ctrlRegGrav_ref'].append(pin_utils.get_u_grav(self.ddp.xs[-1][:ddp_data['nq']], ddp_data['pin_model'], ddp_data['armature']))
    return ddp_data

  def plot_ddp_results(self, DDP_DATA, which_plots='all', labels=None, markers=None, colors=None, sampling_plot=1, SHOW=False):
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
                      fig_x, ax_x = self.plot_ddp_state(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
              if('w' in which_plots or which_plots =='all' or 'all' in which_plots):
                  if('us' in data.keys()):
                      fig_u, ax_u = self.plot_ddp_control(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
              if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                  if('xs' in data.keys()):
                      fig_ee_lin, ax_ee_lin = self.plot_ddp_endeff_linear(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                      fig_ee_ang, ax_ee_ang = self.plot_ddp_endeff_angular(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
              if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                  if('fs' in data.keys()):
                      fig_f, ax_f = self.plot_ddp_force(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
          else:
              if(k%sampling_plot==0):
                  if('y' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('xs' in data.keys()):
                          self.plot_ddp_state(data, fig=fig_x, ax=ax_x, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                  if('w' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('us' in data.keys()):
                          self.plot_ddp_control(data, fig=fig_u, ax=ax_u, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                  if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('xs' in data.keys()):
                          self.plot_ddp_endeff_linear(data, fig=fig_ee_lin, ax=ax_ee_lin, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                          self.plot_ddp_endeff_angular(data, fig=fig_ee_ang, ax=ax_ee_ang, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                  if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('fs' in data.keys()):
                          self.plot_ddp_force_LPF(data, fig=fig_f, ax=ax_f, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
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

  def plot_ddp_state(self, ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
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
      tau = x[:,-self.n_lpf:]
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
      for i in range(nv):
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
      for i in range(self.n_lpf):
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

  def plot_ddp_control(self, ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
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




# LPF MPC data handler : initialize, extract data + generate fancy plots
class MPCDataHandlerLPF(MPCDataHandlerAbstract):

  def __init__(self, config, robot, n_lpf):
    super().__init__(config, robot)
    self.ny = self.nx + self.n_lpf

  # Allocate data 
  def init_predictions(self):
    '''
    Allocate data for state, control & force predictions
    '''
    self.state_pred     = np.zeros((self.N_plan, self.N_h+1, self.ny)) # Predicted states  ( self.ddp.xs : {x* = (q*, v*)} )
    self.ctrl_pred      = np.zeros((self.N_plan, self.N_h, self.nu))   # Predicted torques ( self.ddp.us : {u*} )
    self.force_pred     = np.zeros((self.N_plan, self.N_h, 6))         # Predicted EE contact forces
    self.state_des_PLAN = np.zeros((self.N_plan+1, self.ny))           # Predicted states at planner frequency  ( x* interpolated at PLAN freq )
    self.ctrl_des_PLAN  = np.zeros((self.N_plan, self.nu))             # Predicted torques at planner frequency ( u* interpolated at PLAN freq )
    self.force_des_PLAN = np.zeros((self.N_plan, 6))                   # Predicted EE contact forces planner frequency  
    self.state_des_CTRL = np.zeros((self.N_ctrl+1, self.ny))           # Reference state at motor drivers freq ( x* interpolated at CTRL freq )
    self.ctrl_des_CTRL  = np.zeros((self.N_ctrl, self.nu))             # Reference input at motor drivers freq ( u* interpolated at CTRL freq )
    self.force_des_CTRL = np.zeros((self.N_ctrl, 6))                   # Reference EE contact force at motor drivers freq
    self.state_des_SIMU = np.zeros((self.N_simu+1, self.ny))           # Reference state at actuation freq ( x* interpolated at SIMU freq )
    self.ctrl_des_SIMU  = np.zeros((self.N_simu, self.nu))             # Reference input at actuation freq ( u* interpolated at SIMU freq )
    self.force_des_SIMU = np.zeros((self.N_simu, 6))                   # Reference EE contact force at actuation freq

  def init_measurements(self, y0):
    '''
    Allocate data for simulation state & force measurements 
    '''
    self.state_mea_SIMU                = np.zeros((self.N_simu+1, self.ny))            # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq )
    self.state_mea_no_noise_SIMU       = np.zeros((self.N_simu+1, self.ny))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq ) without noise
    self.force_mea_SIMU                = np.zeros((self.N_simu, 6)) 
    self.state_mea_SIMU[0, :]          = y0
    self.state_mea_no_noise_SIMU[0, :] = y0

  def init_sim_data(self, y0):
    '''
    Allocate and initialize MPC simulation data
    '''
    # sim_data = {}
    # MPC & simulation parameters
    self.N_plan = int(self.T_tot*self.plan_freq)         # Total number of planning steps in the simulation
    self.N_ctrl = int(self.T_tot*self.ctrl_freq)         # Total number of control steps in the simulation 
    self.N_simu = int(self.T_tot*self.simu_freq)         # Total number of simulation steps 
    self.T_h = self.N_h*self.dt                          # Duration of the MPC horizon (s)
    self.dt_ctrl = float(1./self.ctrl_freq)              # Duration of 1 control cycle (s)
    self.dt_plan = float(1./self.plan_freq)              # Duration of 1 planning cycle (s)
    self.dt_simu = float(1./self.simu_freq)              # Duration of 1 simulation cycle (s)
    self.OCP_TO_PLAN_RATIO = self.dt_plan / self.dt
    # Init actuation model
    self.init_actuation_model()
    # Cost references 
    self.init_cost_references()
    # Predictions
    self.init_predictions()
    # Measurements
    self.init_measurements(y0)

    # DDP solver-specific data
    if(self.RECORD_SOLVER_DATA):
      self.init_solver_data()
   
    logger.info("Initialized MPC simulation data.")

    if(self.INIT_LOG):
      self.print_sim_params(self.init_log_display_time)


  def record_predictions(self, nb_plan, ddpSolver):
    '''
    - Records the MPC prediction of at the current step (state, control and forces if contact is specified)
    '''
    # logger.debug(str(np.shape(self.state_pred)))
    self.state_pred[nb_plan, :, :] = np.array(ddpSolver.xs)
    self.ctrl_pred[nb_plan, :, :] = np.array(ddpSolver.us)
    # Extract relevant predictions for interpolations to MPC frequency
    self.y_curr = self.state_pred[nb_plan, 0, :]    # y0* = measured state    (q^,  v^ , tau^ )
    self.y_pred = self.state_pred[nb_plan, 1, :]    # y1* = predicted state   (q1*, v1*, tau1*) 
    self.w_curr = self.ctrl_pred[nb_plan, 0, :]     # w0* = optimal control   (w0*) !! UNFILTERED TORQUE !!
    # Record forces in the right frame
    if(self.is_contact):
        id_endeff = self.rmodel.getFrameId(self.contactFrameName)
        if(self.PIN_REF_FRAME == pin.LOCAL):
            self.force_pred[nb_plan, :, :] = \
                np.array([ddpSolver.problem.runningDatas[i].differential.multibody.contacts.contacts[self.contactFrameName].f.vector for i in range(self.N_h)])
        elif(self.PIN_REF_FRAME == pin.LOCAL_WORLD_ALIGNED or self.PIN_REF_FRAME == pin.WORLD):
            self.force_pred[nb_plan, :, :] = \
                np.array([self.rdata.oMf[id_endeff].action @ ddpSolver.problem.runningDatas[i].differential.multibody.contacts.contacts[self.contactFrameName].f.vector for i in range(self.N_h)])
        else:
            logger.error("The Pinocchio reference frame must be in ['LOCAL', LOCAL_WORLD_ALIGNED', 'WORLD']")
        self.f_curr = self.force_pred[nb_plan, 0, :]
        self.f_pred = self.force_pred[nb_plan, 1, :]
  
  def record_plan_cycle_desired(self, nb_plan):
    '''
    - Records the planning cycle data (state, control, force)
    If an interpolation to planning frequency is needed, here is the place where to implement it
    '''
    if(nb_plan==0):
        self.state_des_PLAN[nb_plan, :] = self.y_curr  
    self.ctrl_des_PLAN[nb_plan, :]      = self.w_curr   
    self.state_des_PLAN[nb_plan+1, :]   = self.y_curr + self.OCP_TO_PLAN_RATIO * (self.y_pred - self.y_curr)    
    if(self.is_contact):
        self.force_des_PLAN[nb_plan, :] = self.f_curr + self.OCP_TO_PLAN_RATIO * (self.f_pred - self.f_curr)    

  def record_ctrl_cycle_desired(self, nb_ctrl):
    '''
    - Records the control cycle data (state, control, force)
    If an interpolation to control frequency is needed, here is the place where to implement it
    '''
    # Record stuff
    if(nb_ctrl==0):
        self.state_des_CTRL[nb_ctrl, :]   = self.y_curr  
    self.ctrl_des_CTRL[nb_ctrl, :]    = self.w_curr   
    self.state_des_CTRL[nb_ctrl+1, :] = self.y_curr + self.OCP_TO_PLAN_RATIO * (self.y_pred - self.y_curr)   
    if(self.is_contact):
        self.force_des_CTRL[nb_ctrl, :] =  self.f_curr + self.OCP_TO_PLAN_RATIO * (self.f_pred - self.f_curr)   

  def record_simu_cycle_desired(self, nb_simu):
    '''
    - Records the control cycle data (state, control, force)
    If an interpolation to control frequency is needed, here is the place where to implement it
    '''
    self.y_ref_SIMU  = self.y_curr + self.OCP_TO_PLAN_RATIO * (self.y_pred - self.y_curr)
    self.w_ref_SIMU  = self.w_curr 
    if(nb_simu==0):
        self.state_des_SIMU[nb_simu, :] = self.y_curr  
    self.ctrl_des_SIMU[nb_simu, :]   = self.w_ref_SIMU 
    self.state_des_SIMU[nb_simu+1, :] = self.y_ref_SIMU 
    if(self.is_contact):
        self.force_des_SIMU[nb_simu, :] =  self.f_curr + self.OCP_TO_PLAN_RATIO * (self.f_pred - self.f_curr)  
    return 


  # Extract MPC simu-specific plotting data from sim data
  def extract_data(self, frame_of_interest):
    '''
    Extract plot data from simu data
    '''
    logger.info('Extracting plot data from simulation data...')
    
    plot_data = self.__dict__.copy()
    # Get costs
    plot_data['WHICH_COSTS'] = self.WHICH_COSTS
    # Robot model & params
    plot_data['pin_model'] = self.rmodel
    self.id_endeff = self.rmodel.getFrameId(frame_of_interest)
    nq = self.nq ; nv = self.nv ; nu = self.nq
    # Control predictions
    plot_data['w_pred'] = self.ctrl_pred
      # Extract 1st prediction
    plot_data['w_des_PLAN'] = self.ctrl_des_PLAN
    plot_data['w_des_CTRL'] = self.ctrl_des_CTRL
    plot_data['w_des_SIMU'] = self.ctrl_des_SIMU
    # State predictions (at PLAN freq)
    plot_data['q_pred']     = self.state_pred[:,:,:nq]
    plot_data['v_pred']     = self.state_pred[:,:,nq:nq+nv]
    plot_data['tau_pred']   = self.state_pred[:,:,-nu:]
    plot_data['q_des_PLAN'] = self.state_des_PLAN[:,:nq]
    plot_data['v_des_PLAN'] = self.state_des_PLAN[:,nq:nq+nv] 
    plot_data['tau_des_PLAN'] = self.state_des_PLAN[:,-nu:]
    plot_data['q_des_CTRL'] = self.state_des_CTRL[:,:nq] 
    plot_data['v_des_CTRL'] = self.state_des_CTRL[:,nq:nq+nv]
    plot_data['tau_des_CTRL'] = self.state_des_CTRL[:,-nu:]
    plot_data['q_des_SIMU'] = self.state_des_SIMU[:,:nq]
    plot_data['v_des_SIMU'] = self.state_des_SIMU[:,nq:nq+nv]
    plot_data['tau_des_SIMU'] = self.state_des_SIMU[:,-nu:] 
    # State measurements (at SIMU freq)
    plot_data['q_mea']          = self.state_mea_SIMU[:,:nq]
    plot_data['v_mea']          = self.state_mea_SIMU[:,nq:nq+nv]
    plot_data['tau_mea'] = self.state_mea_SIMU[:,-nu:]
    plot_data['q_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,:nq]
    plot_data['v_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,nq:nq+nv]
    plot_data['tau_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,-nu:]
    # Extract gravity torques
    plot_data['grav'] = np.zeros((self.N_simu+1, nq))
    # print(plot_data['pin_model'])
    for i in range(plot_data['N_simu']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'], self.armature)
    # EE predictions (at PLAN freq)
      # Linear position velocity of EE
    plot_data['lin_pos_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3))
    plot_data['lin_vel_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3))
      # Angular position velocity of EE
    plot_data['ang_pos_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3)) 
    plot_data['ang_vel_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3)) 
    for node_id in range(self.N_h+1):
        plot_data['lin_pos_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['lin_vel_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['ang_pos_ee_pred'][:, node_id, :] = pin_utils.get_rpy_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['ang_vel_ee_pred'][:, node_id, :] = pin_utils.get_w_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
    # EE measurements (at SIMU freq)
      # Linear
    plot_data['lin_pos_ee_mea']          = pin_utils.get_p_(plot_data['q_mea'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_mea']          = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel, self.id_endeff)
    plot_data['lin_pos_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    plot_data['lin_vel_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
      # Angular
    plot_data['ang_pos_ee_mea']          = pin_utils.get_rpy_(plot_data['q_mea'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_mea']          = pin_utils.get_w_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel, self.id_endeff)
    plot_data['ang_pos_ee_mea_no_noise'] = pin_utils.get_rpy_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    plot_data['ang_vel_ee_mea_no_noise'] = pin_utils.get_w_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    # EE des
      # Linear
    plot_data['lin_pos_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['lin_pos_ee_des_CTRL'] = pin_utils.get_p_(plot_data['q_des_CTRL'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_des_CTRL'] = pin_utils.get_v_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], self.rmodel, self.id_endeff)
    plot_data['lin_pos_ee_des_SIMU'] = pin_utils.get_p_(plot_data['q_des_SIMU'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_des_SIMU'] = pin_utils.get_v_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], self.rmodel, self.id_endeff)
      # Angular
    plot_data['ang_pos_ee_des_PLAN'] = pin_utils.get_rpy_(plot_data['q_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_des_PLAN'] = pin_utils.get_w_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['ang_pos_ee_des_CTRL'] = pin_utils.get_rpy_(plot_data['q_des_CTRL'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_des_CTRL'] = pin_utils.get_w_(plot_data['q_des_CTRL'], plot_data['v_des_CTRL'], self.rmodel, self.id_endeff)
    plot_data['ang_pos_ee_des_SIMU'] = pin_utils.get_rpy_(plot_data['q_des_SIMU'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_des_SIMU'] = pin_utils.get_w_(plot_data['q_des_SIMU'], plot_data['v_des_SIMU'], self.rmodel, self.id_endeff)
    # Extract EE force
    plot_data['f_ee_pred'] = self.force_pred
    plot_data['f_ee_mea'] = self.force_mea_SIMU
    plot_data['f_ee_des_PLAN'] = self.force_des_PLAN
    plot_data['f_ee_des_CTRL'] = self.force_des_CTRL
    plot_data['f_ee_des_SIMU'] = self.force_des_SIMU

    # Solver data (optional)
    if(self.RECORD_SOLVER_DATA):
      self.extract_solver_data(plot_data)
    
    return plot_data
    
  def extract_solver_data(self, plot_data):

    nq = self.nq ; nv = self.nv ; nu = nq ; ny = self.ny
    # Get SVD & diagonal of Ricatti + record in sim data
    plot_data['K_svd'] = np.zeros((self.N_plan, self.N_h, nq))
    plot_data['Kp_diag'] = np.zeros((self.N_plan, self.N_h, nq))
    plot_data['Kv_diag'] = np.zeros((self.N_plan, self.N_h, nv))
    plot_data['Ktau_diag'] = np.zeros((self.N_plan, self.N_h, nu))
    for i in range(self.N_plan):
      for j in range(self.N_h):
        plot_data['Kp_diag'][i, j, :] = self.K[i, j, :, :nq].diagonal()
        plot_data['Kv_diag'][i, j, :] = self.K[i, j, :, nq:nq+nv].diagonal()
        plot_data['Ktau_diag'][i, j, :] = self.K[i, j, :, -nu:].diagonal()
        _, sv, _ = np.linalg.svd(self.K[i, j, :, :])
        plot_data['K_svd'][i, j, :] = np.sort(sv)[::-1]
    # Get diagonal and eigenvals of Vxx + record in sim data
    plot_data['Vxx_diag'] = np.zeros((self.N_plan,self.N_h+1, ny))
    plot_data['Vxx_eig'] = np.zeros((self.N_plan, self.N_h+1, ny))
    for i in range(self.N_plan):
      for j in range(self.N_h+1):
        plot_data['Vxx_diag'][i, j, :] = self.Vxx[i, j, :, :].diagonal()
        plot_data['Vxx_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Vxx[i, j, :, :]))[::-1]
    # Get diagonal and eigenvals of Quu + record in sim data
    plot_data['Quu_diag'] = np.zeros((self.N_plan,self.N_h, nu))
    plot_data['Quu_eig'] = np.zeros((self.N_plan, self.N_h, nu))
    for i in range(self.N_plan):
      for j in range(self.N_h):
        plot_data['Quu_diag'][i, j, :] = self.Quu[i, j, :, :].diagonal()
        plot_data['Quu_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Quu[i, j, :, :]))[::-1]
    # Get Jacobian
    plot_data['J_rank'] = self.J_rank
    # Get solve regs
    plot_data['xreg'] = self.xreg
    plot_data['ureg'] = self.ureg


  # Plot data - classical OCP specific plotting functions
  def plot_mpc_results(self, plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
                                                pred_plot_sampling=100, 
                                                SAVE=False, SAVE_DIR='/tmp', SAVE_NAME=None,
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
          figs['y'], axes['y'] = self.plot_mpc_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False)
      
      if('w' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          figs['w'], axes['w'] = self.plot_mpc_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)

      if('ee' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          figs['ee_lin'], axes['ee_lin'] = self.plot_mpc_endeff_linear(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)
          figs['ee_ang'], axes['ee_ang'] = self.plot_mpc_endeff_angular(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)

      if('f' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          figs['f'], axes['f'] = self.plot_mpc_force(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)

      if('K' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('K_diag' in plot_data.keys()):
              figs['K_diag'], axes['K_diag'] = self.plot_mpc_ricatti_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)
          if('K_svd' in plot_data.keys()):
              figs['K_svd'], axes['K_svd'] = self.plot_mpc_ricatti_svd(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)

      if('V' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('V_diag' in plot_data.keys()):
              figs['V_diag'], axes['V_diag'] = self.plot_mpc_Vxx_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)
          if('V_eig' in plot_data.keys()):
              figs['V_eig'], axes['V_eig'] = self.plot_mpc_Vxx_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)

      if('S' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('S' in plot_data.keys()):
              figs['S'], axes['S'] = self.plot_mpc_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)

      if('J' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('J' in plot_data.keys()):
              figs['J'], axes['J'] = self.plot_mpc_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)

      if('Q' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('Q_diag' in plot_data.keys()):
              figs['Q_diag'], axes['Q_diag'] = self.plot_mpc_Quu_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)
          if('Q_eig' in plot_data.keys()):
              figs['Q_eig'], axes['Q_eig'] = self.plot_mpc_Quu_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)
      
      if(SHOW):
          plt.show() 
      
      return figs, axes

  def plot_mpc_state(self, plot_data, PLOT_PREDICTIONS=False, 
                                    pred_plot_sampling=100, 
                                    SAVE=False, SAVE_DIR='/tmp', SAVE_NAME=None,
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
                  tspan_y_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                  tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                  # Set up lists of (x,y) points for predicted positions and velocities
                  points_q = np.array([tspan_y_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  points_v = np.array([tspan_y_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  points_tau = np.array([tspan_y_pred, tau_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  # Set up lists of segments
                  segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                  segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                  segs_tau= np.concatenate([points_tau[:-1], points_tau[1:]], axis=1)
                  # Make collections segments
                  cm = plt.get_cmap('Greys_r') 
                  lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                  lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                  lc_tau = LineCollection(segs_tau, cmap=cm, zorder=-1)
                  lc_q.set_array(tspan_y_pred)
                  lc_v.set_array(tspan_y_pred) 
                  lc_tau.set_array(tspan_y_pred)
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
                  ax[i,0].scatter(tspan_y_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                  ax[i,1].scatter(tspan_y_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
                  ax[i,2].scatter(tspan_y_pred, tau_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 

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
          if('ctrlReg' in plot_data['WHICH_COSTS'] or 'ctrlRegGrav' in plot_data['WHICH_COSTS']):
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
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig, ax

  def plot_mpc_control(self, plot_data, PLOT_PREDICTIONS=False, 
                                      pred_plot_sampling=100, 
                                      SAVE=False, SAVE_DIR='/tmp', SAVE_NAME=None,
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
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 

      return fig, ax

  def plot_mpc_ricatti_diag(self, plot_data, SAVE=False, SAVE_DIR='/tmp', SAVE_NAME=None,
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
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_K

  def plot_mpc_Vxx_eig(self, plot_data, SAVE=False, SAVE_DIR='/tmp', SAVE_NAME=None,
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
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_V

  def plot_mpc_Vxx_diag(self, plot_data, SAVE=False, SAVE_DIR='/tmp', SAVE_NAME=None,
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
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_V

