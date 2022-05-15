
"""
@package force_feedback
@file lpf_mpc/init_data.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initialize / extract data for MPC simulation
"""

import numpy as np
from utils import pin_utils
from classical_mpc.data import DDPDataParserClassical

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from utils.data_utils import MPCDataHandlerAbstract

class DDPDataParserLPF(DDPDataParserClassical):
  def __init__(self, ddp):
    super().__init__(ddp)

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



class MPCDataHandlerLPF(MPCDataHandlerAbstract):

  def __init__(self, config, robot):
    super().__init__(config, robot)
    self.ny = self.nx + self.nu

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


  # Extract MPC simu-specific plotting data from sim data
  def extract_plot_data_from_sim_data(self, frame_of_interest):
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
    print(plot_data['pin_model'])
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