
"""
@package force_feedback
@file classical_mpc/init_data.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initialize / extract data for MPC simulation
"""

import time
import numpy as np
from utils import pin_utils
import pinocchio as pin
from utils.data_utils import MPCDataHandlerAbstract

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


#### Classical OCP
class DDPDataParserClassical:
  def __init__(self, ddp):

    self.ddp = ddp

  def extract_data(self, ee_frame_name, ct_frame_name):
    '''
    extract data to plot
    '''
    logger.info("Extracting DDP data...")
    # Store data
    ddp_data = {}
    # OCP params
    ddp_data['T'] = self.ddp.problem.T
    ddp_data['dt'] = self.ddp.problem.runningModels[0].dt
    ddp_data['nq'] = self.ddp.problem.runningModels[0].state.nq
    ddp_data['nv'] = self.ddp.problem.runningModels[0].state.nv
    ddp_data['nu'] = self.ddp.problem.runningModels[0].differential.actuation.nu
    ddp_data['nx'] = self.ddp.problem.runningModels[0].state.nx
    # Pin model
    ddp_data['pin_model'] = self.ddp.problem.runningModels[0].differential.pinocchio
    ddp_data['armature'] = self.ddp.problem.runningModels[0].differential.armature
    ddp_data['frame_id'] = ddp_data['pin_model'].getFrameId(ee_frame_name)
    # Solution trajectories
    ddp_data['xs'] = self.ddp.xs
    ddp_data['us'] = self.ddp.us
    ddp_data['CONTACT_TYPE'] = None
    # Extract force at EE frame and contact info
    if(hasattr(self.ddp.problem.runningModels[0].differential, 'contacts')):
      # Get refs for contact model
      contactModelRef0 = self.ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.reference
      # Case 6D contact (x,y,z,Ox,Oy,Oz)
      if(hasattr(contactModelRef0, 'rotation')):
        ddp_data['contact_rotation'] = [self.ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.rotation for i in range(self.ddp.problem.T)]
        ddp_data['contact_rotation'].append(self.ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.rotation)
        ddp_data['contact_translation'] = [self.ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.translation for i in range(self.ddp.problem.T)]
        ddp_data['contact_translation'].append(self.ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.translation)
        ddp_data['CONTACT_TYPE'] = '6D'
        PIN_REF_FRAME = pin.LOCAL
      # Case 3D contact (x,y,z)
      elif(np.size(contactModelRef0)==3):
        if(self.ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 3):
          # Get ref translation for 3D 
          ddp_data['contact_translation'] = [self.ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(self.ddp.problem.T)]
          ddp_data['contact_translation'].append(self.ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
          ddp_data['CONTACT_TYPE'] = '3D'
        elif(self.ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 1):
          # Case 1D contact
          ddp_data['contact_translation'] = [self.ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(self.ddp.problem.T)]
          ddp_data['contact_translation'].append(self.ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
          ddp_data['CONTACT_TYPE'] = '1D'
        else: 
          print(self.ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 3)
          logger.error("Contact must be 1D or 3D !")
        # Check which reference frame is used 
        if(self.ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.type == pin.pinocchio_pywrap.ReferenceFrame.LOCAL):
          PIN_REF_FRAME = pin.LOCAL
        else:
          PIN_REF_FRAME = pin.LOCAL_WORLD_ALIGNED
      # Get contact force
      datas = [self.ddp.problem.runningDatas[i].differential.multibody.contacts.contacts[ct_frame_name] for i in range(self.ddp.problem.T)]
      # data.f = force exerted at parent joint expressed in WORLD frame (oMi)
      # express it in LOCAL contact frame using jMf 
      ee_forces = [data.jMf.actInv(data.f).vector for data in datas] 
      ddp_data['fs'] = [ee_forces[i] for i in range(self.ddp.problem.T)]
      # Express in WORLD aligned frame otherwise
      if(PIN_REF_FRAME == pin.LOCAL_WORLD_ALIGNED or PIN_REF_FRAME == pin.WORLD):
        ct_frame_id = ddp_data['pin_model'].getFrameId(ct_frame_name)
        Ms = [pin_utils.get_SE3_(self.ddp.xs[i][:ddp_data['nq']], ddp_data['pin_model'], ct_frame_id) for i in range(self.ddp.problem.T)]
        ddp_data['fs'] = [Ms[i].action @ ee_forces[i] for i in range(self.ddp.problem.T)]
    # Extract refs for active costs 
    # TODO : active costs may change along horizon : how to deal with that when plotting? 
    ddp_data['active_costs'] = self.ddp.problem.runningModels[0].differential.costs.active.tolist()
    if('stateReg' in ddp_data['active_costs']):
        ddp_data['stateReg_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['stateReg'].cost.residual.reference for i in range(self.ddp.problem.T)]
        ddp_data['stateReg_ref'].append(self.ddp.problem.terminalModel.differential.costs.costs['stateReg'].cost.residual.reference)
    if('ctrlReg' in ddp_data['active_costs']):
        ddp_data['ctrlReg_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].cost.residual.reference for i in range(self.ddp.problem.T)]
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ddp_data['ctrlRegGrav_ref'] = [pin_utils.get_u_grav(self.ddp.xs[i][:ddp_data['nq']], ddp_data['pin_model'], ddp_data['armature']) for i in range(self.ddp.problem.T)]
    if('stateLim' in ddp_data['active_costs']):
        ddp_data['stateLim_ub'] = [self.ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.ub for i in range(self.ddp.problem.T)]
        ddp_data['stateLim_lb'] = [self.ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.lb for i in range(self.ddp.problem.T)]
        ddp_data['stateLim_ub'].append(self.ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.ub)
        ddp_data['stateLim_lb'].append(self.ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.lb)
    if('ctrlLim' in ddp_data['active_costs']):
        ddp_data['ctrlLim_ub'] = [self.ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub for i in range(self.ddp.problem.T)]
        ddp_data['ctrlLim_lb'] = [self.ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb for i in range(self.ddp.problem.T)]
        ddp_data['ctrlLim_ub'].append(self.ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub)
        ddp_data['ctrlLim_lb'].append(self.ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb)
    if('placement' in ddp_data['active_costs']):
        ddp_data['translation_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.translation for i in range(self.ddp.problem.T)]
        ddp_data['translation_ref'].append(self.ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.translation)
        ddp_data['rotation_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.rotation for i in range(self.ddp.problem.T)]
        ddp_data['rotation_ref'].append(self.ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.rotation)
    if('translation' in ddp_data['active_costs']):
        ddp_data['translation_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(self.ddp.problem.T)]
        ddp_data['translation_ref'].append(self.ddp.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
    if('velocity' in ddp_data['active_costs']):
        ddp_data['velocity_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['velocity'].cost.residual.reference.vector for i in range(self.ddp.problem.T)]
        ddp_data['velocity_ref'].append(self.ddp.problem.terminalModel.differential.costs.costs['velocity'].cost.residual.reference.vector)
        # ddp_data['frame_id'] = self.ddp.problem.runningModels[0].differential.costs.costs['velocity'].cost.residual.id
    if('rotation' in ddp_data['active_costs']):
        ddp_data['rotation_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['rotation'].cost.residual.reference for i in range(self.ddp.problem.T)]
        ddp_data['rotation_ref'].append(self.ddp.problem.terminalModel.differential.costs.costs['rotation'].cost.residual.reference)
    if('force' in ddp_data['active_costs']): 
        ddp_data['force_ref'] = [self.ddp.problem.runningModels[i].differential.costs.costs['force'].cost.residual.reference.vector for i in range(self.ddp.problem.T)]
    return ddp_data





class MPCDataHandlerClassical(MPCDataHandlerAbstract):

  def __init__(self, config, robot):
    super().__init__(config, robot)

  def init_predictions(self):
    '''
    Allocate data for state, control & force predictions
    '''
    self.state_pred     = np.zeros((self.N_plan, self.N_h+1, self.nx)) # Predicted states  ( self.ddp.xs : {x* = (q*, v*)} )
    self.ctrl_pred      = np.zeros((self.N_plan, self.N_h, self.nu))   # Predicted torques ( self.ddp.us : {u*} )
    self.force_pred     = np.zeros((self.N_plan, self.N_h, 6))         # Predicted EE contact forces
    self.state_des_PLAN = np.zeros((self.N_plan+1, self.nx))           # Predicted states at planner frequency  ( x* interpolated at PLAN freq )
    self.ctrl_des_PLAN  = np.zeros((self.N_plan, self.nu))             # Predicted torques at planner frequency ( u* interpolated at PLAN freq )
    self.force_des_PLAN = np.zeros((self.N_plan, 6))                   # Predicted EE contact forces planner frequency  
    self.state_des_CTRL = np.zeros((self.N_ctrl+1, self.nx))           # Reference state at motor drivers freq ( x* interpolated at CTRL freq )
    self.ctrl_des_CTRL  = np.zeros((self.N_ctrl, self.nu))             # Reference input at motor drivers freq ( u* interpolated at CTRL freq )
    self.force_des_CTRL = np.zeros((self.N_ctrl, 6))                   # Reference EE contact force at motor drivers freq
    self.state_des_SIMU = np.zeros((self.N_simu+1, self.nx))           # Reference state at actuation freq ( x* interpolated at SIMU freq )
    self.ctrl_des_SIMU  = np.zeros((self.N_simu, self.nu))             # Reference input at actuation freq ( u* interpolated at SIMU freq )
    self.force_des_SIMU = np.zeros((self.N_simu, 6))                   # Reference EE contact force at actuation freq

  def init_measurements(self, x0):
    '''
    Allocate data for simulation state & force measurements 
    '''
    self.state_mea_SIMU                = np.zeros((self.N_simu+1, self.nx))            # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq )
    self.state_mea_no_noise_SIMU       = np.zeros((self.N_simu+1, self.nx))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq ) without noise
    self.force_mea_SIMU                = np.zeros((self.N_simu, 6)) 
    self.state_mea_SIMU[0, :]          = x0
    self.state_mea_no_noise_SIMU[0, :] = x0


  def init_sim_data(self, x0):
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
    self.init_measurements(x0)

    # DDP solver-specific data
    if(self.RECORD_SOLVER_DATA):
      self.init_solver_data()
   
    logger.info("Initialized MPC simulation data.")

    if(self.INIT_LOG):
      self.print_sim_params(self.init_log_display_time)

    # return sim_data


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
    nq = self.nq ; nv = self.nv ; nx = self.nx ; nu = self.nu
    # Control predictions
    plot_data['u_pred'] = self.ctrl_pred
    plot_data['u_des_PLAN'] = self.ctrl_des_PLAN
    plot_data['u_des_CTRL'] = self.ctrl_des_CTRL
    plot_data['u_des_SIMU'] = self.ctrl_des_SIMU
    # State predictions (at PLAN freq)
    plot_data['q_pred']     = self.state_pred[:,:,:nq]
    plot_data['v_pred']     = self.state_pred[:,:,nq:nq+nv]
    plot_data['q_des_PLAN'] = self.state_des_PLAN[:,:nq]
    plot_data['v_des_PLAN'] = self.state_des_PLAN[:,nq:nq+nv] 
    plot_data['q_des_CTRL'] = self.state_des_CTRL[:,:nq] 
    plot_data['v_des_CTRL'] = self.state_des_CTRL[:,nq:nq+nv]
    plot_data['q_des_SIMU'] = self.state_des_SIMU[:,:nq]
    plot_data['v_des_SIMU'] = self.state_des_SIMU[:,nq:nq+nv]
    # State measurements (at SIMU freq)
    plot_data['q_mea']          = self.state_mea_SIMU[:,:nq]
    plot_data['v_mea']          = self.state_mea_SIMU[:,nq:nq+nv]
    plot_data['q_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,:nq]
    plot_data['v_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,nq:nq+nv]
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

    # # Solver data (optional)
    # if(self.RECORD_SOLVER_DATA']):
    #   # Get SVD & diagonal of Ricatti + record in sim data
    #   plot_data['K_svd'] = np.zeros((self.N_plan'], self.N_h'], nq))
    #   plot_data['Kp_diag'] = np.zeros((self.N_plan'], self.N_h'], nq))
    #   plot_data['Kv_diag'] = np.zeros((self.N_plan'], self.N_h'], nv))
    #   plot_data['Ktau_diag'] = np.zeros((self.N_plan'], self.N_h'], nu))
    #   for i in range(self.N_plan']):
    #     for j in range(self.N_h']):
    #       plot_data['Kp_diag'][i, j, :] = self.K'][i, j, :, :nq].diagonal()
    #       plot_data['Kv_diag'][i, j, :] = self.K'][i, j, :, nq:nq+nv].diagonal()
    #       plot_data['Ktau_diag'][i, j, :] = self.K'][i, j, :, -nu:].diagonal()
    #       _, sv, _ = np.linalg.svd(self.K'][i, j, :, :])
    #       plot_data['K_svd'][i, j, :] = np.sort(sv)[::-1]
    #   # Get diagonal and eigenvals of Vxx + record in sim data
    #   plot_data['Vxx_diag'] = np.zeros((self.N_plan'],self.N_h']+1, nx))
    #   plot_data['Vxx_eig'] = np.zeros((self.N_plan'], self.N_h']+1, nx))
    #   for i in range(self.N_plan']):
    #     for j in range(self.N_h']+1):
    #       plot_data['Vxx_diag'][i, j, :] = self.Vxx'][i, j, :, :].diagonal()
    #       plot_data['Vxx_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Vxx'][i, j, :, :]))[::-1]
    #   # Get diagonal and eigenvals of Quu + record in sim data
    #   plot_data['Quu_diag'] = np.zeros((self.N_plan'],self.N_h'], nu))
    #   plot_data['Quu_eig'] = np.zeros((self.N_plan'], self.N_h'], nu))
    #   for i in range(self.N_plan']):
    #     for j in range(self.N_h']):
    #       plot_data['Quu_diag'][i, j, :] = self.Quu'][i, j, :, :].diagonal()
    #       plot_data['Quu_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Quu'][i, j, :, :]))[::-1]
    #   # Get Jacobian
    #   plot_data['J_rank'] = self.J_rank']
    #   # Get solve regs
    #   plot_data['xreg'] = self.xreg']
    #   plot_data['ureg'] = self.ureg']
    return plot_data
