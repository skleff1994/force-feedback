"""
@package force_feedback
@file init_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2022-05-12
@brief Initializes the OCP + DDP solver
"""

import crocoddyl
import numpy as np
from core_mpc import ocp, pin_utils

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



class OptimalControlProblemLPF(ocp.OptimalControlProblemAbstract):
  '''
  Helper class for Low-Pass Filter (LPF) OCP setup with Crocoddyl
   to allow joint torque feedback in the MPC 
  '''
  def __init__(self, robot, config):
    '''
    Override base class constructor if necessary
    '''
    super().__init__(robot, config)
  
  def check_config(self):
    '''
    Override base class checks if necessary
    '''
    super().check_config()
    self.check_attribute('f_c')
    self.check_attribute('LPF_TYPE')
    self.check_attribute('tau_plus_integration')
    self.check_attribute('wRegWeight')
    # self.check_attribute('w_gravity_reg')
    self.check_attribute('wLimWeight')
   
  def initialize(self, y0, callbacks=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file describing task and MPC params
          y0          : initial augmented state of shooting problem (q0, v0, tau0)
          callbacks   : display Crocoddyl's DDP solver callbacks
      OUTPUT:
          FDDP solver

     A cost term on a variable z(x,u) has the generic form w * r( a( z(x,u) ) )
     where w <--> cost weight, e.g. 'stateRegWeight' in config file
           r <--> residual model depending on some reference, e.g. 'stateRegRef'
                  Wen set to 'None' in config file, default references are hard-coded here
           a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
           z <--> can be state x, control u, frame position or velocity, contact force, etc.
    '''
    
  # State and actuation models
    state = crocoddyl.StateMultibody(self.rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
  
  # Contact or not ?
    if(not hasattr(self, 'contacts')):
      self.nb_contacts = 0
    else:
      self.nb_contacts = len(self.contacts)
      self.contact_types = [ct['contactModelType'] for ct in self.contacts]
      logger.debug("Detected "+str(len(self.contacts))+" contacts with types = "+str(self.contact_types))

  # LPF parameters (a.k.a simplified actuation model)
    # Approx. LPF obtained from Z.O.H. discretization on CT LPF 
    if(self.LPF_TYPE==0):
        alpha = np.exp(-2*np.pi*self.f_c*self.dt)
    # Approx. LPF obtained from 1st order Euler int. on CT LPF
    if(self.LPF_TYPE==1):
        alpha = 1./float(1+2*np.pi*self.f_c*self.dt)
    # Exact LPF obtained from E.M.A model (IIR)
    if(self.LPF_TYPE==2):
        y = np.cos(2*np.pi*self.f_c*self.dt)
        alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 
    logger.info("Setup Low-Pass Filter (LPF)")
    logger.info("          self.f_c   = "+str(self.f_c))
    logger.info("          alpha = "+str(alpha))

  # Regularization cost of unfiltered torque (inside IAM_LPF in Crocoddyl)
    if(not hasattr(self, 'wRegRef')):
      logger.error("Need to specify 'wRegRef' in YAML config. Please select in ['zero', 'tau0', 'gravity']")
    elif(self.wRegRef == 'gravity'):
      # If no reference is provided, assume default reg w.r.t. gravity torque
      w_gravity_reg = True
      w_reg_ref = np.zeros(self.nq) # dummy reference not used
      log_msg_w_reg = 'gravity torque'
    else:
      # Otherwise, take the user-provided constant torque reference for w_reg
      w_gravity_reg = False
      if(self.wRegRef== 'zero'):
        w_reg_ref = np.zeros(self.nq)
      elif(self.wRegRef== 'tau0'):
        w_reg_ref = pin_utils.get_u_grav(y0[:self.nq], self.rmodel, self.armature)
      else:
        logger.error("Unknown 'wRegRef' in YAML config. Please select in ['zero', 'tau0', 'gravity']")
      # w_reg_ref = np.asarray(self.wRegRef'])
      log_msg_w_reg = 'constant reference '+self.wRegRef
    logger.debug("Unfiltered torque regularization w.r.t. "+log_msg_w_reg)


  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
      # Create DAM (Contact or FreeFwd)
        # Initialize contact model if necessary and create appropriate DAM
        if(self.nb_contacts > 0):
          contactModels = []
          for ct in self.contacts:
            contactModels.append(self.create_contact_model(ct, state, actuation))

          # Create DAMContactDyn                    
          dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                    actuation, 
                                                                    crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                    crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                    inv_damping=0., 
                                                                    enable_force=True)
        # Otherwise just create DAM
        else:
          # Create DAMFreeDyn
          dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                                 actuation, 
                                                                 crocoddyl.CostModelSum(state, nu=actuation.nu))
      
      # Create IAMLPF from DAM
        runningModels.append(crocoddyl.IntegratedActionModelLPF( dam, 
                                                                stepTime=self.dt, 
                                                                withCostResidual=True, 
                                                                fc=self.f_c, 
                                                                tau_plus_integration=self.tau_plus_integration,
                                                                filter=self.LPF_TYPE,
                                                                is_terminal=False))  
        # Add cost on unfiltered control torque (reg + lim)
        runningModels[i].set_control_reg_cost(self.wRegWeight, w_reg_ref) 
        runningModels[i].set_control_lim_cost(self.wLimWeight) 
      
      # Create and add cost function terms to current IAM
        # State regularization 
        if('stateReg' in self.WHICH_COSTS):
          xRegCost = self.create_state_reg_cost(state, actuation)
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, self.stateRegWeight)
        # Control regularization
        if('ctrlReg' in self.WHICH_COSTS):
          uRegCost = self.create_ctrl_reg_cost(state)
          runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, self.ctrlRegWeight)
        # Control regularization (gravity)
        if('ctrlRegGrav' in self.WHICH_COSTS):
          uRegGravCost = self.create_ctrl_reg_grav_cost(state)
          runningModels[i].differential.costs.addCost("ctrlRegGrav", uRegGravCost, self.ctrlRegWeight)
        # State limits penalization
        if('stateLim' in self.WHICH_COSTS):
          xLimitCost = self.create_state_limit_cost(state, actuation)
          runningModels[i].differential.costs.addCost("stateLim", xLimitCost, self.stateLimWeight)
        # Control limits penalization
        if('ctrlLim' in self.WHICH_COSTS):
          uLimitCost = self.create_ctrl_limit_cost(state)
          runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, self.ctrlLimWeight)
        # End-effector placement 
        if('placement' in self.WHICH_COSTS):
          framePlacementCost = self.create_frame_placement_cost(state, actuation)
          runningModels[i].differential.costs.addCost("placement", framePlacementCost, self.framePlacementWeight)
        # End-effector velocity
        if('velocity' in self.WHICH_COSTS): 
          frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
          runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, self.frameVelocityWeight)
        # Frame translation cost
        if('translation' in self.WHICH_COSTS):
          frameTranslationCost = self.create_frame_translation_cost(state, actuation)
          runningModels[i].differential.costs.addCost("translation", frameTranslationCost, self.frameTranslationWeight)
        # End-effector orientation 
        if('rotation' in self.WHICH_COSTS):
          frameRotationCost = self.create_frame_rotation_cost(state, actuation)
          runningModels[i].differential.costs.addCost("rotation", frameRotationCost, self.frameRotationWeight)
        # Frame force cost
        if('force' in self.WHICH_COSTS):
          frameForceCost = self.create_frame_force_cost(state, actuation)
          runningModels[i].differential.costs.addCost("force", frameForceCost, self.frameForceWeight)
        # Friction cone 
        if('friction' in self.WHICH_COSTS):
          frictionConeCost = self.create_friction_force_cost(state, actuation)
          runningModels[i].differential.costs.addCost("friction", frictionConeCost, self.frictionConeWeight)
      
      # Armature 
        # Add armature to current IAM
        runningModels[i].differential.armature = np.asarray(self.armature)
      
      # Contact model
        # Add contact model to current IAM
        if(self.nb_contacts > 0):
          for k,contactModel in enumerate(contactModels):
            runningModels[i].differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])




  # Terminal DAM (Contact or FreeFwd)
    # If contact, initialize terminal contact model and create terminal DAMContactDyn
    if(self.nb_contacts > 0):
      contactModels = []
      for ct in self.contacts:
        contactModels.append(self.create_contact_model(ct, state, actuation))

      # Create terminal DAMContactDyn
      dam_t = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                actuation, 
                                                                crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                inv_damping=0., 
                                                                enable_force=True)
    # If no contact create DAMFreeDyn
    else:
      dam_t = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                            actuation, 
                                                            crocoddyl.CostModelSum(state, nu=actuation.nu))  
  
  # Create terminal IAM from terminal DAM
    terminalModel = crocoddyl.IntegratedActionModelLPF( dam_t, 
                                                        stepTime=0., 
                                                        withCostResidual=False, 
                                                        fc=self.f_c, 
                                                        # cost_weight_w_reg=self.wRegWeight, 
                                                        # cost_ref_w_reg=w_reg_ref,
                                                        # w_gravity_reg=w_gravity_reg,
                                                        # cost_weight_w_lim=self.wLimWeight,
                                                        tau_plus_integration=self.tau_plus_integration,
                                                        filter=self.LPF_TYPE,
                                                        is_terminal=True)   

  # Create and add terminal cost models to terminal IAM
    # State regularization
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      terminalModel.differential.costs.addCost("stateReg", xRegCost, self.stateRegWeightTerminal*self.dt)
    # Ctrl regularization
    if('ctrlReg' in self.WHICH_COSTS):
      uRegCost = self.create_ctrl_reg_cost(state)
      terminalModel.differential.costs.addCost("ctrlReg", uRegCost, self.ctrlRegWeightTerminal*self.dt)
    # Control limit
    if('ctrlLim' in self.WHICH_COSTS):
      uLimCost = self.create_ctrl_limit_cost(state)
      terminalModel.differential.costs.addCost("ctrlLim", uLimCost, self.ctrlLimWeightTerminal*self.dt)
    # State limits
    if('stateLim' in self.WHICH_COSTS):
      xLimitCost = self.create_state_limit_cost(state, actuation)
      terminalModel.differential.costs.addCost("stateLim", xLimitCost, self.stateLimWeightTerminal*self.dt)
    # EE placement
    if('placement' in self.WHICH_COSTS):
      framePlacementCost = self.create_frame_placement_cost(state, actuation)
      terminalModel.differential.costs.addCost("placement", framePlacementCost, self.framePlacementWeightTerminal*self.dt)
    # EE velocity
    if('velocity' in self.WHICH_COSTS):
      frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, self.frameVelocityWeightTerminal*self.dt)
    # EE translation
    if('translation' in self.WHICH_COSTS):
      frameTranslationCost = self.create_frame_translation_cost(state, actuation)
      terminalModel.differential.costs.addCost("translation", frameTranslationCost, self.frameTranslationWeightTerminal*self.dt)
    # End-effector orientation 
    if('rotation' in self.WHICH_COSTS):
      frameRotationCost = self.create_frame_rotation_cost(state, actuation)
      terminalModel.differential.costs.addCost("rotation", frameRotationCost, self.frameRotationWeightTerminal*self.dt)

  # Add armature
    terminalModel.differential.armature = np.asarray(self.armature)   
  
  # Add contact model
    if(self.nb_contacts > 0):
      for k,contactModel in enumerate(contactModels):
        terminalModel.differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])
    
    logger.info("Created IAMs.")  

  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
    
  # Warm start by default
    ddp.xs = [y0 for i in range(self.N_h+1)]
    ddp.us = [pin_utils.get_u_grav(y0[:self.nq], self.rmodel, self.armature) for i in range(self.N_h)]
  
  # Finish
    logger.info("OCP (LPF) is ready")
    logger.info("    COSTS   = "+str(self.WHICH_COSTS))
    if(self.nb_contacts):
      logger.info("    CONTACT = "+str(self.nb_contacts))
      for ct in self.contacts:
        logger.info("      Found [ "+str(ct['contactModelType'])+" ] (Baumgarte stab. gains = "+str(ct['contactModelGains'])+" , active = "+str(ct['active'])+" )")
    else:
      logger.info("    CONTACT = "+str(self.nb_contacts))
    return ddp