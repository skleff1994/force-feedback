"""
@package force_feedback
@file soft_mpc/ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initializes the OCP + DDP solver (visco-elastic contact)
"""

import crocoddyl
import numpy as np
from core_mpc import ocp, pin_utils

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



USE_SOBEC_BINDINGS = True



if(USE_SOBEC_BINDINGS):
  from sobec import DAMSoftContact3DAugmentedFwdDynamics as DAMSoft3DAugmented
  from sobec import IAMSoftContact3DAugmented as IAMSoft3DAugmented
else:
  from soft_mpc.soft_models_3D_augmented import DAMSoftContactDynamics3D as DAMSoft3DAugmented
  from soft_mpc.soft_models_3D_augmented import IAMSoftContactDynamics3D as IAMSoft3DAugmented


class OptimalControlProblemSoftContactAugmented(ocp.OptimalControlProblemAbstract):
  '''
  Helper class for soft contact (augmented) OCP setup with Crocoddyl
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
    self.check_attribute('Kp')
    self.check_attribute('Kv')
    self.check_attribute('oPc_offset')
    self.check_attribute('pinRefFrame')
    self.check_attribute('contactType')

  def initialize(self, y0, softContactModel, callbacks=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
    Soft contact (visco-elastic) augmented formulation, i.e. visco-elastic
    contact force is part of the state . Supported 3D formulation only for now
      INPUT: 
          y0                : initial state of shooting problem
          softContactModel  : SoftContactModel3D (see in utils)
          callbacks         : display Crocoddyl's DDP solver callbacks
      OUTPUT:
        FDDP solver

     A cost term on a variable z(x,u) has the generic form w * a( r( z(x,u) - z0 ) )
     where w <--> cost weight, e.g. 'stateRegWeight' in config file
           r <--> residual model depending on some reference z0, e.g. 'stateRegRef'
                  When ref is set to 'DEFAULT' in YAML file, default references hard-coded here are used
           a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
           z <--> can be state x, control u, frame position or velocity, contact force, etc.
    ''' 
    
  # State and actuation models
    state = crocoddyl.StateMultibody(self.rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    
    
  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
        # Create DAMContactDyn     
        if(softContactModel.nc == 3):
          dam = DAMSoft3DAugmented(state, 
                                  actuation, 
                                  crocoddyl.CostModelSum(state, nu=actuation.nu),
                                  softContactModel.frameId, 
                                  softContactModel.Kp,
                                  softContactModel.Kv,
                                  softContactModel.oPc,
                                  softContactModel.pinRefFrame )
        # elif(softContactModel.nc == 1):
        #   dam = DAMSoft1D(state, 
        #                   actuation, 
        #                   crocoddyl.CostModelSum(state, nu=actuation.nu),
        #                   softContactModel.frameId, 
        #                   softContactModel.contactType,
        #                   softContactModel.Kp,
        #                   softContactModel.Kv,
        #                   softContactModel.oPc,
        #                   softContactModel.pinRefFrame )
        else:
          logger.error("softContactModel.nc = 3 or 1")

      # Create IAM from DAM
        runningModels.append(IAMSoft3DAugmented(dam, self.dt))
        
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
          # not supported yet
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
          if(softContactModel.nc == 3):
            forceRef = np.asarray(self.frameForceRef)[:3]
          else:
            forceRef = np.asarray(self.frameForceRef)[softContactModel.mask]
          runningModels[i].differential.set_force_cost(forceRef, self.frameForceWeight)

      # Armature 
        # Add armature to current IAM
        runningModels[i].differential.armature = np.asarray(self.armature)

  # Terminal DAM (Contact or FreeFwd)
    # Create terminal DAMContactDyn
    if(softContactModel.nc == 3):
      dam_t = DAMSoft3DAugmented(state, 
                                actuation, 
                                crocoddyl.CostModelSum(state, nu=actuation.nu),
                                softContactModel.frameId, 
                                softContactModel.Kp,
                                softContactModel.Kv,
                                softContactModel.oPc,
                                softContactModel.pinRefFrame )
    # elif(softContactModel.nc == 1):
    #   dam_t = DAMSoft1D(state, 
    #                     actuation, 
    #                     crocoddyl.CostModelSum(state, nu=actuation.nu),
    #                     softContactModel.frameId, 
    #                     softContactModel.contactType,
    #                     softContactModel.Kp,
    #                     softContactModel.Kv,
    #                     softContactModel.oPc,
    #                     softContactModel.pinRefFrame )
    else:
      logger.error("softContactModel.nc = 3 or 1")

  # Create terminal IAM from terminal DAM
    terminalModel = IAMSoft3DAugmented( dam_t, 0. )

  # Create and add terminal cost models to terminal IAM
    # State regularization
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      terminalModel.differential.costs.addCost("stateReg", xRegCost, self.stateRegWeightTerminal*self.dt)
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

    logger.info("Created IAMs.")  



  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
  
  # Callbacks
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
  
  # Warm start : initial state + gravity compensation
    ddp.xs = [y0 for i in range(self.N_h+1)]
    fext0 = softContactModel.computeExternalWrench_(self.rmodel, y0[:self.nq], y0[:self.nv])
    ddp.us = [pin_utils.get_tau(y0[:self.nq], y0[:self.nv], np.zeros(self.nv), fext0, self.rmodel, np.zeros(self.nq)) for i in range(self.N_h)] #ddp.problem.quasiStatic(xs_init[:-1])

  # Finish
    logger.info("OCP is ready !")
    logger.info(  "USE_SOBEC_BINDINGS = "+str(USE_SOBEC_BINDINGS))
    logger.info("    COSTS   = "+str(self.WHICH_COSTS))
    logger.info("    SOFT CONTACT MODEL [ oPc="+str(softContactModel.oPc)+\
      " , Kp="+str(softContactModel.Kp)+\
        ', Kv='+str(softContactModel.Kv)+\
        ', pinRefFrame='+str(softContactModel.pinRefFrame)+']')
    
    return ddp