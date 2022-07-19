"""
@package force_feedback
@file init_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initializes the OCP + DDP solver (visco-elastic contact)
"""

import crocoddyl
import sobec
import numpy as np
from core_mpc import ocp

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from soft_mpc.dam3d import DAMSoftContactDynamics 


class OptimalControlProblemSoftContact(ocp.OptimalControlProblemAbstract):
  '''
  Helper class for classical OCP setup with Crocoddyl
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

  def initialize(self, x0, frameId, Kp, Kv, oPc, pinRefFrame, callbacks=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          x0          : initial state of shooting problem
          frameId     : frame at which the soft contact is defined
          Kp, Kv      : stiffness and damping coefficient of the visco-elastic contact model
          oPc         : anchor point of the contact 
          pinRefFrame : reference frame in which the contact model is expressed (pin.LOCAL or pin.LWA)
          callbacks   : display Crocoddyl's DDP solver callbacks
      OUTPUT:
        FDDP solver

     A cost term on a variable z(x,u) has the generic form w * a( r( z(x,u) - z0 ) )
     where w <--> cost weight, e.g. 'stateRegWeight' in config file
           r <--> residual model depending on some reference z0, e.g. 'stateRegRef'
                  When ref is set to 'DEFAULT' in YAML file, default references hard-coded here are used
           a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
           z <--> can be state x, control u, frame position or velocity, contact force, etc.
    ''' 
    self.nb_contacts = 0
    
  # State and actuation models
    state = crocoddyl.StateMultibody(self.rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    

  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
        # Create DAMContactDyn                    
        dam = DAMSoftContactDynamics(state, 
                                    actuation, 
                                    crocoddyl.CostModelSum(state, nu=actuation.nu),
                                    frameId, 
                                    Kp=Kp,
                                    Kv=Kv,
                                    oPc=oPc,
                                    pinRefFrame=pinRefFrame )

      # Create IAM from DAM
        runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=self.dt))
        
      # Create and add cost function terms to current IAM
        # State regularization 
        if('stateReg' in self.WHICH_COSTS):
          xRegCost = self.create_state_reg_cost(state, actuation)
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, self.stateRegWeight)
        # Control regularization
        if('ctrlReg' in self.WHICH_COSTS):
          uRegCost = self.create_ctrl_reg_cost(state)
          runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, self.ctrlRegWeight)
        # # Control regularization (gravity)
        # if('ctrlRegGrav' in self.WHICH_COSTS):
        #   uRegGravCost = self.create_ctrl_reg_grav_cost(state)
        #   runningModels[i].differential.costs.addCost("ctrlRegGrav", uRegGravCost, self.ctrlRegWeight)
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
          runningModels[i].differential.set_force_cost(np.asarray(self.frameForceRef)[:3], self.frameForceWeight)
        #   frameForceCost = self.create_frame_force_cost(state, actuation)
        #   runningModels[i].differential.costs.addCost("force", frameForceCost, self.frameForceWeight)
        # Friction cone 
        if('friction' in self.WHICH_COSTS):
          frictionConeCost = self.create_friction_force_cost(state, actuation)
          runningModels[i].differential.costs.addCost("friction", frictionConeCost, self.frictionConeWeight)
      
      # Armature 
        # Add armature to current IAM
        runningModels[i].differential.armature = np.asarray(self.armature)

  # Terminal DAM (Contact or FreeFwd)
    # Create terminal DAMContactDyn
    dam_t = DAMSoftContactDynamics(state, 
                                    actuation, 
                                    crocoddyl.CostModelSum(state, nu=actuation.nu),
                                    frameId, 
                                    Kp=Kp,
                                    Kv=Kv,
                                    oPc=oPc,
                                    pinRefFrame=pinRefFrame )

  # Create terminal IAM from terminal DAM
    terminalModel = crocoddyl.IntegratedActionModelEuler( dam_t, stepTime=0. )

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
    # Frame force cost
    if('force' in self.WHICH_COSTS):
        runningModels[i].differential.set_force_cost(np.asarray(self.frameForceRef)[:3], self.frameForceWeight)
        
  # Add armature
    terminalModel.differential.armature = np.asarray(self.armature)   

    logger.info("Created IAMs.")  



  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
  
  # Callbacks
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
  
  # Finish
    logger.info("OCP is ready")
    logger.info("    COSTS   = "+str(self.WHICH_COSTS))
    logger.info("    SOFT CONTACT MODEL [ oPc="+str(oPc)+" , Kp="+str(Kp)+', Kv='+str(Kv)+', pinRefFrame='+str(pinRefFrame)+']')
    
    return ddp