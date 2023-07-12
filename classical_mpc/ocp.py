
"""
@package force_feedback
@file init_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initializes the OCP + DDP solver
"""

import crocoddyl
import numpy as np
from core_mpc import ocp

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# Check installed pkg
import importlib
FOUND_SOBEC = importlib.util.find_spec("sobec") is not None
if(FOUND_SOBEC):
    import sobec 
else:
    logger.warning('You need to install Sobec !')


class OptimalControlProblemClassical(ocp.OptimalControlProblemAbstract):
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

  def init_running_cost_model(self, state, actuation, runningModel):
    '''
    Populate running cost model
    '''
  # Create and add cost function terms to current IAM
    # State regularization 
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      runningModel.differential.costs.addCost("stateReg", xRegCost, self.stateRegWeight)
    # Control regularization
    if('ctrlReg' in self.WHICH_COSTS):
      uRegCost = self.create_ctrl_reg_cost(state)
      runningModel.differential.costs.addCost("ctrlReg", uRegCost, self.ctrlRegWeight)
    # Control regularization (gravity)
    if('ctrlRegGrav' in self.WHICH_COSTS):
      uRegGravCost = self.create_ctrl_reg_grav_cost(state)
      runningModel.differential.costs.addCost("ctrlRegGrav", uRegGravCost, self.ctrlRegWeight)
    # State limits penalization
    if('stateLim' in self.WHICH_COSTS):
      xLimitCost = self.create_state_limit_cost(state, actuation)
      runningModel.differential.costs.addCost("stateLim", xLimitCost, self.stateLimWeight)
    # Control limits penalization
    if('ctrlLim' in self.WHICH_COSTS):
      uLimitCost = self.create_ctrl_limit_cost(state)
      runningModel.differential.costs.addCost("ctrlLim", uLimitCost, self.ctrlLimWeight)
    # End-effector placement 
    if('placement' in self.WHICH_COSTS):
      framePlacementCost = self.create_frame_placement_cost(state, actuation)
      runningModel.differential.costs.addCost("placement", framePlacementCost, self.framePlacementWeight)
    # End-effector velocity
    if('velocity' in self.WHICH_COSTS): 
      frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
      runningModel.differential.costs.addCost("velocity", frameVelocityCost, self.frameVelocityWeight)
    # Frame translation cost
    if('translation' in self.WHICH_COSTS):
      frameTranslationCost = self.create_frame_translation_cost(state, actuation)
      runningModel.differential.costs.addCost("translation", frameTranslationCost, self.frameTranslationWeight)
    # End-effector orientation 
    if('rotation' in self.WHICH_COSTS):
      frameRotationCost = self.create_frame_rotation_cost(state, actuation)
      runningModel.differential.costs.addCost("rotation", frameRotationCost, self.frameRotationWeight)
    # Frame force cost
    if('force' in self.WHICH_COSTS):
      frameForceCost = self.create_frame_force_cost(state, actuation)
      runningModel.differential.costs.addCost("force", frameForceCost, self.frameForceWeight)
    # Friction cone 
    if('friction' in self.WHICH_COSTS):
      frictionConeCost = self.create_friction_force_cost(state, actuation)
      runningModel.differential.costs.addCost("friction", frictionConeCost, self.frictionConeWeight)
    if('collision' in self.WHICH_COSTS):
      collisionCost = self.create_collision_cost(state, actuation)
      runningModel.differential.costs.addCost("collision", collisionCost, self.collisionCostWeight)

  def init_terminal_cost_model(self, state, actuation, terminalModel):
    ''' 
    Populate terminal cost model
    '''
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
    # End-effector orientation 
    if('collision' in self.WHICH_COSTS):
      collisionCost = self.create_collision_cost(state, actuation)
      terminalModel.differential.costs.addCost("collision", collisionCost, self.collisionCostWeightTerminal*self.dt)

  def initialize(self, x0, callbacks=False, USE_GNMS=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          x0          : initial state of shooting problem
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
          if(FOUND_SOBEC):
            dam = sobec.DifferentialActionModelContactFwdDynamics(state, 
                                                                      actuation, 
                                                                      sobec.ContactModelMultiple(state, actuation.nu), 
                                                                      crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                      inv_damping=0., 
                                                                      enable_force=True)
          else:
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
      
      # Create IAM from DAM
        runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=self.dt))
        
      # Create and add cost function terms to current IAM
        self.init_running_cost_model(state, actuation, runningModels[i])

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
      if(FOUND_SOBEC):
        dam_t = sobec.DifferentialActionModelContactFwdDynamics(state, 
                                                                  actuation, 
                                                                  sobec.ContactModelMultiple(state, actuation.nu), 
                                                                  crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                  inv_damping=0., 
                                                                  enable_force=True)
      else:
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
    terminalModel = crocoddyl.IntegratedActionModelEuler( dam_t, stepTime=0. )

  # Create and add terminal cost models to terminal IAM
    self.init_terminal_cost_model(state, actuation, terminalModel)


  # Add armature
    terminalModel.differential.armature = np.asarray(self.armature)   
  
  # Add contact model
    if(self.nb_contacts > 0):
      for k,contactModel in enumerate(contactModels):
        terminalModel.differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])
    
    logger.info("Created IAMs.")  



  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    ddp = crocoddyl.SolverGNMS(problem)
    # ddp = crocoddyl.SolverFDDP(problem)
  
  # Callbacks
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
  
  # Finish
    logger.info("OCP is ready")
    logger.info("    COSTS   = "+str(self.WHICH_COSTS))
    if(self.nb_contacts > 0):
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
      for ct in self.contacts:
        logger.info("      Found [ "+str(ct['contactModelType'])+" ] (Baumgarte stab. gains = "+str(ct['contactModelGains'])+" , active = "+str(ct['active'])+" )")
    else:
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
    return ddp





class OptimalControlProblemClassicalWithConstraints(ocp.OptimalControlProblemAbstract):
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
    self.check_attribute('WHICH_CONSTRAINTS')

  def initialize(self, x0, callbacks=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          x0          : initial state of shooting problem
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
          if(FOUND_SOBEC):
            dam = sobec.DifferentialActionModelContactFwdDynamics(state, 
                                                                      actuation, 
                                                                      sobec.ContactModelMultiple(state, actuation.nu), 
                                                                      crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                      inv_damping=0., 
                                                                      enable_force=True)
          else:
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
        if('collision' in self.WHICH_COSTS):
          collisionCost = self.create_collision_cost(state, actuation)
          runningModels[i].differential.costs.addCost("collision", collisionCost, self.collisionCostWeight)

      # Armature 
        # Add armature to current IAM
        runningModels[i].differential.armature = np.asarray(self.armature)
      
      # Contact model
        # Add contact model to current IAM
        if(self.nb_contacts > 0):
          for k,contactModel in enumerate(contactModels):
            runningModels[i].differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])

      # Constraint model 
        nc = 0
        constraint_models_stack_list = []
        # State limits
        if('stateBox' in self.WHICH_CONSTRAINTS):
          if(i == 0):
            stateBoxConstraint = self.create_no_constraint(state, 'None', actuation)
          else:
            stateBoxConstraint = self.create_state_constraint(state, 'stateBox', actuation)   
            nc += stateBoxConstraint.nc
          constraint_models_stack_list.append(stateBoxConstraint)
        # Control limits
        if('ctrlBox' in self.WHICH_CONSTRAINTS):
          ctrlBoxConstraint = self.create_ctrl_constraint(state, 'ctrlBox', actuation)
          nc += ctrlBoxConstraint.nc
          constraint_models_stack_list.append(ctrlBoxConstraint)
        # End-effector position limits
        if('translationBox' in self.WHICH_CONSTRAINTS):
          if(i == 0):
            translationBoxConstraint = self.create_no_constraint(state, 'None', actuation)
          else:
            translationBoxConstraint = self.create_translation_constraint(state, 'translationBox', actuation)
            nc += translationBoxConstraint.nc
          constraint_models_stack_list.append(translationBoxConstraint)
        if('forceBox' in self.WHICH_CONSTRAINTS):
          if(i==0):
            forceBoxConstraint = self.create_no_constraint(state, 'None', actuation)
          else:
            forceBoxConstraint = self.create_force_constraint(state, 'forceBox', actuation)
            nc += forceBoxConstraint.nc
          constraint_models_stack_list.append(forceBoxConstraint)
        # No constraints
        if('None' in self.WHICH_CONSTRAINTS):
          noConstraintModel = self.create_no_constraint(state, 'None', actuation)
          constraint_models_stack_list.append(noConstraintModel)

        # Running constraint model stack
        runningConstraintModel = crocoddyl.ConstraintStack(constraint_models_stack_list, state, nc, actuation.nu, 'constraint_'+str(i))



  # Terminal DAM (Contact or FreeFwd)
    # If contact, initialize terminal contact model and create terminal DAMContactDyn
    if(self.nb_contacts > 0):
      contactModels = []
      for ct in self.contacts:
        contactModels.append(self.create_contact_model(ct, state, actuation))

      # Create terminal DAMContactDyn
      if(FOUND_SOBEC):
        dam_t = sobec.DifferentialActionModelContactFwdDynamics(state, 
                                                                  actuation, 
                                                                  sobec.ContactModelMultiple(state, actuation.nu), 
                                                                  crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                  inv_damping=0., 
                                                                  enable_force=True)
      else:
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
    # End-effector orientation 
    if('collision' in self.WHICH_COSTS):
      collisionCost = self.create_collision_cost(state, actuation)
      terminalModel.differential.costs.addCost("collision", collisionCost, self.collisionCostWeightTerminal*self.dt)

  # Add armature
    terminalModel.differential.armature = np.asarray(self.armature)   
  
  # Add contact model
    if(self.nb_contacts > 0):
      for k,contactModel in enumerate(contactModels):
        terminalModel.differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])

  # Constraint model 
    nc = 0
    constraint_models_stack_list_terminal = []
    # State limits
    if('stateBox' in self.WHICH_CONSTRAINTS):
      stateBoxConstraint = self.create_state_constraint(state, 'stateBox', actuation) 
      nc += stateBoxConstraint.nc 
      constraint_models_stack_list_terminal.append(stateBoxConstraint)
    # Control limits
    if('ctrlBox' in self.WHICH_CONSTRAINTS):
      ctrlBoxConstraint = self.create_ctrl_constraint(state, 'ctrlBox', actuation)
      nc += ctrlBoxConstraint.nc
      constraint_models_stack_list_terminal.append(ctrlBoxConstraint)
    # End-effector position limits
    if('translationBox' in self.WHICH_CONSTRAINTS):
      translationBoxConstraint = self.create_translation_constraint(state, 'translationBox', actuation)
      nc += translationBoxConstraint.nc
      constraint_models_stack_list_terminal.append(translationBoxConstraint)
    # No constraint
    if('None' in self.WHICH_CONSTRAINTS):
      noConstraintModel = self.create_no_constraint(state, 'None', actuation)
      constraint_models_stack_list_terminal.append(noConstraintModel)

    # Terminal constraint model stack
    terminalConstraintModel = crocoddyl.ConstraintStack(constraint_models_stack_list_terminal, state, nc, actuation.nu, 'constraint_terminal')
    

    constraintModels = [runningConstraintModel]*(self.N_h) + [terminalConstraintModel] 
    # logger.warning("Constraint models = \n")
    # logger.warning(constraintModels)

    logger.info("Created IAMs.")  



  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    self.check_attribute('USE_PROXQP')
    if(self.USE_PROXQP):
      logger.warning('Using PROXQP solver')
      ddp = crocoddyl.SolverPROXQP(problem, constraintModels) 
    else:
      logger.warning('Using FADMM solver')
      ddp = crocoddyl.SolverFADMM(problem, constraintModels)

  # Callbacks & solver parameters
    self.check_attribute('with_callbacks')
    self.check_attribute('use_filter_ls')
    self.check_attribute('filter_size')
    self.check_attribute('warm_start')
    self.check_attribute('termination_tol')
    self.check_attribute('max_qp_iters')
    self.check_attribute('qp_termination_tol_abs')
    self.check_attribute('qp_termination_tol_rel')
    self.check_attribute('warm_start_y')
    self.check_attribute('reset_rho')
    ddp.with_callbacks = self.with_callbacks
    ddp.use_filter_ls = self.use_filter_ls
    ddp.filter_size = self.filter_size
    ddp.warm_start = self.warm_start
    ddp.termination_tol = self.termination_tol
    ddp.max_qp_iters = self.max_qp_iters
    ddp.eps_abs = self.qp_termination_tol_abs
    ddp.eps_rel = self.qp_termination_tol_rel
    ddp.warm_start_y = self.warm_start_y
    ddp.reset_rho = self.reset_rho
  
  # Finish
    logger.info("OCP is ready")
    logger.info("    COSTS         = "+str(self.WHICH_COSTS))
    logger.info("    CONSTRAINTS   = "+str(self.WHICH_CONSTRAINTS))
    if(self.nb_contacts > 0):
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
      for ct in self.contacts:
        logger.info("      Found [ "+str(ct['contactModelType'])+" ] (Baumgarte stab. gains = "+str(ct['contactModelGains'])+" , active = "+str(ct['active'])+" )")
    else:
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
    return ddp
