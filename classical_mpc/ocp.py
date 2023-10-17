
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
    logger.error('You need to install Sobec !')

FOUND_MIM_SOLVERS = importlib.util.find_spec("mim_solvers") is not None
if(FOUND_MIM_SOLVERS):
    import mim_solvers 
else:
    logger.error('You need to install mim_solvers !')


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

  def parse_contacts(self):
    '''
    Parses the YAML dict of contacts and count them
    '''
    if(not hasattr(self, 'contacts')):
      self.nb_contacts = 0
    else:
      self.nb_contacts = len(self.contacts)
      self.contact_types = [ct['contactModelType'] for ct in self.contacts]
      logger.debug("Detected "+str(len(self.contacts))+" contacts with types = "+str(self.contact_types))
      
  def create_differential_action_model(self, state, actuation):
    '''
    Initialize a differential action model with or without contacts
    '''
    # If there are contacts, defined constrained DAM
    contactModels = []
    if(self.nb_contacts > 0):
      for ct in self.contacts:
        contactModels.append(self.create_contact_model(ct, state, actuation))   
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
    # Otherwise just create free DAM
    else:
      dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                              actuation, 
                                                              crocoddyl.CostModelSum(state, nu=actuation.nu))
    return dam, contactModels
  
  def init_running_model(self, state, actuation, runningModel, contactModels):
    '''
  Populate running model with costs and contacts
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
      runningModel.differential.costs.addCost("ctrlRegGrav", uRegGravCost, self.ctrlRegGravWeight)
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

    # Armature 
    runningModel.differential.armature = np.asarray(self.armature)
    
    # Contact model
    if(len(contactModels) > 0):
      for k,contactModel in enumerate(contactModels):
        runningModel.differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])

  def init_terminal_model(self, state, actuation, terminalModel, contactModels):
    ''' 
    Populate terminal model with costs and contacts 
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

    # Add armature
    terminalModel.differential.armature = np.asarray(self.armature)   
  
    # Add contact model
    if(len(contactModels)):
      for k,contactModel in enumerate(contactModels):
        terminalModel.differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])

  def success_log(self):
    '''
    Log of successful OCP initialization + important information
    '''
    logger.info("OCP is ready !")
    logger.info("    COSTS         = "+str(self.WHICH_COSTS))
    if(self.nb_contacts > 0):
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
      for ct in self.contacts:
        logger.info("      Found [ "+str(ct['contactModelType'])+" ] (Baumgarte stab. gains = "+str(ct['contactModelGains'])+" , active = "+str(ct['active'])+" )")
    else:
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
      
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
    self.parse_contacts()

  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
      # Create DAM (Contact or FreeFwd), IAM Euler and initialize costs+contacts
        dam, contactModels = self.create_differential_action_model(state, actuation) 
        runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=self.dt))
        self.init_running_model(state, actuation, runningModels[i], contactModels)

    # Terminal model
    dam_t, contactModels = self.create_differential_action_model(state, actuation)  
    terminalModel = crocoddyl.IntegratedActionModelEuler( dam_t, stepTime=0. )
    self.init_terminal_model(state, actuation, terminalModel, contactModels)
    
    logger.info("Created IAMs.")  



  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    if(self.SOLVER == 'sqp'):
      logger.warning("Using the SQP solver.")
      ddp = mim_solvers.SolverSQP(problem)
    elif(self.SOLVER == 'fddp'):
      logger.warning("Using the FDDP solver.")
      ddp = mim_solvers.SolverFDDP(problem)
        
  # Callbacks
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
  
  # Finish
    self.success_log()
    return ddp





class OptimalControlProblemClassicalWithConstraints(OptimalControlProblemClassical):
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
    self.check_attribute('use_filter_ls')
    self.check_attribute('filter_size')
    self.check_attribute('warm_start')
    self.check_attribute('termination_tol')
    self.check_attribute('max_qp_iters')
    self.check_attribute('qp_termination_tol_abs')
    self.check_attribute('qp_termination_tol_rel')
    self.check_attribute('warm_start_y')
    self.check_attribute('reset_rho')

  def create_differential_action_model(self, state, actuation):
    '''
    Initialize a differential action model with or without contacts, 
    and with explicit constraints
    '''
    # If there are contacts, defined constrained DAM
    contactModels = []
    if(self.nb_contacts > 0):
      for ct in self.contacts:
        contactModels.append(self.create_contact_model(ct, state, actuation))   
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                  actuation, 
                                                                  crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                  crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                  crocoddyl.ConstraintModelManager(state, actuation.nu),
                                                                  inv_damping=0., 
                                                                  enable_force=True)
    # Otherwise just create free DAM
    else:
      dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                              actuation, 
                                                              crocoddyl.CostModelSum(state, nu=actuation.nu),
                                                              crocoddyl.ConstraintModelManager(state, actuation.nu))
    return dam, contactModels
  
  def init_running_model(self, state, actuation, runningModel, node_id):
    '''
    Populate running model with costs, contacts and constraints
    '''
    # Add costs and contacts 
    super().init_running_model(self, state, actuation, runningModel)

    # Add constraints
    # State limits
    if('stateBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      stateBoxConstraint = self.create_state_constraint(state, actuation)   
      runningModel.differential.constraints.addConstraint('stateBox', stateBoxConstraint)
    # Control limits
    if('ctrlBox' in self.WHICH_CONSTRAINTS):
      ctrlBoxConstraint = self.create_ctrl_constraint(state, actuation)
      runningModel.differential.constraints.addConstraint('ctrlBox', ctrlBoxConstraint)
    # End-effector position limits
    if('translationBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      translationBoxConstraint = self.create_translation_constraint(state, actuation)
      runningModel.differential.constraints.addConstraint('translationBox', translationBoxConstraint)
    # Contact force 
    if('forceBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      forceBoxConstraint = self.create_force_constraint(state, actuation)
      runningModel.differential.constraints.addConstraint('forceBox', forceBoxConstraint)
      
  def init_terminal_model(self, state, actuation, terminalModel, node_id):
    ''' 
    Populate terminal model with costs, contacts and constraints
    '''
    # Add costs and contacts 
    super().init_terminal_model(self, state, actuation, terminalModel)

    # Add constraints
    # State limits
    if('stateBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      stateBoxConstraint = self.create_state_constraint(state, actuation)   
      terminalModel.differential.constraints.addConstraint('stateBox', stateBoxConstraint)
    # Control limits
    if('ctrlBox' in self.WHICH_CONSTRAINTS):
      ctrlBoxConstraint = self.create_ctrl_constraint(state, actuation)
      terminalModel.differential.constraints.addConstraint('ctrlBox', ctrlBoxConstraint)
    # End-effector position limits
    if('translationBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      translationBoxConstraint = self.create_translation_constraint(state, actuation)
      terminalModel.differential.constraints.addConstraint('translationBox', translationBoxConstraint)
    # Contact force 
    if('forceBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      forceBoxConstraint = self.create_force_constraint(state, actuation)
      terminalModel.differential.constraints.addConstraint('forceBox', forceBoxConstraint)

  def success_log(self):
    '''
    Log of successful OCP initialization + important information
    '''
    super().success_log()
    logger.info("    CONSTRAINTS   = "+str(self.WHICH_CONSTRAINTS))
       
  def initialize(self, x0, callbacks=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          x0          : initial state of shooting problem
          callbacks   : display Crocoddyl's DDP solver callbacks
      OUTPUT:
          mim_solver.SolverXXX object (derived from crocoddyl.SolverAbstract)

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
    self.parse_contacts()

  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
      # Create DAM (Contact or FreeFwd), IAM Euler and initialize costs+contacts+constraints
        dam, contactModels = self.create_differential_action_model(state, actuation) 
        runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=self.dt))
        self.init_running_model(state, actuation, runningModels[i], contactModels, i)

    # Terminal model
    dam_t, contactModels = self.create_differential_action_model(state, actuation)  
    terminalModel = crocoddyl.IntegratedActionModelEuler( dam_t, stepTime=0. )
    self.init_terminal_model(state, actuation, terminalModel, contactModels, i)
    
    logger.info("Created IAMs.")  


  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    if(self.SOLVER == 'proxqp'):
      logger.warning("Using the ProxQP solver.")
      ddp = mim_solvers.SolverProxQP(problem)
    elif(self.SOLVER == 'cssqp'):
      logger.warning("Using the CSSQP solver.")
      ddp = mim_solvers.SolverCSQP(problem)
      
  # Callbacks & solver parameters
    ddp.with_callbacks = callbacks
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
    self.success_log()
    
    return ddp



    
