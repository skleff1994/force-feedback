
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
import pinocchio as pin
from utils import pin_utils

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



# Setup OCP and solver using Crocoddyl
def init_DDP(robot, config, x0, callbacks=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file of OCP params
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
  # OCP parameters
    dt = config['dt']                   
    N_h = config['N_h']               
    nq, nv = robot.model.nq, robot.model.nv
  
  # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
  
  # Contact or not ?
    if('contacts' not in config.keys()):
      CONTACT = False
    else:
      cts = config['contacts']
      CONTACT = True
      CONTACT_TYPES = [ct['contactModelType'] for ct in cts]
      logger.debug("Detected "+str(len(cts))+" contacts with types = "+str(CONTACT_TYPES))

  # Create IAMs
    runningModels = []
    for i in range(N_h):  
      # Create DAM (Contact or FreeFwd)
        # Initialize contact model if necessary and create appropriate DAM
        if(CONTACT):
          contactModels = []
          for ct in cts:
            contactModels.append(create_contact_model(ct, robot, state, actuation))

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
      
      # Create IAM from DAM
        runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=dt))
        
      # Create and add cost function terms to current IAM
        # State regularization 
        if('stateReg' in config['WHICH_COSTS']):
          # Default reference = initial state
          if(config['stateRegRef']=='DEFAULT'):
            stateRegRef = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) #np.zeros(nq+nv) 
            # logger.debug("stateRegRef = "+str(stateRegRef))
          else:
            stateRegRef = np.asarray(config['stateRegRef'])
          stateRegWeights = np.asarray(config['stateRegWeights'])
          xRegCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                                crocoddyl.ResidualModelState(state, stateRegRef, actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['stateRegWeight'])
        # Control regularization
        if('ctrlReg' in config['WHICH_COSTS']):
          # Default reference = zero torque 
          if(config['ctrlRegRef']=='DEFAULT'):
            u_reg_ref = np.zeros(nq)
          else:
            u_reg_ref = np.asarray(config['ctrlRegRef'])
          residual = crocoddyl.ResidualModelControl(state, u_reg_ref)
          ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
          uRegCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                                residual)
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['ctrlRegWeight'])
        # Control regularization (gravity)
        if('ctrlRegGrav' in config['WHICH_COSTS']):
          # Contact or not?
          if(CONTACT):
            residual = crocoddyl.ResidualModelContactControlGrav(state)
          else:
            residual = crocoddyl.ResidualModelControlGrav(state)
          ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
          uRegGravCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                                residual)
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("ctrlRegGrav", uRegGravCost, config['ctrlRegWeight'])
        # State limits penalization
        if('stateLim' in config['WHICH_COSTS']):
          # Default reference = zero state
          stateLimRef = np.zeros(nq+nv)
          x_max = config['coef_xlim']*state.ub 
          x_min = config['coef_xlim']*state.lb
          stateLimWeights = np.asarray(config['stateLimWeights'])
          xLimitCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max), stateLimWeights), 
                                                crocoddyl.ResidualModelState(state, stateLimRef, actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeight'])
        # Control limits penalization
        if('ctrlLim' in config['WHICH_COSTS']):
          # Default reference = zero torque
          ctrlLimRef = np.zeros(nq)
          u_min = -config['coef_ulim']*state.pinocchio.effortLimit #np.asarray(config['ctrlBounds']) 
          u_max = +config['coef_ulim']*state.pinocchio.effortLimit #np.asarray(config['ctrlBounds']) 
          ctrlLimWeights = np.asarray(config['ctrlLimWeights'])
          uLimitCost = crocoddyl.CostModelResidual(state, 
                                                  crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max), ctrlLimWeights), 
                                                  crocoddyl.ResidualModelControl(state, ctrlLimRef))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['ctrlLimWeight'])
        # End-effector placement 
        if('placement' in config['WHICH_COSTS']):
          framePlacementFrameName = config['framePlacementFrameName']
          framePlacementFrameId = robot.model.getFrameId(framePlacementFrameName)
          # Default translation reference = initial translation
          if(config['framePlacementTranslationRef']=='DEFAULT'):
            framePlacementTranslationRef = robot.data.oMf[framePlacementFrameId].translation.copy()
          else:
            framePlacementTranslationRef = np.asarray(config['framePlacementTranslationRef'])
          # Default rotation reference = initial rotation
          if(config['framePlacementRotationRef']=='DEFAULT'):
            framePlacementRotationRef = robot.data.oMf[framePlacementFrameId].rotation.copy()
          else:
            framePlacementRotationRef = np.asarray(config['framePlacementRotationRef'])
          framePlacementRef = pin.SE3(framePlacementRotationRef, framePlacementTranslationRef)
          framePlacementWeights = np.asarray(config['framePlacementWeights'])
          framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                          crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                          crocoddyl.ResidualModelFramePlacement(state, 
                                                                                                framePlacementFrameId, 
                                                                                                framePlacementRef, 
                                                                                                actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeight'])
        # End-effector velocity
        if('velocity' in config['WHICH_COSTS']): 
          frameVelocityFrameName = config['frameVelocityFrameName']
          frameVelocityFrameId = robot.model.getFrameId(frameVelocityFrameName)
          # Default reference = zero velocity
          if(config['frameVelocityRef']=='DEFAULT'):
            frameVelocityRef = pin.Motion( np.zeros(6) )
          else:
            frameVelocityRef = pin.Motion( np.asarray( config['frameVelocityRef'] ) )
          frameVelocityWeights = np.asarray(config['frameVelocityWeights'])
          frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                          crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                          crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                              frameVelocityFrameId, 
                                                                                              frameVelocityRef, 
                                                                                              pin.LOCAL, 
                                                                                              actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
        # Frame translation cost
        if('translation' in config['WHICH_COSTS']):
          frameTranslationFrameName = config['frameTranslationFrameName']
          frameTranslationFrameId = robot.model.getFrameId(frameTranslationFrameName)
          # Default reference translation = initial translation
          if(config['frameTranslationRef']=='DEFAULT'):
            frameTranslationRef = robot.data.oMf[frameTranslationFrameId].translation.copy()
          else:
            frameTranslationRef = np.asarray(config['frameTranslationRef'])
          if('frameTranslationWeights' in config):
            frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
            frameTranslationActivation = crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2)
          elif('alpha_quadflatlog' in config):
            alpha_quadflatlog = config['alpha_quadflatlog']
            frameTranslationActivation = crocoddyl.ActivationModelQuadFlatLog(3, alpha_quadflatlog)
          else:
            logger.error("Please specify either 'alpha_quadflatlog' or 'frameTranslationWeights' in config file")
          frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                          frameTranslationActivation, 
                                                          crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                                  frameTranslationFrameId, 
                                                                                                  frameTranslationRef, 
                                                                                                  actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeight'])
        # End-effector orientation 
        if('rotation' in config['WHICH_COSTS']):
          frameRotationFrameName = config['frameRotationFrameName']
          frameRotationFrameId = robot.model.getFrameId(frameRotationFrameName)
          # Default rotation reference = initial rotation
          if(config['frameRotationRef']=='DEFAULT'):
            frameRotationRef = robot.data.oMf[frameRotationFrameId].rotation.copy()
          else:
            frameRotationRef   = np.asarray(config['frameRotationRef'])
          frameRotationWeights = np.asarray(config['frameRotationWeights'])
          frameRotationCost    = crocoddyl.CostModelResidual(state, 
                                                             crocoddyl.ActivationModelWeightedQuad(frameRotationWeights**2), 
                                                             crocoddyl.ResidualModelFrameRotation(state, 
                                                                                                  frameRotationFrameId, 
                                                                                                  frameRotationRef, 
                                                                                                  actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("rotation", frameRotationCost, config['frameRotationWeight'])
        # Frame force cost
        if('force' in config['WHICH_COSTS']):
          if(not CONTACT):
            logger.error("Force cost but no contact model is defined ! ")
          frameForceFrameName = config['frameForceFrameName']
          frameForceFrameId = robot.model.getFrameId(frameForceFrameName) 
          found_ct_force_frame = False
          for ct in cts:
            if(frameForceFrameName==ct['contactModelFrameName']):
              found_ct_force_frame = True
              ct_force_frame_type  = ct['contactModelType']
          if(not found_ct_force_frame):
            logger.error("Could not find force cost frame name in contact frame names. Make sure that the frame name of the force cost matches one of the contact frame names.")
          # 6D contact case : wrench = linear in (x,y,z) + angular in (Ox,Oy,Oz)
          if(ct_force_frame_type=='6D'):
            # Default force reference = zero force
            frameForceRef = pin.Force( np.asarray(config['frameForceRef']) )
            frameForceWeights = np.asarray(config['frameForceWeights']) 
            frameForceCost = crocoddyl.CostModelResidual(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                        crocoddyl.ResidualModelContactForce(state, 
                                                                                            frameForceFrameId, 
                                                                                            frameForceRef, 
                                                                                            6, 
                                                                                            actuation.nu))
          # 3D contact case : linear force in (x,y,z) (LOCAL)
          if(ct_force_frame_type=='3D'):
            # Default force reference = zero force
            frameForceRef = pin.Force( np.asarray(config['frameForceRef']) )
            frameForceWeights = np.asarray(config['frameForceWeights'])[:3]
            frameForceCost = crocoddyl.CostModelResidual(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                        crocoddyl.ResidualModelContactForce(state, 
                                                                                            frameForceFrameId, 
                                                                                            frameForceRef, 
                                                                                            3, 
                                                                                            actuation.nu))
          # 1D contact case : linear force along z (LOCAL)
          if('1D' in ct_force_frame_type):
            if('x' in ct_force_frame_type): constrainedAxis = crocoddyl.x
            if('y' in ct_force_frame_type): constrainedAxis = crocoddyl.y
            if('z' in ct_force_frame_type): constrainedAxis = crocoddyl.z
            # Default force reference = zero force
            frameForceRef = pin.Force( np.asarray(config['frameForceRef']) )
            frameForceWeights = np.asarray(config['frameForceWeights'])[constrainedAxis:constrainedAxis+1]
            frameForceCost = crocoddyl.CostModelResidual(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                        crocoddyl.ResidualModelContactForce(state, 
                                                                                            frameForceFrameId, 
                                                                                            frameForceRef, 
                                                                                            1, 
                                                                                            actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("force", frameForceCost, config['frameForceWeight'])
        # Friction cone 
        if('friction' in config['WHICH_COSTS']):
          if(not CONTACT):
            logger.error("Friction cost but no contact model is defined !!! ")
          # nsurf = cone_rotation.dot(np.matrix(np.array([0, 0, 1])).T)
          mu = config['mu']
          frictionConeFrameName = config['frictionConeFrameName']
          frictionConeFrameId = robot.model.getFrameId(frictionConeFrameName)  
          # axis_
          cone_placement = robot.data.oMf[frictionConeFrameId].copy()
          # Rotate 180° around x+ to make z become -z
          normal = cone_placement.rotation.T.dot(np.array([0.,0.,1.]))
          # cone_rotation = cone_placement.rotation.dot(pin.utils.rpyToMatrix(+np.pi, 0., 0.))
          # cone_rotation = robot.data.oMf[frictionConeFrameId].rotation.copy() #contactModelPlacementRef.rotation
          frictionCone = crocoddyl.FrictionCone(normal, mu, 4, False) #, 0, 1000)
          frictionConeCost = crocoddyl.CostModelResidual(state,
                                                        crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(frictionCone.lb , frictionCone.ub)),
                                                        crocoddyl.ResidualModelContactFrictionCone(state, frictionConeFrameId, frictionCone, actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("friction", frictionConeCost, config['frictionConeWeight'])
      
      # Armature 
        # Add armature to current IAM
        runningModels[i].differential.armature = np.asarray(config['armature'])
      
      # Contact model
        # Add contact model to current IAM
        if(CONTACT):
          for k,contactModel in enumerate(contactModels):
            runningModels[i].differential.contacts.addContact(cts[k]['contactModelFrameName'], contactModel, active=cts[k]['active'])



  # Terminal DAM (Contact or FreeFwd)
    # If contact, initialize terminal contact model and create terminal DAMContactDyn
    if(CONTACT):
      contactModels = []
      for ct in cts:
        contactModels.append(create_contact_model(ct, robot, state, actuation))

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
    terminalModel = crocoddyl.IntegratedActionModelEuler( dam_t, stepTime=0. )

  # Create and add terminal cost models to terminal IAM
    # State regularization
    if('stateReg' in config['WHICH_COSTS']):
      # Default reference = initial state
      if(config['stateRegRef']=='DEFAULT'):
        stateRegRef = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) 
      else:
        stateRegRef = np.asarray(config['stateRegRef'])
      stateRegWeights = np.asarray(config['stateRegWeights'])
      xRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                            crocoddyl.ResidualModelState(state, stateRegRef, actuation.nu))
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("stateReg", xRegCost, config['stateRegWeightTerminal']*dt)
    # State limits
    if('stateLim' in config['WHICH_COSTS']):
      # Default reference = zero state
      stateLimRef = np.zeros(nq+nv)
      x_max = config['coef_xlim']*state.ub 
      x_min = config['coef_xlim']*state.lb
      stateLimWeights = np.asarray(config['stateLimWeights'])
      xLimitCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max), stateLimWeights), 
                                            crocoddyl.ResidualModelState(state, stateLimRef, actuation.nu))
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeightTerminal']*dt)
    # EE placement
    if('placement' in config['WHICH_COSTS']):
      framePlacementFrameName = config['framePlacementFrameName']
      framePlacementFrameId = robot.model.getFrameId(framePlacementFrameName)      
      if(config['framePlacementTranslationRef']=='DEFAULT'):
        framePlacementTranslationRef = robot.data.oMf[framePlacementFrameId].translation.copy()
      else:
        framePlacementTranslationRef = np.asarray(config['framePlacementTranslationRef'])
      # Default rotation reference = initial rotation
      if(config['framePlacementRotationRef']=='DEFAULT'):
        framePlacementRotationRef = robot.data.oMf[framePlacementFrameId].rotation.copy()
      else:
        framePlacementRotationRef = np.asarray(config['framePlacementRotationRef'])
      framePlacementRef = pin.SE3(framePlacementRotationRef, framePlacementTranslationRef)
      framePlacementWeights = np.asarray(config['framePlacementWeights'])
      framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                      crocoddyl.ResidualModelFramePlacement(state, 
                                                                                            framePlacementFrameId, 
                                                                                            framePlacementRef, 
                                                                                            actuation.nu)) 
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal']*dt)
    # EE velocity
    if('velocity' in config['WHICH_COSTS']):
      frameVelocityFrameName = config['frameVelocityFrameName']
      frameVelocityFrameId = robot.model.getFrameId(frameVelocityFrameName)      
      # Default reference = zero velocity
      if(config['frameVelocityRef']=='DEFAULT'):
        frameVelocityRef = pin.Motion( np.zeros(6) )
      else:
        frameVelocityRef = pin.Motion( np.asarray( config['frameVelocityRef'] ) )
      frameVelocityWeights = np.asarray(config['frameVelocityWeights'])
      frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                      crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                          frameVelocityFrameId, 
                                                                                          frameVelocityRef, 
                                                                                          pin.LOCAL, 
                                                                                          actuation.nu)) 
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal']*dt)
    # EE translation
    if('translation' in config['WHICH_COSTS']):
      frameTranslationFrameName = config['frameTranslationFrameName']
      frameTranslationFrameId = robot.model.getFrameId(frameTranslationFrameName)      
      if(config['frameTranslationRef']=='DEFAULT'):
        frameTranslationRef = robot.data.oMf[frameTranslationFrameId].translation.copy()
      else:
        frameTranslationRef = np.asarray(config['frameTranslationRef'])
      if('frameTranslationWeights' in config):
        frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
        frameTranslationActivation = crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2)
      elif('alpha_quadflatlog' in config):
        alpha_quadflatlog = config['alpha_quadflatlog']
        frameTranslationActivation = crocoddyl.ActivationModelQuadFlatLog(3, alpha_quadflatlog)
      else:
        logger.error("Please specify either 'alpha_quadflatlog' or 'frameTranslationWeights' in config file")
      frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                      frameTranslationActivation, 
                                                      crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                              frameTranslationFrameId, 
                                                                                              frameTranslationRef, 
                                                                                              actuation.nu)) 
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeightTerminal']*dt)
    # End-effector orientation 
    if('rotation' in config['WHICH_COSTS']):
      frameRotationFrameName = config['frameRotationFrameName']
      frameRotationFrameId = robot.model.getFrameId(frameRotationFrameName)      
      if(config['frameRotationRef']=='DEFAULT'):
        frameRotationRef = robot.data.oMf[frameRotationFrameId].rotation.copy()
      else:
        frameRotationRef   = np.asarray(config['frameRotationRef'])
      frameRotationWeights = np.asarray(config['frameRotationWeights'])
      frameRotationCost    = crocoddyl.CostModelResidual(state, 
                                                          crocoddyl.ActivationModelWeightedQuad(frameRotationWeights**2), 
                                                          crocoddyl.ResidualModelFrameRotation(state, 
                                                                                              frameRotationFrameId, 
                                                                                              frameRotationRef, 
                                                                                              actuation.nu)) 
      terminalModel.differential.costs.addCost("rotation", frameRotationCost, config['frameRotationWeightTerminal']*dt)

  # Add armature
    terminalModel.differential.armature = np.asarray(config['armature'])   
  
  # Add contact model
    if(CONTACT):
      for k,contactModel in enumerate(contactModels):
        terminalModel.differential.contacts.addContact(cts[k]['contactModelFrameName'], contactModel, active=cts[k]['active'])
    
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
    logger.info("    COSTS   = "+str(config['WHICH_COSTS']))
    if(CONTACT):
      logger.info("    CONTACT = "+str(CONTACT))
      for ct in cts:
        logger.info("      Found [ "+str(ct['contactModelType'])+" ] (Baumgarte stab. gains = "+str(ct['contactModelGains'])+" , active = "+str(ct['active'])+" )")
    else:
      logger.info("    CONTACT = "+str(CONTACT))
    return ddp



