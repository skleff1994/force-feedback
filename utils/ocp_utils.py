"""
@package force_feedback
@file croco_helper.py
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

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Interpolator
def linear_interpolation(data, N):
    '''
    linear interpolation of trajectory with N interpolation knots
     INPUT: 
       data   : input trajectory of type np.array((N_samples, sample_dim))
       N      : number of sub-intervals bewteen 2 consecutive samples
                ( N = 1 ==> no interpolation )
     OUTPUT:
       interp : interpolated trajectory of size N_samples
    '''
    n = data.shape[0] # Number of input samples 
    d = data.shape[1] # Dimension of each input sample
    m = N*(n-1)+1     # Number of output samples (interpolated)
    interp = np.zeros((m, d))
    sample = 0        # Index of input sample 
    for i in range(m):
      coef = float(i % N) / N
      if(i > 0 and coef==0):
        sample+=1
      interp[i] = data[sample]*(1-coef) + data[min(sample+1, n-1)]*coef
    return interp 

def linear_interpolation_demo():
    '''
     Demo of linear interpolation of order N on example data
    '''
    # Generate data if None provided
    data = np.ones((10,2))
    for i in range(data.shape[0]):
        data[i] = i**2
    N = 5
    logger.info("Input data = \n")
    logger.info(data)
    # Interpolate
    logger.info("Interpolate with "+str(N)+" intermediate knots")
    interp = linear_interpolation(data, N)
    # Plot
    import matplotlib.pyplot as plt
    input, = plt.plot(np.linspace(0, 1, data.shape[0]), data[:,1], 'ro', label='input data')
    output, = plt.plot(np.linspace(0, 1, interp.shape[0]), interp[:,1], 'g*', label='interpolated')
    plt.legend(handles=[input, output])
    plt.grid()
    plt.show()



# Cost weights profiles, useful for reaching tasks/cost design
def cost_weight_tanh(i, N, max_weight=1., alpha=1., alpha_cut=0.25):
    '''
    Monotonically increasing weight profile over [0,...,N]
    based on a custom scaled hyperbolic tangent 
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : value of the weight at the end of the window (must be >0)
       alpha      : controls the sharpness of the tanh (alpha high <=> very sharp)
       alpha_cut  : shifts tanh over the time window (i.e. time of inflexion point)
     OUPUT:
       Cost weight at step i : it tarts at weight=0 (when i=0) and
       ends at weight<= max_weight (at i=N). As alpha --> inf, we tend
       toward max_weight
    '''
    return 0.5*max_weight*( np.tanh(alpha*(i/N) -alpha*alpha_cut) + np.tanh(alpha*alpha_cut) )

def cost_weight_tanh_demo():
    '''
     Demo of hyperbolic tangent profile
    '''
    # Generate data if None provided
    N = 200
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    weights_4 = np.zeros(N)
    weights_5 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_tanh(i, N, alpha=4., alpha_cut=0.50)
      weights_2[i] = cost_weight_tanh(i, N, alpha=10., alpha_cut=0.50)
      weights_3[i] = cost_weight_tanh(i, N, alpha=4., alpha_cut=0.75)
      weights_4[i] = cost_weight_tanh(i, N, alpha=10., alpha_cut=0.75)
      weights_5[i] = cost_weight_tanh(i, N, alpha=10., alpha_cut=0.25)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p0, = plt.plot(span, [1.]*N, 'k-.', label='max_weight=1')
    p1, = plt.plot(span, weights_1, 'r-', label='alpha=4, cut=0.50')
    p2, = plt.plot(span, weights_2, 'g-', label='alpha=10, cut=0.50')
    p3, = plt.plot(span, weights_3, 'b-', label='alpha=4, cut=0.75')
    p4, = plt.plot(span, weights_4, 'y-', label='alpha=10, cut=0.75')
    p5, = plt.plot(span, weights_5, 'c-', label='alpha=10, cut=0.25')
    plt.legend(handles=[p1, p2, p3, p4, p5])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()


def cost_weight_linear(i, N, min_weight=0., max_weight=1.):
    '''
    Linear cost weight profile over [0,...,N]
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : value of the weight at the end of the window (must be >=min_weight)
       min_weight : value of the weight at the start of the window (must be >=0)
     OUPUT:
       Cost weight at step i
    '''
    return (max_weight-min_weight)/N * i + min_weight

def cost_weight_linear_demo():
    '''
     Demo of linear profile
    '''
    # Generate data if None provided
    N = 200
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_linear(i, N, min_weight=0., max_weight=1)
      weights_2[i] = cost_weight_linear(i, N, min_weight=2., max_weight=4)
      weights_3[i] = cost_weight_linear(i, N, min_weight=0, max_weight=2)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p1, = plt.plot(span, weights_1, 'r-', label='min_weight=0, max_weight=1')
    p2, = plt.plot(span, weights_2, 'g-', label='min_weight=2, max_weight=4')
    p3, = plt.plot(span, weights_3, 'b-', label='min_weight=0, max_weight=2')
    plt.legend(handles=[p1, p2, p3])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()


def cost_weight_parabolic(i, N, min_weight=0., max_weight=1.):
    '''
    Parabolic cost weight profile over [0,...,N] with min at i=N/2
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       min_weight : minimum weight reached when i=N/2
       max_weight : maximum weight reached at t=0 and i=N
     OUPUT:
       Cost weight at step i
    '''
    return min_weight + 4.*(max_weight-min_weight)/float(N**2) * (i-N/2)**2

def cost_weight_parabolic_demo():
    '''
     Demo of parabolic weight profile
    '''
    # Generate data if None provided
    N = 200
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_parabolic(i, N, min_weight=0., max_weight=4)
      weights_2[i] = cost_weight_parabolic(i, N, min_weight=2., max_weight=4)
      weights_3[i] = cost_weight_parabolic(i, N, min_weight=0, max_weight=2)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p1, = plt.plot(span, weights_1, 'r-', label='min_weight=0, max_weight=4')
    p2, = plt.plot(span, weights_2, 'g-', label='min_weight=2, max_weight=4')
    p3, = plt.plot(span, weights_3, 'b-', label='min_weight=0, max_weight=2')
    plt.legend(handles=[p1, p2, p3])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()


def cost_weight_normal_clamped(i, N, min_weight=0.1, max_weight=1., peak=1., alpha=0.01):
    '''
    Gaussian cost weight profile over [0,...,N] with max at i=N/2
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : maximum weight reached at t=N/2
     OUPUT:
       Cost weight at step i
    '''
    return min(max(peak*np.exp(-alpha*float(i-N/2)**2/2), min_weight), max_weight)

def cost_weight_normal_clamped_demo():
    '''
     Demo of clamped normal (Gaussian) weight profile
    '''
    # Generate data if None provided
    N = 500
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    weights_4 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_normal_clamped(i, N, min_weight=0., max_weight=np.inf, peak=1, alpha=0.0001)
      weights_2[i] = cost_weight_normal_clamped(i, N, min_weight=0., max_weight=np.inf, peak=1, alpha=0.01)
      weights_3[i] = cost_weight_normal_clamped(i, N, min_weight=0.0, max_weight=0.8, peak=1, alpha=0.001)
      weights_4[i] = cost_weight_normal_clamped(i, N, min_weight=0.2, max_weight=1.1, peak=1.2, alpha=0.002)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p1, = plt.plot(span, weights_1, 'r-', label='min_weight=0., max_weight=np.inf, peak=1, alpha=0.0001')
    p2, = plt.plot(span, weights_2, 'g-', label='min_weight=0., max_weight=np.inf, peak=1, alpha=0.01')
    p3, = plt.plot(span, weights_3, 'b-', label='min_weight=0.0, max_weight=0.8, peak=1, alpha=0.001')
    p4, = plt.plot(span, weights_4, 'y-', label='min_weight=0.2, max_weight=1.1, peak=1.2, alpha=0.002')
    plt.legend(handles=[p1, p2, p3, p4])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()



def activation_decreasing_exponential(r, alpha=1., max_weight=1., min_weight=0.5):
    '''
    Activation function of decreasing exponential clamped btw max and min 
     INPUT: 
       r          : residual 
       alpha      : sharpness of the decreasing exponential
       min_weight : minimum weight when r->infty (clamp)
       max_weight : maximum weight when r->0 (clamp)
     OUPUT:
       Cost activation
    '''
    return max(min(np.exp(1/(alpha*r+1))-1, max_weight), min_weight)


# Utils for circle trajectory tracking (position of EE frame) task

def circle_point_LOCAL(t, radius=1., omega=1.):
  '''
  Returns the LOCAL frame coordinates of the point reached at time t
  on a circle trajectory with given radius and angular frequency 
  The circle is drawn in the LOCAL (x,y)-plane of the initial frame of interest
  '''
  # LOCAL coordinates 
  return np.array([radius*(1-np.cos(-omega*t)), radius*np.sin(-omega*t), 0.])


def circle_point_WORLD(t, M_ct, radius=1., omega=1.):
  '''
  Returns the WORLD frame coordinates of the point reached at time t
  on a circle trajectory with given radius and angular frequency 
  M_ct is the initial placement of the frame of interest
  '''
  # LOCAL coordinates 
  return M_ct.act(circle_point_LOCAL(t, radius=radius, omega=omega))





# Setup OCP and solver using Crocoddyl
def init_DDP(robot, config, x0, callbacks=False, 
                                WHICH_COSTS=['all']):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file of OCP params
          x0          : initial state of shooting problem
          callbacks   : display Crocoddyl's DDP solver callbacks
          WHICH_COSTS : which cost terms in the running & terminal cost?
                          'placement', 'velocity', 'stateReg', 'ctrlReg', 'ctrlRegGrav'
                          'stateLim', 'ctrlLim', 'force', 'friction', 'translation'
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
    CONTACT      = False
    CONTACT_TYPE = 'None'
    if('contactModelFrameName' in config.keys()):
      CONTACT = True
      # 6D or 3D ?
      if('contactModelRotationRef' in config.keys()):
        CONTACT_TYPE = '6D'
      else:
        CONTACT_TYPE = '3D'
  
  
  # Create IAMs
    runningModels = []
    for i in range(N_h):  
      # Create DAM (Contact or FreeFwd)
        # Initialize contact model if necessary and create appropriate DAM
        if(CONTACT):
            contactModelGains = np.asarray(config['contactModelGains'])
            contactModelFrameId = robot.model.getFrameId(config['contactModelFrameName'])
            # Default contact reference translation = initial translation
            if(config['contactModelTranslationRef']=='DEFAULT'):
              contactModelTranslationRef = robot.data.oMf[contactModelFrameId].translation.copy()
            else:
              contactModelTranslationRef = config['contactModelTranslationRef']
            
            # 6D Contact model if rotation is specified in config file
            if(CONTACT_TYPE=='6D'):
              # Default rotation = initial rotation of EE frame
              if(config['contactModelRotationRef']=='DEFAULT'):
                contactModelRotationRef = robot.data.oMf[contactModelFrameId].rotation.copy()
              else:
                contactModelRotationRef = config['contactModel6DRotationRef']
              contactModelPlacementRef = pin.SE3(contactModelRotationRef, contactModelTranslationRef)
              contactModel = crocoddyl.ContactModel6D(state, 
                                                      contactModelFrameId, 
                                                      contactModelPlacementRef, 
                                                      contactModelGains) 
            # Otherwise (default) 3D contact model
            elif(CONTACT_TYPE=='3D'):
              contactModel = crocoddyl.ContactModel3D(state, 
                                                      contactModelFrameId, 
                                                      contactModelTranslationRef, 
                                                      contactModelGains)  

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
        if('stateReg' in WHICH_COSTS):
          # Default reference = initial state
          if(config['stateRegRef']=='DEFAULT'):
            stateRegRef = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) #np.zeros(nq+nv) 
          else:
            stateRegRef = np.asarray(config['stateRegRef'])
          stateRegWeights = np.asarray(config['stateRegWeights'])
          xRegCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                                crocoddyl.ResidualModelState(state, stateRegRef, actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['stateRegWeight'])
        # Control regularization
        if('ctrlReg' in WHICH_COSTS):
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
        if('ctrlRegGrav' in WHICH_COSTS):
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
        if('stateLim' in WHICH_COSTS):
          # Default reference = zero state
          if(config['stateLimRef']=='DEFAULT'):
            stateLimRef = np.zeros(nq+nv)
          else:
            stateLimRef = np.asarray(config['stateLimRef'])
          x_max = state.ub 
          x_min = state.lb
          stateLimWeights = np.asarray(config['stateLimWeights'])
          xLimitCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max), stateLimWeights), 
                                                crocoddyl.ResidualModelState(state, stateLimRef, actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeight'])
        # Control limits penalization
        if('ctrlLim' in WHICH_COSTS):
          # Default reference = zero torque
          if(config['ctrlLimRef']=='DEFAULT'):
            ctrlLimRef = np.zeros(nq)
          else:
            ctrlLimRef = np.asarray(config['ctrlLimRef'])
          u_min = -np.asarray(config['ctrlBounds']) 
          u_max = +np.asarray(config['ctrlBounds']) 
          ctrlLimWeights = np.asarray(config['ctrlLimWeights'])
          uLimitCost = crocoddyl.CostModelResidual(state, 
                                                  crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max), ctrlLimWeights), 
                                                  crocoddyl.ResidualModelControl(state, ctrlLimRef))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['ctrlLimWeight'])
        # End-effector placement 
        if('placement' in WHICH_COSTS):
          framePlacementFrameId = robot.model.getFrameId(config['framePlacementFrameName'])
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
        if('velocity' in WHICH_COSTS): 
          frameVelocityFrameId = robot.model.getFrameId(config['frameVelocityFrameName'])
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
                                                                                              pin.WORLD, 
                                                                                              actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
        # Frame translation cost
        if('translation' in WHICH_COSTS):
          frameTranslationFrameId = robot.model.getFrameId(config['frameTranslationFrameName'])
          # Default reference translation = initial translation
          if(config['frameTranslationRef']=='DEFAULT'):
            frameTranslationRef = robot.data.oMf[frameTranslationFrameId].translation.copy()
          else:
            frameTranslationRef = np.asarray(config['frameTranslationRef'])
          frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
          frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                          crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                          crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                                  frameTranslationFrameId, 
                                                                                                  frameTranslationRef, 
                                                                                                  actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeight'])
        # End-effector orientation 
        if('rotation' in WHICH_COSTS):
          frameRotationFrameId = robot.model.getFrameId(config['frameRotationFrameName'])
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
        if('force' in WHICH_COSTS):
          if(not CONTACT):
            logger.error("Force cost but no contact model is defined ! ")
          # 6D contact case
          if(CONTACT_TYPE=='6D'):
            # Default force reference = zero force
            if(config['frameForceRef']=='DEFAULT'):
              frameForceRef = pin.Force( np.zeros(6) )
            else:
              frameForceRef = pin.Force( np.asarray(config['frameForceRef']) )
            frameForceWeights = np.asarray(config['frameForceWeights'])
            frameForceFrameId = robot.model.getFrameId(config['frameForceFrameName'])
            frameForceCost = crocoddyl.CostModelResidual(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                        crocoddyl.ResidualModelContactForce(state, 
                                                                                            frameForceFrameId, 
                                                                                            frameForceRef, 
                                                                                            6, 
                                                                                            actuation.nu))
          # 3D contact case
          elif(CONTACT_TYPE=='3D'):
            # Default force reference = zero force
            if(config['frameForceRef']=='DEFAULT'):
              frameForceRef = pin.Force( np.zeros(6) )
            else:
              frameForceRef = pin.Force( np.asarray(config['frameForceRef']) )
            frameForceWeights = np.asarray(config['frameForceWeights'])[:3]
            frameForceFrameId = robot.model.getFrameId(config['frameForceFrameName'])
            frameForceCost = crocoddyl.CostModelResidual(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                        crocoddyl.ResidualModelContactForce(state, 
                                                                                            frameForceFrameId, 
                                                                                            frameForceRef, 
                                                                                            3, 
                                                                                            actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("force", frameForceCost, config['frameForceWeight'])
        # Friction cone 
        if('friction' in WHICH_COSTS):
          if(not CONTACT):
            logger.error("Friction cost but no contact model is defined !!! ")
          cone_rotation = contactModelPlacementRef.rotation
          # nsurf = cone_rotation.dot(np.matrix(np.array([0, 0, 1])).T)
          mu = config['mu']
          frictionConeFrameId = robot.model.getFrameId(config['frictionConeFrameName'])
          frictionCone = crocoddyl.FrictionCone(cone_rotation, mu, 4, False) #, 0, 1000)
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
          runningModels[i].differential.contacts.addContact("contact", contactModel, active=True)



  # Terminal DAM (Contact or FreeFwd)
    # If contact, initialize terminal contact model and create terminal DAMContactDyn
    if(CONTACT):
      # Initialize terminal contact model
      contactModelGains = np.asarray(config['contactModelGains'])
      contactModelFrameId = robot.model.getFrameId(config['contactModelFrameName'])
      # Default contact reference translation = initial translation
      if(config['contactModelTranslationRef']=='DEFAULT'):
        contactModelTranslationRef = robot.data.oMf[contactModelFrameId].translation.copy()
      else:
        contactModelTranslationRef = config['contactModelTranslationRef']

      # Contact model 6D if rotation is specified in config file
      if(CONTACT_TYPE=='6D'):
        # Default contact reference rotation = initial rotation
        if(config['contactModelRotationRef']=='DEFAULT'):
          contactModelRotationRef = robot.data.oMf[contactModelFrameId].rotation.copy()
        else:
          contactModelRotationRef = config['contactModelRotationRef']
        contactModelPlacementRef = pin.SE3(contactModelRotationRef, contactModelTranslationRef)
        contactModel = crocoddyl.ContactModel6D(state, 
                                                contactModelFrameId, 
                                                contactModelPlacementRef, 
                                                contactModelGains) 

      # Otherwise (default) 3D contact model
      elif(CONTACT_TYPE=='3D'):
        contactModel = crocoddyl.ContactModel3D(state, 
                                                contactModelFrameId, 
                                                contactModelTranslationRef, 
                                                contactModelGains)
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
    if('stateReg' in WHICH_COSTS):
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
    if('stateLim' in WHICH_COSTS):
      # Default reference = zero state
      if(config['stateLimRef']=='DEFAULT'):
        stateLimRef = np.zeros(nq+nv)
      else:
        stateLimRef = np.asarray(config['stateLimRef'])
      x_max = state.ub 
      x_min = state.lb
      stateLimWeights = np.asarray(config['stateLimWeights'])
      xLimitCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max), stateLimWeights), 
                                            crocoddyl.ResidualModelState(state, stateLimRef, actuation.nu))
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeightTerminal']*dt)
    # EE placement
    if('placement' in WHICH_COSTS):
      framePlacementFrameId = robot.model.getFrameId(config['framePlacementFrameName'])
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
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal']*dt)
    # EE velocity
    if('velocity' in WHICH_COSTS):
      frameVelocityFrameId = robot.model.getFrameId(config['frameVelocityFrameName'])
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
                                                                                          pin.WORLD, 
                                                                                          actuation.nu)) 
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal']*dt)
    # EE translation
    if('translation' in WHICH_COSTS):
      frameTranslationFrameId = robot.model.getFrameId(config['frameTranslationFrameName'])
      # Default reference translation = initial translation
      if(config['frameTranslationRef']=='DEFAULT'):
        frameTranslationRef = robot.data.oMf[frameTranslationFrameId].translation.copy()
      else:
        frameTranslationRef = np.asarray(config['frameTranslationRef'])
      frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
      frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                      crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                              frameTranslationFrameId, 
                                                                                              frameTranslationRef, 
                                                                                              actuation.nu)) 
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeightTerminal']*dt)
    # End-effector orientation 
    if('rotation' in WHICH_COSTS):
      frameRotationFrameId = robot.model.getFrameId(config['frameRotationFrameName'])
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
      terminalModel.differential.costs.addCost("rotation", frameRotationCost, config['frameRotationWeightTerminal']*dt)

  # Add armature
    terminalModel.differential.armature = np.asarray(config['armature'])   
  
  # Add contact model
    if(CONTACT):
      terminalModel.differential.contacts.addContact("contact", contactModel, active=True)
    
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
    logger.info("    COSTS   = "+str(WHICH_COSTS))
    logger.info("    CONTACT = "+str(CONTACT)+" ("+str(CONTACT_TYPE)+").")
    return ddp





# Setup OCP and solver using Crocoddyl
def init_DDP_LPF(robot, config, y0, callbacks=False, 
                                    w_reg_ref='gravity',
                                    TAU_PLUS=False,
                                    LPF_TYPE=0,
                                    WHICH_COSTS=['all']):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file describing task and MPC params
          x0          : initial state of shooting problem
          callbacks   : display Crocoddyl's DDP solver callbacks
          w_reg_ref   : reference for reg. cost on unfiltered input w
          TAU_PLUS    : use "TAU_PLUS" integration if True, "TAU" otherwise
          LPF_TYPE    : use expo moving avg (0), classical lpf (1) or exact (2)
          WHICH_COSTS : which cost terms in the running & terminal cost?
                          'placement', 'velocity', 'stateReg', 'ctrlReg', 'ctrlRegGrav'
                          'stateLim', 'ctrlLim', 'translation', 'friction', 'force'
      OUTPUT:
          FDDP solver

     A cost term on a variable z(x,u) has the generic form w * r( a( z(x,u) ) )
     where w <--> cost weight, e.g. 'stateRegWeight' in config file
           r <--> residual model depending on some reference, e.g. 'stateRegRef'
                  Wen set to 'None' in config file, default references are hard-coded here
           a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
           z <--> can be state x, control u, frame position or velocity, contact force, etc.
    '''
    
  # OCP parameters 
    dt = config['dt']                   # OCP integration step (s)               
    N_h = config['N_h']                 # Number of knots in the horizon 
    nq, nv = robot.model.nq, robot.model.nv
    nx = nq+nv
  
  # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
  
  # Contact or not ?
    if('contactModelFrameName' in config.keys()):
      CONTACT = True
    else:
      CONTACT = False

  # LPF parameters (a.k.a simplified actuation model)
    f_c = config['f_c']    
    # Approx. LPF obtained from Z.O.H. discretization on CT LPF 
    if(LPF_TYPE==0):
        alpha = np.exp(-2*np.pi*f_c*dt)
    # Approx. LPF obtained from 1st order Euler int. on CT LPF
    if(LPF_TYPE==1):
        alpha = 1./float(1+2*np.pi*f_c*dt)
    # Exact LPF obtained from E.M.A model (IIR)
    if(LPF_TYPE==2):
        y = np.cos(2*np.pi*f_c*dt)
        alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 
    logger.info("Setup Low-Pass Filter (LPF)")
    logger.info("          f_c   = "+str(f_c))
    logger.info("          alpha = "+str(alpha))

  # Regularization cost of unfiltered torque (inside IAM_LPF in Crocoddyl)
    if(w_reg_ref is None or w_reg_ref == 'gravity'):
      # If no reference is provided, assume default reg w.r.t. gravity torque
      w_gravity_reg = True
      w_reg_ref = np.zeros(nq) # dummy reference not used
      log_msg_w_reg = 'gravity torque'
    else:
      # Otherwise, take the user-provided constant torque reference for w_reg
      w_gravity_reg = False
      log_msg_w_reg = 'constant reference'
    logger.info("Unfiltered torque regularization w.r.t. "+str(log_msg_w_reg)+".")


  # Create IAMs
    runningModels = []
    for i in range(N_h):  

      # Create DAM (Contact or FreeFwd)
        # Initialize contact model if necessary and create appropriate DAM
        if(CONTACT):
            contactModelGains = np.asarray(config['contactModelGains'])
            contactModelFrameId = robot.model.getFrameId(config['contactModelFrameName'])
            # Default contact reference translation = initial translation
            if(config['contactModelTranslationRef']=='DEFAULT'):
              contactModelTranslationRef = robot.data.oMf[contactModelFrameId].translation.copy()
            else:
              contactModelTranslationRef = config['contactModelTranslationRef']
            # Default contact reference rotation = initial rotation
            if(config['contactModelRotationRef']=='DEFAULT'):
              contactModelRotationRef = robot.data.oMf[contactModelFrameId].rotation.copy()
            else:
              contactModelRotationRef = config['contactModelRotationRef']
            contactModelPlacementRef = pin.SE3(contactModelRotationRef, contactModelTranslationRef)
            contact6d = crocoddyl.ContactModel6D(state, 
                                                contactModelFrameId, 
                                                contactModelPlacementRef, 
                                                contactModelGains) 
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
                                                                stepTime=dt, 
                                                                withCostResidual=True, 
                                                                fc=f_c, 
                                                                cost_weight_w_reg=config['wRegWeight'], 
                                                                cost_ref_w_reg=w_reg_ref,
                                                                w_gravity_reg=w_gravity_reg,
                                                                cost_weight_w_lim=config['wLimWeight'],
                                                                tau_plus_integration=TAU_PLUS,
                                                                filter=LPF_TYPE,
                                                                is_terminal=False))      
      
      # Create and add cost function terms to current IAM
        # State regularization 
        if('stateReg' in WHICH_COSTS):
          # Default reference = initial state
          if(config['stateRegRef']=='DEFAULT'):
            stateRegRef = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) #np.zeros(nq+nv) 
          else:
            stateRegRef = np.asarray(config['stateRegRef'])
          stateRegWeights = np.asarray(config['stateRegWeights'])
          xRegCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                                crocoddyl.ResidualModelState(state, stateRegRef, actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['stateRegWeight'])
        # Control regularization
        if('ctrlReg' in WHICH_COSTS):
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
        if('ctrlRegGrav' in WHICH_COSTS):
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
        if('stateLim' in WHICH_COSTS):
          # Default reference = zero state
          if(config['stateLimRef']=='DEFAULT'):
            stateLimRef = np.zeros(nq+nv)
          else:
            stateLimRef = np.asarray(config['stateLimRef'])
          x_max = state.ub 
          x_min = state.lb
          stateLimWeights = np.asarray(config['stateLimWeights'])
          xLimitCost = crocoddyl.CostModelResidual(state, 
                                                crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max), stateLimWeights), 
                                                crocoddyl.ResidualModelState(state, stateLimRef, actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeight'])
        # Control limits penalization
        if('ctrlLim' in WHICH_COSTS):
          # Default reference = zero torque
          if(config['ctrlLimRef']=='DEFAULT'):
            ctrlLimRef = np.zeros(nq)
          else:
            ctrlLimRef = np.asarray(config['ctrlLimRef'])
          u_min = -np.asarray(config['ctrlBounds']) 
          u_max = +np.asarray(config['ctrlBounds']) 
          ctrlLimWeights = np.asarray(config['ctrlLimWeights'])
          uLimitCost = crocoddyl.CostModelResidual(state, 
                                                  crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max), ctrlLimWeights), 
                                                  crocoddyl.ResidualModelControl(state, ctrlLimRef))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['ctrlLimWeight'])
        # End-effector placement 
        if('placement' in WHICH_COSTS):
          framePlacementFrameId = robot.model.getFrameId(config['framePlacementFrameName'])
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
        if('velocity' in WHICH_COSTS): 
          frameVelocityFrameId = robot.model.getFrameId(config['frameVelocityFrameName'])
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
                                                                                              pin.WORLD, 
                                                                                              actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
        # Frame translation cost
        if('translation' in WHICH_COSTS):
          frameTranslationFrameId = robot.model.getFrameId(config['frameTranslationFrameName'])
          # Default reference translation = initial translation
          if(config['frameTranslationRef']=='DEFAULT'):
            frameTranslationRef = robot.data.oMf[frameTranslationFrameId].translation.copy()
          else:
            frameTranslationRef = np.asarray(config['frameTranslationRef'])
          frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
          frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                          crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                          crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                                  frameTranslationFrameId, 
                                                                                                  frameTranslationRef, 
                                                                                                  actuation.nu)) 
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeight'])
        # Frame force cost
        if('force' in WHICH_COSTS):
          if(not CONTACT):
            logger.error("Force cost but no contact model is defined ! ")
          # Default force reference = zero force
          if(config['frameForceRef']=='DEFAULT'):
            frameForceRef = pin.Force( np.zeros(6) )
          else:
            frameForceRef = pin.Force( np.asarray(config['frameForceRef']) )
          frameForceWeights = np.asarray(config['frameForceWeights'])
          frameForceFrameId = robot.model.getFrameId(config['frameForceFrameName'])
          frameForceCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                      crocoddyl.ResidualModelContactForce(state, 
                                                                                          frameForceFrameId, 
                                                                                          frameForceRef, 
                                                                                          6, 
                                                                                          actuation.nu))
          # Add cost term to IAM
          runningModels[i].differential.costs.addCost("force", frameForceCost, config['frameForceWeight'])
        # Friction cone 
        if('friction' in WHICH_COSTS):
          if(not CONTACT):
            logger.error("Friction cost but no contact model is defined !!! ")
          cone_rotation = contactModelPlacementRef.rotation
          # nsurf = cone_rotation.dot(np.matrix(np.array([0, 0, 1])).T)
          mu = config['mu']
          frictionConeFrameId = robot.model.getFrameId(config['frictionConeFrameName'])
          frictionCone = crocoddyl.FrictionCone(cone_rotation, mu, 4, False) #, 0, 1000)
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
          runningModels[i].differential.contacts.addContact("contact", contact6d, active=True)



  # Terminal DAM (Contact or FreeFwd)
    # If contact, initialize terminal contact model and create terminal DAMContactDyn
    if(CONTACT):
      # Initialize terminal contact model
      contactModelGains = np.asarray(config['contactModelGains'])
      contactModelFrameId = robot.model.getFrameId(config['contactModelFrameName'])
      # Default contact reference translation = initial translation
      if(config['contactModelTranslationRef']=='DEFAULT'):
        contactModelTranslationRef = robot.data.oMf[contactModelFrameId].translation.copy()
      else:
        contactModelTranslationRef = config['contactModelTranslationRef']
      # Default contact reference rotation = initial rotation
      if(config['contactModelRotationRef']=='DEFAULT'):
        contactModelRotationRef = robot.data.oMf[contactModelFrameId].rotation.copy()
      else:
        contactModelRotationRef = config['contactModelRotationRef']
      contactModelPlacementRef = pin.SE3(contactModelRotationRef, contactModelTranslationRef)
      contact6d = crocoddyl.ContactModel6D(state, 
                                          contactModelFrameId, 
                                          contactModelPlacementRef, 
                                          contactModelGains) 
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
                                                        withCostResidual=True, 
                                                        fc=f_c, 
                                                        cost_weight_w_reg=config['wRegWeight'], 
                                                        cost_ref_w_reg=w_reg_ref,
                                                        w_gravity_reg=w_gravity_reg,
                                                        cost_weight_w_lim=config['wLimWeight'],
                                                        tau_plus_integration=TAU_PLUS,
                                                        filter=LPF_TYPE,
                                                        is_terminal=True)   

  # Create and add terminal cost models to terminal IAM
    # State regularization
    if('stateReg' in WHICH_COSTS):
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
    if('stateLim' in WHICH_COSTS):
      # Default reference = zero state
      if(config['stateLimRef']=='DEFAULT'):
        stateLimRef = np.zeros(nq+nv)
      else:
        stateLimRef = np.asarray(config['stateLimRef'])
      x_max = state.ub 
      x_min = state.lb
      stateLimWeights = np.asarray(config['stateLimWeights'])
      xLimitCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max), stateLimWeights), 
                                            crocoddyl.ResidualModelState(state, stateLimRef, actuation.nu))
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeightTerminal']*dt)
    # EE placement
    if('placement' in WHICH_COSTS):
      framePlacementFrameId = robot.model.getFrameId(config['framePlacementFrameName'])
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
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal']*dt)
    # EE velocity
    if('velocity' in WHICH_COSTS):
      frameVelocityFrameId = robot.model.getFrameId(config['frameVelocityFrameName'])
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
                                                                                          pin.WORLD, 
                                                                                          actuation.nu)) 
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal']*dt)
    # EE translation
    if('translation' in WHICH_COSTS):
      frameTranslationFrameId = robot.model.getFrameId(config['frameTranslationFrameName'])
      # Default reference translation = initial translation
      if(config['frameTranslationRef']=='DEFAULT'):
        frameTranslationRef = robot.data.oMf[frameTranslationFrameId].translation.copy()
      else:
        frameTranslationRef = np.asarray(config['frameTranslationRef'])
      frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
      frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                      crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                              frameTranslationFrameId, 
                                                                                              frameTranslationRef, 
                                                                                              actuation.nu)) 
      # Add cost term to terminal IAM
      terminalModel.differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeightTerminal']*dt)
  
  # Add armature
    terminalModel.differential.armature = np.asarray(config['armature'])   
  
  # Add contact model
    if(CONTACT):
      terminalModel.differential.contacts.addContact("contact", contact6d, active=True)
    
    logger.info("Created IAMs.")  



  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)
  
  # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
    
  # Warm start yb default
    ddp.xs = [y0 for i in range(N_h+1)]
    ddp.us = [pin_utils.get_u_grav(y0[:nq], robot.model) for i in range(N_h)]
  
  # Final 
    logger.info("OCP is ready : COSTS = "+str(WHICH_COSTS)+', CONTACT = '+str(CONTACT)+'.')
    return ddp