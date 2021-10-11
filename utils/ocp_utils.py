"""
@package force_feedback
@file croco_helper.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initializes the OCP + DDP solver
"""

from os import POSIX_FADV_NOREUSE
import crocoddyl
import numpy as np
import pinocchio as pin
from utils import pin_utils

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
    print("Input data = \n")
    print(data)
    # Interpolate
    print("Interpolate with "+str(N)+" intermediate knots")
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

# N=1000
# r = np.linspace(1,2, N+1)
# a = np.zeros(N+1)
# for i in range(1001):
#     a[i] = activation_decreasing_exponential(r[i], alpha=0.01, max_weight=10., min_weight=1.)
# import matplotlib.pyplot as plt
# plt.plot(r, a)
# plt.grid()
# plt.show()




# Setup OCP and solver using Crocoddyl
def init_DDP(robot, config, x0, callbacks=False, 
                                WHICH_COSTS=['all'], 
                                CONTACT=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      - Running cost: EE placement (Mref) + x_reg (xref) + u_reg (uref)
      - Terminal cost: EE placement (Mref) + EE velocity (0) + x_reg (xref)
      Mref = initial frame placement read in config
      xref = initial state read in config
      uref = initial gravity compensation torque (from xref)
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file describing task and MPC params
          x0          : initial state of shooting problem
          callbacks   : display Crocoddyl's DDP solver callbacks
          WHICH_COSTS : which cost terms in the running & terminal cost?
                          'placement', 'velocity', 'stateReg', 'ctrlReg'
                          'stateLim', 'ctrlLim', 'force'
      OUTPUT:
        FDDP solver
    '''
    # OCP parameters
    dt = config['dt']                   # OCP integration step (s)    
    N_h = config['N_h']                 # Number of knots in the horizon 
    # Model params
    id_endeff = robot.model.getFrameId('contact')
    M_ee = robot.data.oMf[id_endeff]
    nq, nv = robot.model.nq, robot.model.nv
    # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
    # Contact or not?
    if(CONTACT):
      baumgarte_gains = np.array([0., 0.])
      contact_placement = robot.data.oMf[id_endeff]
      contact_placement.translation = contact_placement.act(np.array([0., 0., 0.03]))
      contact6d = crocoddyl.ContactModel6D(state, id_endeff, contact_placement, baumgarte_gains) 
    
    
    # Construct cost function terms
    # State regularization
    if('all' in WHICH_COSTS or 'stateReg' in WHICH_COSTS):
      stateRegWeights = np.asarray(config['stateRegWeights'])
      x_reg_ref = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) #np.zeros(nq+nv)     
      xRegCost = crocoddyl.CostModelResidual(state, 
                                            # crocoddyl.ActivationModelSmooth2Norm(nr=14,eps=.001),
                                            # crocoddyl.ActivationModelQuadFlatExp(nr=14,alpha=10),
                                            crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                            crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
      # print("[OCP] Added state reg cost.")  
    # Control regularization
    if('all' in WHICH_COSTS or 'ctrlReg' in WHICH_COSTS):
      if(CONTACT):
        residual = crocoddyl.ResidualModelContactControlGrav(state)
      else:
        residual = crocoddyl.ResidualModelControlGrav(state)
      ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
      uRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                            residual)
      # print("[OCP] Added ctrl reg cost.")
    # State limits penalization
    if('all' in WHICH_COSTS or 'stateLim' in WHICH_COSTS):
      x_lim_ref  = np.zeros(nq+nv)
      x_max = state.ub 
      x_min = state.lb
      stateLimWeights = np.asarray(config['stateLimWeights'])
      xLimitCost = crocoddyl.CostModelResidual(state, 
                                            # crocoddyl.ActivationModelSmooth1Norm(nr=14,eps=10.),
                                            crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max),stateLimWeights), 
                                            crocoddyl.ResidualModelState(state, x_lim_ref, actuation.nu))
      # print("[OCP] Added state lim cost.")
    # Control limits penalization
    if('all' in WHICH_COSTS or 'ctrlLim' in WHICH_COSTS):
      u_min = -np.asarray(config['ctrl_lim']) 
      u_max = +np.asarray(config['ctrl_lim']) 
      u_lim_ref = np.zeros(nq)
      uLimitCost = crocoddyl.CostModelResidual(state, 
                                              crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                              crocoddyl.ResidualModelControl(state, u_lim_ref))
      # print("[OCP] Added ctrl lim cost.")
    # End-effector placement 
    if('all' in WHICH_COSTS or 'placement' in WHICH_COSTS):
      if(config['p_des']=='None'):
        p_des = M_ee.translation.copy()
      else:
        p_des = np.asarray(config['p_des'])
      desiredFramePlacement = pin.SE3(M_ee.rotation, p_des)
      framePlacementWeights = np.asarray(config['framePlacementWeights'])
      framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                      crocoddyl.ResidualModelFramePlacement(state, 
                                                                                            id_endeff, 
                                                                                            desiredFramePlacement, 
                                                                                            actuation.nu)) 
      # print("[OCP] Added frame placement cost.")
    # End-effector velocity
    if('all' in WHICH_COSTS or 'velocity' in WHICH_COSTS): 
      desiredFrameMotion = pin.Motion(np.concatenate([np.asarray(config['v_des']), np.zeros(3)]))
      frameVelocityWeights = np.asarray(config['frameVelocityWeights'])
      frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                      crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                          id_endeff, 
                                                                                          desiredFrameMotion, 
                                                                                          pin.LOCAL, 
                                                                                          actuation.nu)) 
      # print("[OCP] Added frame velocity cost.")
    # Frame translation cost
    if('all' in WHICH_COSTS or 'translation' in WHICH_COSTS):
      if(config['p_des']=='None'):
        desiredFrameTranslation = M_ee.translation.copy()
      else:
        desiredFrameTranslation = np.asarray(config['p_des'])
      frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
      frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                      crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                              id_endeff, 
                                                                                              desiredFrameTranslation, 
                                                                                              actuation.nu)) 
    # Frame force cost
    if('all' in WHICH_COSTS or 'force' in WHICH_COSTS):
      if(not CONTACT):
        print("[OCP] !! ERROR !! ")
        print("[OCP]  >>> No contact model is defined !")
      desiredFrameForce = pin.Force(np.asarray(config['f_des']))
      frameForceWeights = np.asarray(config['frameForceWeights'])
      frameForceCost = crocoddyl.CostModelResidual(state, 
                                                   crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                   crocoddyl.ResidualModelContactForce(state, id_endeff, desiredFrameForce, 6, actuation.nu))
    
    # Create IAMs
    runningModels = []
    for i in range(N_h):
        # Create DAM (Contact or FreeFwd)
        if(CONTACT):
          dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                    actuation, 
                                                                    crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                    crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                    inv_damping=0., 
                                                                    enable_force=True)
        else:
          dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                                 actuation, 
                                                                 crocoddyl.CostModelSum(state, nu=actuation.nu))
        # Create IAM
        runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=dt))
        
        # Add cost models
        if('all' in WHICH_COSTS or 'placement' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeight'])
        if('all' in WHICH_COSTS or 'translation' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeight'])
        if('all' in WHICH_COSTS or 'velocity' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
        if('all' in WHICH_COSTS or 'stateReg' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['stateRegWeight'])
        if('all' in WHICH_COSTS or 'ctrlReg' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['ctrlRegWeight'])
        if('all' in WHICH_COSTS or 'stateLim' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeight'])
        if('all' in WHICH_COSTS or 'ctrlLim' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['ctrlLimWeight'])
        if('all' in WHICH_COSTS or 'force' in WHICH_COSTS):
          runningModels[i].differential.costs.addCost("force", frameForceCost, config['frameForceWeight'])
        
        # Add armature
        runningModels[i].differential.armature = np.asarray(config['armature'])
      
        # Contact model
        if(CONTACT):
          runningModels[i].differential.contacts.addContact("contact", contact6d, active=True)
        
    # Terminal DAM (Contact or FreeFwd)
    if(CONTACT):
      dam_t = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                actuation, 
                                                                crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                inv_damping=0., 
                                                                enable_force=True)
    else:
      dam_t = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                            actuation, 
                                                            crocoddyl.CostModelSum(state, nu=actuation.nu))    
    
    # Terminal IAM
    terminalModel = crocoddyl.IntegratedActionModelEuler( dam_t, stepTime=0. )
    
    # Add cost models
    if('all' in WHICH_COSTS or 'placement' in WHICH_COSTS):
      terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
    if('all' in WHICH_COSTS or 'velocity' in WHICH_COSTS):
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])
    if('all' in WHICH_COSTS or 'translation' in WHICH_COSTS):
      terminalModel.differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeightTerminal'])
    if('all' in WHICH_COSTS or 'stateReg' in WHICH_COSTS):
      terminalModel.differential.costs.addCost("stateReg", xRegCost, config['stateRegWeightTerminal'])
    if('all' in WHICH_COSTS or 'stateLim' in WHICH_COSTS):
      terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeightTerminal'])

    # Add armature
    terminalModel.differential.armature = np.asarray(config['armature']) 
    
    # Add contact model
    if(CONTACT):
      terminalModel.differential.contacts.addContact("contact", contact6d, active=True)
    
    print("[OCP] Created IAMs.")  
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)

    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
    
    print("[OCP] OCP is ready ! (CONTACT="+str(CONTACT)+")")
    print("[OCP]   Costs = "+str(WHICH_COSTS))
    return ddp





# Setup OCP and solver using Crocoddyl
def init_DDP_LPF(robot, config, y0, callbacks=False, 
                                    cost_w_reg=0.1,
                                    cost_w_lim=1., 
                                    tau_plus=True,
                                    lpf_type=0,
                                    WHICH_COSTS=['all'],
                                    CONTACT=False):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file describing task and MPC params
          x0          : initial state of shooting problem
          callbacks   : display Crocoddyl's DDP solver callbacks
          cost_w_reg  : cost weight on reg. of unfiltered input w around u_grav
          cost_w_lim  : cost weight on limit of unfiltered input w 
          tau_plus    : use "tau_plus" integration if True, "tau" otherwise
          lpf_type    : use expo moving avg (0), classical lpf (1) or exact (2)
          WHICH_COSTS : which cost terms in the running & terminal cost?
                          'placement', 'velocity', 'stateReg', 'ctrlReg'
                          'stateLim', 'ctrlLim', 'translation'
      OUTPUT:
          FDDP solver
    '''
    
    # OCP parameters 
    dt = config['dt']                   # OCP integration step (s)               
    N_h = config['N_h']                 # Number of knots in the horizon 
    # Model params
    id_endeff = robot.model.getFrameId('contact')
    M_ee = robot.data.oMf[id_endeff]
    nq, nv = robot.model.nq, robot.model.nv
    nx = nq+nv
    # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
    # Contact model 
    if(CONTACT):
      baumgarte_gains = np.array([0., 0.])
      contact6d = crocoddyl.ContactModel6D(state, id_endeff, robot.data.oMf[id_endeff], baumgarte_gains) 
    

    # Cost function terms
    # State regularization
    if('all' in WHICH_COSTS or 'stateReg' in WHICH_COSTS):
      stateRegWeights = np.asarray(config['stateRegWeights'])
      x_reg_ref = y0[:nx]   
      xRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                            crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
      print('[OCP] Added state regularization cost.')
    # Control regularization
    if('all' in WHICH_COSTS or 'ctrlReg' in WHICH_COSTS):
      if(CONTACT):
        residual = crocoddyl.ResidualModelContactControlGrav(state)
      else:
        residual = crocoddyl.ResidualModelControlGrav(state)
      ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
      uRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                            residual)
      print('[OCP] Added control regularization cost.')
    # State limits penalization
    if('all' in WHICH_COSTS or 'stateLim' in WHICH_COSTS):
      x_lim_ref  = np.zeros(nq+nv)
      xLimitCost = crocoddyl.CostModelResidual(state, 
                                              crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(state.lb, state.ub)), 
                                              crocoddyl.ResidualModelState(state, x_lim_ref, actuation.nu))
      print('[OCP] Added state limit cost.')
    # Control limits penalization
    if('all' in WHICH_COSTS or 'ctrlLim' in WHICH_COSTS):
      u_min = -np.asarray(config['u_lim']) 
      u_max = +np.asarray(config['u_lim']) 
      u_lim_ref = np.zeros(nq)
      uLimitCost = crocoddyl.CostModelResidual(state, 
                                              crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                              crocoddyl.ResidualModelControl(state, u_lim_ref))
      print('[OCP] Added control limit cost.')
    # End-effector placement 
    if('all' in WHICH_COSTS or 'placement' in WHICH_COSTS):
      # p_des = np.asarray(config['p_des']) 
      desiredFramePlacement = M_ee.copy() #pin.SE3(M_ee.rotation.T, p_des)
      framePlacementWeights = np.asarray(config['framePlacementWeights'])
      framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                      crocoddyl.ResidualModelFramePlacement(state, 
                                                                                            id_endeff, 
                                                                                            desiredFramePlacement, 
                                                                                            actuation.nu)) 
      print('[OCP] Added frame placement cost.')
    # End-effector translation
    if('all' in WHICH_COSTS or 'translation' in WHICH_COSTS):
      if(config['p_des']=='None'):
        desiredFrameTranslation = M_ee.translation.copy()
      else:
        desiredFrameTranslation = np.asarray(config['p_des'])
      frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
      frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                      crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                              id_endeff, 
                                                                                              desiredFrameTranslation, 
                                                                                              actuation.nu)) 
      # print("[OCP] Desired frame translation = ", desiredFrameTranslation)
      print("[OCP] Added frame translation cost.")
    # End-effector velocity 
    if('all' in WHICH_COSTS or 'velocity' in WHICH_COSTS):
      desiredFrameMotion = pin.Motion(np.concatenate([np.asarray(config['v_des']), np.zeros(3)]))
      frameVelocityWeights = np.asarray(config['frameVelocityWeights'])
      frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                      crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                          id_endeff, 
                                                                                          desiredFrameMotion, 
                                                                                          pin.LOCAL, 
                                                                                          actuation.nu)) 
      print('[OCP] Added frame velocity cost.')
    # Frame force cost
    if('all' in WHICH_COSTS or 'force' in WHICH_COSTS):
      if(not CONTACT):
        print("[OCP] !! frameForceCost WARNING !! \n")
        print("[OCP]  >>> No contact model is defined")
      desiredFrameForce = pin.Force(np.asarray(config['f_des']))
      frameForceWeights = np.asarray(config['frameForceWeights'])
      frameForceCost = crocoddyl.CostModelResidual(state, 
                                                   crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                   crocoddyl.ResidualModelContactForce(state, id_endeff, desiredFrameForce, 6, actuation.nu))
    

    # LPF parameters (a.k.a simplified actuation model)
    f_c = config['f_c']    
    # Approx. LPF obtained from Z.O.H. discretization on CT LPF 
    if(lpf_type==0):
        alpha = np.exp(-2*np.pi*f_c*dt)
    # Approx. LPF obtained from 1st order Euler int. on CT LPF
    if(lpf_type==1):
        alpha = 1./float(1+2*np.pi*f_c*dt)
    # Exact LPF obtained from E.M.A model (IIR)
    if(lpf_type==2):
        y = np.cos(2*np.pi*f_c*dt)
        alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 
    print("[OCP] LOW-PASS FILTER : ")
    print("          f_c   = ", f_c)
    print("          alpha = ", alpha)
    

    # Create IAMs
    runningModels = []
    for i in range(N_h):
      costs = crocoddyl.CostModelSum(state, nu=actuation.nu)
      if('all' in WHICH_COSTS or 'placement' in WHICH_COSTS):
        costs.addCost("placement", framePlacementCost, config['framePlacementWeight'])
      if('all' in WHICH_COSTS or 'translation' in WHICH_COSTS):
        costs.addCost("translation", frameTranslationCost, config['frameTranslationWeight'])
      if('all' in WHICH_COSTS or 'velocity' in WHICH_COSTS):
        costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
      if('all' in WHICH_COSTS or 'stateReg' in WHICH_COSTS):
        costs.addCost("stateReg", xRegCost, config['stateRegWeight'])
      if('all' in WHICH_COSTS or 'ctrlReg' in WHICH_COSTS):
        costs.addCost("ctrlReg", uRegCost, config['ctrlRegWeight']) 
      if('all' in WHICH_COSTS or 'stateLim' in WHICH_COSTS):
        costs.addCost("stateLim", xLimitCost, config['stateLimWeight'])
      if('all' in WHICH_COSTS or 'ctrlLim' in WHICH_COSTS):
        costs.addCost("ctrlLim", uLimitCost, config['ctrlLimWeight'])
      if('all' in WHICH_COSTS or 'force' in WHICH_COSTS):
        costs.addCost("force", frameForceCost, config['frameForceWeight'])

      # Create DAM (Contact or FreeFwd)
      if(CONTACT):
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                  actuation, 
                                                                  crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                  costs, 
                                                                  inv_damping=0., 
                                                                  enable_force=True)
      else:
        dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, costs)
      
      # IAM LPF
      runningModels.append(crocoddyl.IntegratedActionModelLPF( dam, 
                                                              stepTime=dt, 
                                                              withCostResidual=True, 
                                                              fc=f_c, 
                                                              cost_weight_w_reg=cost_w_reg, 
                                                              cost_weight_w_lim=cost_w_lim,
                                                              tau_plus_integration=tau_plus,
                                                              filter=lpf_type,
                                                              is_terminal=False))

      # Add armature
      runningModels[i].differential.armature = np.asarray(config['armature'])
      
      # Contact model 
      if(CONTACT):
        runningModels[i].differential.contacts.addContact("contact", contact6d, active=True)

    # Terminal cost function 
    terminal_costs = crocoddyl.CostModelSum(state, nu=actuation.nu)
    if('all' in WHICH_COSTS or 'placement' in WHICH_COSTS):
      terminal_costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
    if('all' in WHICH_COSTS or 'translation' in WHICH_COSTS):
      terminal_costs.addCost("translation", frameTranslationCost, config['frameTranslationWeightTerminal'])
    if('all' in WHICH_COSTS or 'velocity' in WHICH_COSTS):
      terminal_costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])
    if('all' in WHICH_COSTS or 'stateReg' in WHICH_COSTS):
      terminal_costs.addCost("stateReg", xRegCost, config['stateRegWeightTerminal'])
    if('all' in WHICH_COSTS or 'ctrlReg' in WHICH_COSTS):
      terminal_costs.addCost("ctrlReg", uRegCost, config['ctrlRegWeightTerminal'])
    if('all' in WHICH_COSTS or 'stateLim' in WHICH_COSTS):
      terminal_costs.addCost("stateLim", xLimitCost, config['stateLimWeightTerminal'])
    if('all' in WHICH_COSTS or 'force' in WHICH_COSTS):
      terminal_costs.addCost("force", frameForceCost, config['frameForceWeightTerminal'])

    # Terminal DAM (Contact or FreeFwd)
    if(CONTACT):
      dam_t = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                actuation, 
                                                                crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                terminal_costs, 
                                                                inv_damping=0., 
                                                                enable_force=True)
    else:
      dam_t = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminal_costs)    
    
    # Terminal IAM
    terminalModel = crocoddyl.IntegratedActionModelLPF( dam_t, 
                                                      stepTime=0., 
                                                      withCostResidual=True, 
                                                      fc=f_c, 
                                                      cost_weight_w_reg=cost_w_reg, 
                                                      cost_weight_w_lim=cost_w_lim,
                                                      tau_plus_integration=tau_plus,
                                                      filter=lpf_type,
                                                      is_terminal=True)                                          
    
    # Add armature
    terminalModel.differential.armature = np.asarray(config['armature'])
    
    # Add contact model
    if(CONTACT):
      terminalModel.differential.contacts.addContact("contact", contact6d, active=True)

    print("[OCP] Created IAMs.")

    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
    
    # Warm start yb default
    ddp.xs = [y0 for i in range(N_h+1)]
    ddp.us = [pin_utils.get_u_grav_(y0[:nq], robot.model) for i in range(N_h)]
    
    print("[OCP] OCP is ready ! (CONTACT="+str(CONTACT)+")")
    print("[OCP]   Costs = "+str(WHICH_COSTS))
    return ddp