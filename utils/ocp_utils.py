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
        
# # Check
# X = np.ones((10,2))
# for i in range(X.shape[0]):
#     X[i] = i**2
# interp = interpolate(X, 3)
# import matplotlib.pyplot as plt
# plt.plot(np.linspace(0,1,X.shape[0]), X[:,1], 'ro')
# plt.plot(np.linspace(0,1,interp.shape[0]), interp[:,1], 'g*')
# plt.show()



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
def init_DDP(robot, config, x0):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
     - Running cost: EE placement (Mref) + x_reg (xref) + u_reg (uref)
     - Terminal cost: EE placement (Mref) + EE velocity (0) + x_reg (xref)
    Mref = initial frame placement read in config
    xref = initial state read in config
    uref = initial gravity compensation torque (from xref)
    INPUT: 
        robot  : pinocchio robot wrapper
        config : dict from YAML config file describing task and MPC params
        x0     : initial state of shooting problem
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
    # Construct cost function terms
      # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
      # State regularization
    stateRegWeights = np.asarray(config['stateRegWeights'])
    x_reg_ref = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) #np.zeros(nq+nv)     
    xRegCost = crocoddyl.CostModelResidual(state, 
                                           crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                           crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
    print("[OCP] Created state reg cost.")
       # Control regularization
    ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
    u_grav = pin.rnea(robot.model, robot.data, x_reg_ref[:nq], np.zeros((nv,1)), np.zeros((nq,1))) #
    uRegCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                          crocoddyl.ResidualModelControl(state, u_grav))
    print("[OCP] Created ctrl reg cost.")
      # State limits penalization
    x_lim_ref  = np.zeros(nq+nv)
    xLimitCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(state.lb, state.ub)), 
                                          crocoddyl.ResidualModelState(state, x_lim_ref, actuation.nu))
    print("[OCP] Created state lim cost.")
      # Control limits penalization
    u_min = -np.asarray(config['u_lim']) 
    u_max = +np.asarray(config['u_lim']) 
    u_lim_ref = np.zeros(nq)
    uLimitCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                            crocoddyl.ResidualModelControl(state, u_lim_ref))
    print("[OCP] Created ctrl lim cost.")
      # End-effector placement 
    # p_target = np.asarray(config['p_des']) 
    # M_target = pin.SE3(M_ee.rotation.T, p_target)
    desiredFramePlacement = M_ee.copy() # M_target
    # p_ref = desiredFramePlacement.translation.copy()
    framePlacementWeights = np.asarray(config['framePlacementWeights'])
    framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                     crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                     crocoddyl.ResidualModelFramePlacement(state, 
                                                                                           id_endeff, 
                                                                                           desiredFramePlacement, 
                                                                                           actuation.nu)) 
    print("[OCP] Created frame placement cost.")
      # End-effector velocity 
    desiredFrameMotion = pin.Motion(np.array([0.,0.,0.,0.,0.,0.]))
    frameVelocityWeights = np.ones(6)
    frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                    crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                    crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                         id_endeff, 
                                                                                         desiredFrameMotion, 
                                                                                         pin.LOCAL, 
                                                                                         actuation.nu)) 
    print("[OCP] Created frame velocity cost.")
    
    # Create IAMs
    runningModels = []
    for i in range(N_h):
        # Create IAM 
        runningModels.append(crocoddyl.IntegratedActionModelEuler( 
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                             actuation, 
                                                             crocoddyl.CostModelSum(state, nu=actuation.nu)), dt ) )
        # Add cost models
        runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['frameWeight'])
        runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['xRegWeight'])
        runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['uRegWeight'])
        # runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['xLimWeight'])
        # runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['uLimWeight'])
        # Add armature
        runningModels[i].differential.armature = np.asarray(config['armature'])
    # Terminal IAM + set armature
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                            actuation, 
                                                            crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
    # Add cost models
    terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
    terminalModel.differential.costs.addCost("stateReg", xRegCost, config['xRegWeightTerminal'])
    terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])
    # terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['xLimWeightTerminal'])
    # Add armature
    terminalModel.differential.armature = np.asarray(config['armature']) 
    print("[OCP] Created IAMs.")
    
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
    print("[OCP] OCP is ready.")
    print("-------------------------------------------------------------------")
    return ddp


# Setup OCP and solver using Crocoddyl
def init_DDP_LPF(robot, config, y0, callbacks=False, 
                                    cost_w=0.1, 
                                    tau_plus=True,
                                    lpf_type=0,
                                    which_costs=['all']):
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
        cost_w      : cost weight on reg. of unfiltered input w around 0
        tau_plus    : use "tau_plus" integration if True, "tau" otherwise
        lpf_type    : use expo moving avg (0), classical lpf (1) or exact (2)
        which_costs : which cost terms in the running & terminal cost?
                        'placement', 'velocity', 'stateReg', 'ctrlReg'
                        'stateLim', 'ctrlLim'
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
    # Construct cost function terms
      # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
      # State regularization
    stateRegWeights = np.asarray(config['stateRegWeights'])
    x_reg_ref = np.concatenate([y0[:nq], np.zeros(nv)]) #np.zeros(nq+nv)     
    xRegCost = crocoddyl.CostModelResidual(state, 
                                           crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                           crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
    print("[OCP] Created state reg cost.")
       # Control regularization
    ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
    u_grav = y0[-nq:] #pin.rnea(robot.model, robot.data, x_reg_ref[:nq], np.zeros((nv,1)), np.zeros((nq,1))) #
    uRegCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                          crocoddyl.ResidualModelControl(state, u_grav))
    print("[OCP] Created ctrl reg cost.")
      # State limits penalization
    x_lim_ref  = np.zeros(nq+nv)
    xLimitCost = crocoddyl.CostModelResidual(state, 
                                             crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(state.lb, state.ub)), 
                                             crocoddyl.ResidualModelState(state, x_lim_ref, actuation.nu))
    print("[OCP] Created state lim cost.")
      # Control limits penalization
    u_min = -np.asarray(config['u_lim']) 
    u_max = +np.asarray(config['u_lim']) 
    u_lim_ref = np.zeros(nq)
    uLimitCost = crocoddyl.CostModelResidual(state, 
                                             crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                             crocoddyl.ResidualModelControl(state, u_lim_ref))
    print("[OCP] Created ctrl lim cost.")
      # End-effector placement 
    p_target = np.asarray(config['p_des']) 
    M_target = pin.SE3(M_ee.rotation.T, p_target)
    desiredFramePlacement = M_target #M_ee.copy()
    # desiredFramePlacement.translation[0] += 0.1
    # desiredFramePlacement.translation[1] -= 0.2
    framePlacementWeights = np.asarray(config['framePlacementWeights'])
    framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                     crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                     crocoddyl.ResidualModelFramePlacement(state, 
                                                                                           id_endeff, 
                                                                                           desiredFramePlacement, 
                                                                                           actuation.nu)) 
    print("[OCP] Created frame placement cost.")
      # End-effector velocity 
    desiredFrameMotion = pin.Motion(np.array([0.,0.,0.,0.,0.,0.]))
    frameVelocityWeights = np.ones(6)
    frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                    crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                    crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                         id_endeff, 
                                                                                         desiredFrameMotion, 
                                                                                         pin.LOCAL, 
                                                                                         actuation.nu)) 
    print("[OCP] Created frame velocity cost.")
    
    # LPF (CT) param   
    f_c = config['f_c']    
    if(lpf_type==0):
        alpha = np.exp(-2*np.pi*f_c*dt)
    if(lpf_type==1):
        alpha = 1./float(1+2*np.pi*f_c*dt)
    if(lpf_type==2):
        y = np.cos(2*np.pi*f_c*dt)
        alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 
    print("LOW-PASS FILTER : ")
    print("f_c   = ", f_c)
    print("alpha = ", alpha)
    
    # Create IAMs
    runningModels = []
    for i in range(N_h):
      # Using pure python
      runningModels.append(crocoddyl.IntegratedActionModelLPF(
                  crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                                   actuation, 
                                                                   crocoddyl.CostModelSum(state, nu=actuation.nu)), 
                                                              stepTime=dt, 
                                                              withCostResidual=True, 
                                                              fc=f_c, 
                                                              cost_weight_w=cost_w, 
                                                              tau_plus_integration=tau_plus,
                                                              filter=lpf_type))

      # Add cost models
      if('all' or 'placement' in which_costs):
        runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['frameWeight']) 
      if('all' or 'velocity' in which_costs):
        runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
      if('all' or 'stateReg' in which_costs):
        runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['xRegWeight'])
      if('all' or 'ctrlReg' in which_costs):
        runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['uRegWeight']) 
      if('all' or 'stateLim' in which_costs):
        runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['xLimWeight'])
      if('all' or 'ctrlLim' in which_costs):
        runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['uLimWeight'])
      # Add armature
      runningModels[i].differential.armature = np.asarray(config['armature'])

    # Terminal IAM + set armature
    # Using pure python
    terminalModel = crocoddyl.IntegratedActionModelLPF(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                             actuation, 
                                                             crocoddyl.CostModelSum(state, nu=actuation.nu)),
                                                      stepTime=0., 
                                                      withCostResidual=True, 
                                                      fc=f_c, 
                                                      cost_weight_w=cost_w, 
                                                      tau_plus_integration=tau_plus,
                                                      filter=lpf_type)
                                                            
    # Add cost models
    if('all' or 'placement' in which_costs):
      terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
    if('all' or 'velocity' in which_costs):
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])
    if('all' or 'stateReg' in which_costs):
      terminalModel.differential.costs.addCost("stateReg", xRegCost, config['xRegWeightTerminal'])
    # Add armature
    terminalModel.differential.armature = np.asarray(config['armature'])
    
    print("[OCP] Created IAMs.")

    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
    print("[OCP] OCP is ready.")
    print("-------------------------------------------------------------------")
    return ddp