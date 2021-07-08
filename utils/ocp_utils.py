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

# Cost weights profiles, useful for reaching tasks/cost design
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
    # Model params
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
    xRegCost = crocoddyl.CostModelState(state, 
                                        crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                        x_reg_ref, 
                                        actuation.nu)
    print("[OCP] Created state reg cost.")
       # Control regularization
    ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
    u_grav = pin.rnea(robot.model, robot.data, x_reg_ref[:nq], np.zeros((nv,1)), np.zeros((nq,1))) #
    uRegCost = crocoddyl.CostModelControl(state, 
                                        crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                        u_grav)
    print("[OCP] Created ctrl reg cost.")
      # State limits penalization
    x_lim_ref  = np.zeros(nq+nv)
    xLimitCost = crocoddyl.CostModelState(state, 
                                        crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(state.lb, state.ub)), 
                                        x_lim_ref, 
                                        actuation.nu)
    print("[OCP] Created state lim cost.")
      # Control limits penalization
    u_min = -np.asarray(config['u_lim']) 
    u_max = +np.asarray(config['u_lim']) 
    u_lim_ref = np.zeros(nq)
    uLimitCost = crocoddyl.CostModelControl(state, 
                                            crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                            u_lim_ref)
    print("[OCP] Created ctrl lim cost.")
      # End-effector placement 
    # p_target = np.asarray(config['p_des']) 
    # M_target = pin.SE3(M_ee.rotation.T, p_target)
    desiredFramePlacement = M_ee.copy() # M_target
    # p_ref = desiredFramePlacement.translation.copy()
    framePlacementWeights = np.asarray(config['framePlacementWeights'])
    framePlacementCost = crocoddyl.CostModelFramePlacement(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                        crocoddyl.FramePlacement(id_endeff, desiredFramePlacement), 
                                                        actuation.nu) 
    print("[OCP] Created frame placement cost.")
      # End-effector velocity 
    desiredFrameMotion = pin.Motion(np.array([0.,0.,0.,0.,0.,0.]))
    frameVelocityWeights = np.ones(6)
    frameVelocityCost = crocoddyl.CostModelFrameVelocity(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                        crocoddyl.FrameMotion(id_endeff, desiredFrameMotion), 
                                                        actuation.nu) 
    print("[OCP] Created frame velocity cost.")
    
    # Create IAMs
    runningModels = []
    for i in range(N_h):
        # Create IAM 
        runningModels.append(crocoddyl.IntegratedActionModelEuler( 
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                            actuation, 
                                                            crocoddyl.CostModelSum(state, nu=actuation.nu)), dt ) )
        # Add cost models
        runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['frameWeight'])
        runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['xRegWeight'])
        runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['uRegWeight'])
        # runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['xLimWeight'])
        # runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['uLimWeight'])
        # Add armature
        runningModels[i].differential.armature = np.asarray(config['armature'])
    # Terminal IAM + set armature
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                            actuation, 
                                                            crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
    # Add cost models
    terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
    terminalModel.differential.costs.addCost("stateReg", xRegCost, config['xRegWeightTerminal'])
    terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])
    # terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['xLimWeightTerminal'])
    # Add armature
    terminalModel.differential.armature = np.asarray(config['armature']) 
    print("[OCP] Created IAMs.")
    
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
    print("[OCP] OCP is ready.")
    print("-------------------------------------------------------------------")
    return ddp
