import importlib_resources
import yaml
import os

import matplotlib.pyplot as plt

import crocoddyl
import numpy as np
import pinocchio as pin


# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f)
    return data 

# Load config file
def load_config_file(config_name):
    '''
    Loads YAML config file in demos/config as a dict
    '''
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../demos', 'config/'))
    config_file = config_path+"/"+config_name+".yml"
    config = load_yaml_file(config_file)
    return config


# Get pin grav torque
def get_u_grav(q, pin_robot):
    '''
    Return gravity torque at q
    '''
    return pin.computeGeneralizedGravity(pin_robot.model, pin_robot.data, q)

# Post-process trajectories with pinocchio
def get_p(q, pin_robot, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        robot     : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    p = np.empty((N,3))
    for i in range(N):
        pin.forwardKinematics(pin_robot.model, pin_robot.data, q[i])
        pin.updateFramePlacements(pin_robot.model, pin_robot.data)
        p[i,:] = pin_robot.data.oMf[id_endeff].translation.T
    return p


# Setup OCP and solver using Crocoddyl
def init_DDP(robot, config, x0, callbacks=False, which_costs=['all'], dt=None, N_h=None):
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
          which_costs : which cost terms in the running & terminal cost?
                          'placement', 'velocity', 'stateReg', 'ctrlReg'
                          'stateLim', 'ctrlLim'
      OUTPUT:
        FDDP solver
    '''
    
    # OCP parameters
    if(dt is None):
      dt = config['dt']                   # OCP integration step (s)    
    if(N_h is None):
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
    if('all' in which_costs or 'stateReg' in which_costs):
      stateRegWeights = np.asarray(config['stateRegWeights'])
      x_reg_ref = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) #np.zeros(nq+nv)     
      xRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                            crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
      print("[OCP] Added state reg cost.")  
    # Control regularization
    if('all' in which_costs or 'ctrlReg' in which_costs):
      ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
      u_grav = pin.rnea(robot.model, robot.data, x0[:nq], np.zeros((nv,1)), np.zeros((nq,1))) #
      uRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                            crocoddyl.ResidualModelControlGrav(state))
      print("[OCP] Added ctrl reg cost.")
    # State limits penalization
    if('all' in which_costs or 'stateLim' in which_costs):
      x_lim_ref  = np.zeros(nq+nv)
      q_max = 0.95*state.ub[:nq] # 95% percent of max q
      v_max = np.ones(7)         # [-1,+1] for max v
      x_max = np.concatenate([q_max, v_max]) # state.ub
      xLimitCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-x_max, x_max)), 
                                            crocoddyl.ResidualModelState(state, x_lim_ref, actuation.nu))
      print("[OCP] Added state lim cost.\n")
    # Control limits penalization
    if('all' in which_costs or 'ctrlLim' in which_costs):
      u_min = -np.asarray(config['u_lim']) 
      u_max = +np.asarray(config['u_lim']) 
      u_lim_ref = np.zeros(nq)
      uLimitCost = crocoddyl.CostModelResidual(state, 
                                              crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                              crocoddyl.ResidualModelControl(state, u_lim_ref))
      print("[OCP] Added ctrl lim cost.")
    # End-effector placement 
    if('all' in which_costs or 'placement' in which_costs):
      p_target = np.asarray(config['p_des']) 
      # M_target = pin.SE3(M_ee.rotation, p_target)
      rot = np.eye(3)
      tx = np.pi/3
      rot[1,1] = np.cos(tx)
      rot[1,2] = -np.sin(tx)
      rot[2,1] = np.sin(tx)
      rot[2,2] = np.cos(tx)
      M_target = pin.SE3(rot, p_target)
      desiredFramePlacement = M_target
      # p_ref = desiredFramePlacement.translation.copy()
      framePlacementWeights = np.asarray(config['framePlacementWeights'])
      framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                      crocoddyl.ResidualModelFramePlacement(state, 
                                                                                            id_endeff, 
                                                                                            desiredFramePlacement, 
                                                                                            actuation.nu)) 
      print("[OCP] Added frame placement cost.\n")
      print("[OCP]    Desired placement : \n") 
      print(M_target) 
    # End-effector velocity
    if('all' in which_costs or 'velocity' in which_costs): 
      desiredFrameMotion = pin.Motion(np.array([0.,0.,0.,0.,0.,0.]))
      frameVelocityWeights = np.ones(6)
      frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                      crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                          id_endeff, 
                                                                                          desiredFrameMotion, 
                                                                                          pin.LOCAL, 
                                                                                          actuation.nu)) 
      print("[OCP] Added frame velocity cost.")
    # Frame translation cost
    if('all' in which_costs or 'translation' in which_costs):
      p_target = np.asarray(config['p_des']) 
      desiredFrameTranslation = p_target
      # p_ref = desiredFramePlacement.translation.copy()
      frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
      frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                      crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                            id_endeff, 
                                                                                            desiredFrameTranslation, 
                                                                                            actuation.nu)) 
      print("[OCP] Added frame translation cost.\n")
      print("[OCP]    Desired translation : \n") 

    # Create IAMs
    runningModels = []
    for i in range(N_h):
        # Create IAM 
        runningModels.append(crocoddyl.IntegratedActionModelEuler( 
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                             actuation, 
                                                             crocoddyl.CostModelSum(state, nu=actuation.nu)), dt ) )
        # Add cost models
        if('all' in which_costs or 'translation' in which_costs):
          runningModels[i].differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeight'])
        if('all' in which_costs or 'velocity' in which_costs):
          runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
        if('all' in which_costs or 'stateReg' in which_costs):
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['stateRegWeight'])
        if('all' in which_costs or 'ctrlReg' in which_costs):
          runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['ctrlRegWeight'])
        if('all' in which_costs or 'stateLim' in which_costs):
          runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeight'])
        if('all' in which_costs or 'ctrlLim' in which_costs):
          runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['ctrlLimWeight'])
        # Add armature
        runningModels[i].differential.armature = np.asarray(config['armature'])
    
    # Terminal IAM + set armature
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                            actuation, 
                                                            crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
    # Add cost models
    if('all' in which_costs or 'placement' in which_costs):
      terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
    if('all' in which_costs or 'translation' in which_costs):
      terminalModel.differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeightTerminal'])
    if('all' in which_costs or 'velocity' in which_costs):
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])
    if('all' in which_costs or 'stateReg' in which_costs):
      terminalModel.differential.costs.addCost("stateReg", xRegCost, config['stateRegWeightTerminal'])
    if('all' in which_costs or 'velocity' in which_costs):
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])
    if('all' in which_costs or 'stateLim' in which_costs):
      terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeightTerminal'])
    # Add armature
    terminalModel.differential.armature = np.asarray(config['armature']) 
    print("[OCP] Created IAMs.")
    
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)

    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
    
    print("[OCP] OCP is ready.")
    print("-------------------------------------------------------------------")
    return ddp


# Plot from DDP solver 
def plot_ddp_results(ddp, robot, name_endeff='contact', which_plots='all', SHOW=False):
    '''
    Plot ddp results from 1 or several DDP solvers
    X, U, EE trajs
    INPUT 
      ddp         : DDP solver or list of ddp solvers
      robot       : pinocchio robot wrapper
      name_endeff : name of end-effector (in pin model) 
    '''
    if(type(ddp) != list):
        ddp = [ddp]
    for k,d in enumerate(ddp):
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all'):
                fig_x, ax_x = plot_ddp_state(ddp[k], SHOW=False)
            if('u' in which_plots or which_plots =='all'):
                fig_u, ax_u = plot_ddp_control(ddp[k], SHOW=False)
            if('p' in which_plots or which_plots =='all'):
                fig_p, ax_p = plot_ddp_endeff(ddp[k], robot, name_endeff, SHOW=False)
        # Overlay on top of first plot
        else:
            if('x' in which_plots or which_plots =='all'):
                plot_ddp_state(ddp[k], fig=fig_x, ax=ax_x, SHOW=False, marker='x')
            if('u' in which_plots or which_plots =='all'):
                plot_ddp_control(ddp[k], fig=fig_u, ax=ax_u, SHOW=False, marker='x')
            if('p' in which_plots or which_plots =='all'):
                plot_ddp_endeff(ddp[k], robot, name_endeff, fig=fig_p, ax=ax_p, SHOW=False, marker='x')

    if(SHOW):
      plt.show()
    
    fig = {}
    fig['p'] = fig_p
    fig['x'] = fig_x
    fig['u'] = fig_u

    ax = {}
    ax['p'] = ax_p
    ax['x'] = ax_x
    ax['u'] = ax_u

    return fig, ax
 
def plot_ddp_state(ddp, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp results (state)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    # Extract pos, vel trajs
    x = np.array(ddp.xs)
    q = x[:,:nq]
    v = x[:,nv:]
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 2, sharex='col') 
    if(label is None):
        label='State'
    for i in range(nq):
        # Positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label)
        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].grid(True)
        # Velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label)
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].grid(True)

    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('State trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control(ddp, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp results (control)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nu = ddp.problem.runningModels[0].nu
    # Extract pos, vel trajs
    u = np.array(ddp.us)
    # Plots
    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col') 
    if(label is None):
        label='Control'    
    for i in range(nu):
        # Positions
        ax[i].plot(tspan, u[:,i], linestyle='-', marker=marker, label=label)
        ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax[i].grid(True)
        # Set xlabel on bottom plot
        if(i == nu-1):
            ax[i].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Control trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff(ddp, robot, name_endeff, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp results (endeff)
    '''
    # Parameters
    N = ddp.problem.T
    dt = ddp.problem.runningModels[0].dt
    nq = ddp.problem.runningModels[0].state.nq
    # Extract EE traj
    x = np.array(ddp.xs)
    q = x[:,:nq]
    id_endeff = robot.model.getFrameId(name_endeff)
    p = get_p(q, robot, id_endeff)
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 1, sharex='col')
    if(label is None):
        label='End-effector'
    ylabels = ['Px', 'Py', 'Pz']
    for i in range(3):
        # Positions
        ax[i].plot(tspan, p[:,i], linestyle='-', marker=marker, label=label)
        ax[i].set_ylabel(ylabel=ylabels[i], fontsize=16)
        ax[i].grid(True)
    handles, labels = ax[i].get_legend_handles_labels()
    ax[i].set_xlabel('t (s)', fontsize=16)
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Endeffector trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

