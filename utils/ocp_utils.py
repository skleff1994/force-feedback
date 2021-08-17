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


# Setup OCP and solver using Crocoddyl
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
    xRegCost = crocoddyl.CostModelResidual(state, 
                                           crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                           crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
    print("[OCP] Created state reg cost.")
       # Control regularization
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
      # Control limits penalization
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



# From Gabriele
class IntegratedActionModelLPF(crocoddyl.ActionModelAbstract):
    '''
        Add a low pass effect on the torque dynamics
            tau+ = alpha * tau + (1 - alpha) * w
        where alpha is a parameter depending of the memory of the system
        tau is the filtered torque included in the state and w the unfiltered control
        The state is augmented so that it includes the filtered torque
            y = [x, tau].T
        Initialized from DAM
    '''
    def __init__(self, diffModel, dt=1e-3, withCostResiduals=True, f_c = np.NaN):
            '''
                If f_c is undefined or NaN, it is assumed to be infinite, unfiltered case
            '''
            crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(diffModel.state.nx + diffModel.nu), diffModel.nu)
            self.differential = diffModel
            self.dt = dt
            self.withCostResiduals = withCostResiduals
            # Set LPF cut-off frequency
            self.set_alpha(f_c)
            self.nx = diffModel.state.nx
            self.ny = self.nu + self.nx
            # Integrate or not?
            if self.dt == 0:
                self.enable_integration_ = False
            else:
                self.enable_integration_ = True
            # Default unfiltered control cost (reg + lim)
            self.set_w_reg_lim_costs(1e-2, 
                                     np.zeros(self.differential.nu), 
                                     1e-1,
                                     np.zeros(self.differential.nu))

    def set_w_reg_lim_costs(self, w_reg_weight, w_reg_ref, w_lim_weight, w_lim_ref):
        '''
        Set cost on unfiltered input
        '''
        self.w_reg_weight = w_reg_weight
        self.w_reg_ref = w_reg_ref
        self.w_lim_weight = w_lim_weight
        self.w_lim_ref = w_lim_ref
        self.activation = crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(-self.differential.state.pinocchio.effortLimit, 
                                                self.differential.state.pinocchio.effortLimit) )

    def createData(self):
        '''
            The data is created with a custom data class that contains the filtered torque tau_plus and the activation
        '''
        data = IntegratedActionDataLPF(self)
        return data

    def set_alpha(self, f_c = None):
        '''
            Sets the parameter alpha according to the cut-off frequency f_c
            alpha = 1 / (1 + 2pi dt f_c)
        '''
        if f_c > 0 and self.dt > 0:
            omega = 1/(2 * np.pi * self.dt * f_c)
            self.alpha = omega/(omega + 1)
        else:
            self.alpha = 0

    def calc(self, data, y, w = None):
        '''
        Euler integration (or no integration depending on dt)
        '''
        # what if w is none?
        x = y[:self.differential.state.nx]
        # filtering the torque with the previous state : get tau_q+ from w 
        data.tau_plus[:] = self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w
        # print("Data.tau_plus = ", data.tau_plus[0])
        # dynamics : get a_q = DAM(q, vq, tau_q+)
        self.differential.calc(data.differential, x, data.tau_plus)
        if self.withCostResiduals:
            data.r = data.differential.r
        # Euler integration step of dt : get v_q+, q+
        if self.enable_integration_:
            data.cost = self.dt * data.differential.cost
            # adding the cost on the unfiltered torque
            self.activation.calc(data.activation, w - self.w_lim_ref)
            data.cost += self.dt * self.w_lim_weight * data.activation.a_value + self.dt * (w - self.w_reg_ref) @ ( w - self.w_reg_ref ) / 2 * self.w_reg_weight
            data.dx = np.concatenate([x[self.differential.state.nq:] * self.dt + data.differential.xout * self.dt**2, data.differential.xout * self.dt])
            data.xnext[:self.nx] = self.differential.state.integrate(x, data.dx)
            data.xnext[self.nx:] = data.tau_plus
        else:
            data.dx = np.zeros(len(y))
            data.xnext[:] = y
            data.cost = data.differential.cost
            # adding the cost on the unfiltered torque
            self.activation.calc(data.activation, w - self.w_lim_ref)
            data.cost += self.w_lim_weight * data.activation.a_value + (w - self.w_reg_ref) @ ( w - self.w_reg_ref ) / 2 * self.w_reg_weight

        return data.xnext, data.cost

    def calcDiff(self, data, y, w=None):
        '''
        Compute derivatives 
        '''
        # First call calc
        self.calc(data, y, w)
        x = y[:-self.differential.nu]
        # Get derivatives of DAM under LP-Filtered input 
        self.differential.calcDiff(data.differential, x, data.tau_plus)
        # Get d(IAM)/dx =  [d(q+)/dx, d(v_q+)/dx] 
        dxnext_dx, dxnext_ddx = self.differential.state.Jintegrate(x, data.dx)
        # Get d(DAM)/dx , d(DAM)/du (why resize?)
        da_dx, da_du = data.differential.Fx, np.resize(data.differential.Fu, (self.differential.state.nv, self.differential.nu))
        ddx_dx = np.vstack([da_dx * self.dt, da_dx])
        # ??? ugly way of coding identity matrix ?
        ddx_dx[range(self.differential.state.nv), range(self.differential.state.nv, 2 * self.differential.state.nv)] += 1
        ddx_du = np.vstack([da_du * self.dt, da_du])

        # In this scope the data.* are in the augmented state coordinates
        # while all the differential dd are in the canonical x coordinates
        # we must set correctly the quantities where needed
        Fx = dxnext_dx + self.dt * np.dot(dxnext_ddx, ddx_dx)
        Fu = self.dt * np.dot(dxnext_ddx, ddx_du) # wrong according to NUM DIFF, no timestep

        # TODO why is this not multiplied by timestep?
        data.Fx[:self.nx, :self.nx] = Fx
        data.Fx[:self.nx, self.nx:self.ny] = self.alpha * Fu
        data.Fx[self.nx:, self.nx:] = self.alpha * np.eye(self.nu)
        # print('Fy : ', data.Fx)
        # TODO CHECKING WITH NUMDIFF, NO TIMESTEP HERE
        if self.nu == 1:
            data.Fu.flat[:self.nx] = (1 - self.alpha) * Fu
            data.Fu.flat[self.nx:] = (1 - self.alpha) * np.eye(self.nu)
        else:
            data.Fu[:self.nx, :self.nu] = (1 - self.alpha) * Fu
            data.Fu[self.nx:, :self.nu] = (1 - self.alpha) * np.eye(self.nu)

        if self.enable_integration_:

            data.Lx[:self.nx] = self.dt * data.differential.Lx
            data.Lx[self.nx:] = self.dt * self.alpha * data.differential.Lu

            data.Lu[:] = self.dt * (1 - self.alpha) * data.differential.Lu

            data.Lxx[:self.nx,:self.nx] = self.dt * data.differential.Lxx
            # TODO reshape is not the best, see better how to cast this
            data.Lxx[:self.nx,self.nx:] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.dt * self.alpha**2 * data.differential.Luu

            data.Lxu[:self.nx] = self.dt * (1 - self.alpha) * data.differential.Lxu
            data.Lxu[self.nx:] = self.dt * (1 - self.alpha) * self.alpha * data.differential.Luu

            data.Luu[:, :] = self.dt * (1 - self.alpha)**2 * data.differential.Luu

            # adding the unfiltered torque cost
            self.activation.calcDiff(data.activation, w - self.w_lim_ref)
            data.Lu[:] += self.dt * self.w_lim_weight * data.activation.Ar + (w - self.w_reg_ref) * self.dt * self.w_reg_weight
            data.Luu[:, :] += self.dt * self.w_lim_weight * data.activation.Arr + np.diag(np.ones(self.nu)) * self.dt * self.w_reg_weight

        else:

            data.Lx[:self.nx] = data.differential.Lx
            data.Lx[self.nx:] = self.alpha * data.differential.Lu

            data.Lu[:] = (1 - self.alpha) * data.differential.Lu

            data.Lxx[:self.nx,:self.nx] = data.differential.Lxx
            data.Lxx[:self.nx,self.nx:] = self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.alpha**2 * data.differential.Luu

            data.Lxu[:self.nx] = (1 - self.alpha) * data.differential.Lxu
            data.Lxu[self.nx:] = (1 - self.alpha) * self.alpha * data.differential.Luu

            data.Luu[:, :] = (1 - self.alpha)**2 * data.differential.Luu

            # adding the unfiltered torque cost
            self.activation.calcDiff(data.activation, w - self.w_lim_ref)
            data.Lu[:] += self.w_lim_weight * data.activation.Ar + (w - self.w_reg_ref) * self.w_reg_weight
            data.Luu[:, :] += self.w_lim_weight * data.activation.Arr + np.diag(np.ones(self.nu)) * self.w_reg_weight


class IntegratedActionDataLPF(crocoddyl.ActionDataAbstract):
    '''
    Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
    '''
    def __init__(self, am):
        crocoddyl.ActionDataAbstract.__init__(self, am)
        self.differential = am.differential.createData()
        self.activation = am.activation.createData()
        self.tau_plus = np.zeros(am.nu)
        self.Fx = np.zeros((am.ny, am.ny))
        self.Fu = np.zeros((am.ny, am.nu))
        self.Lx = np.zeros(am.ny)
        self.Lu = np.zeros(am.nu)
        self.Lxx = np.zeros((am.ny, am.ny))
        self.Lxu = np.zeros((am.ny, am.nu))
        self.Luu = np.zeros((am.nu,am.nu))


# Setup OCP and solver using Crocoddyl
def init_DDP_LPF(robot, config, x0, f_c=100):
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
    xRegCost = crocoddyl.CostModelResidual(state, 
                                           crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                           crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
    print("[OCP] Created state reg cost.")
       # Control regularization
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
      # Control limits penalization
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
    
    # LPF (CT) param                     
    alpha =  1 / (1 + 2*np.pi*dt*f_c) # Smoothing factor : close to 1 means f_c decrease, close to 0 means f_c very large 
    print("LOW-PASS FILTER : ")
    print("f_c   = ", f_c)
    print("alpha = ", alpha)
    
    # Create IAMs
    runningModels = []
    for i in range(N_h):
      # Using pure python
      runningModels.append(IntegratedActionModelLPF(
          crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                           actuation, 
                                                           crocoddyl.CostModelSum(state, nu=actuation.nu)), 
                                                           dt=0, f_c=f_c ) )
      # # Using bindings
      # runningModels.append(crocoddyl.IntegratedActionModelLPF( 
      #     crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
      #                                                         actuation, 
      #                                                         crocoddyl.ContactModelMultiple(state, actuation.nu), 
      #                                                         crocoddyl.CostModelSum(state, nu=actuation.nu), 
      #                                                         inv_damping=0., 
      #                                                         enable_force=False))) #, nu=actuation.nu, dt=dt, alpha=alpha ))

      # Add cost models
      runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['frameWeight']) 
      runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['xRegWeight'])
      runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['uRegWeight']) 
      # Add armature
      runningModels[i].differential.armature = np.asarray(config['armature'])

    # Terminal IAM + set armature
    # Using pure python
    terminalModel = IntegratedActionModelLPF(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                         actuation, 
                                                         crocoddyl.CostModelSum(state, nu=actuation.nu)),
                                                         dt=0, f_c=f_c )
    # # Using bindins
    # terminalModel = crocoddyl.IntegratedActionModelLPF(
    #     crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
    #                                                         actuation, 
    #                                                         crocoddyl.ContactModelMultiple(state, actuation.nu), 
    #                                                         crocoddyl.CostModelSum(state, nu=actuation.nu), 
    #                                                         inv_damping=0., 
    #                                                         enable_force=False)) #, nu=actuation.nu, dt=0, alpha=alpha )
                                                            
    # Add cost models
    terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
    terminalModel.differential.costs.addCost("stateReg", xRegCost, config['xRegWeightTerminal'])
    terminalModel.differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeightTerminal'])                                                
    terminalModel.differential.armature = np.asarray(config['armature'])
    
    print("[OCP] Created IAMs.")
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    print("[OCP] :", problem.runningModels[0].state.nv)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)
    print("[OCP] OCP is ready.")
    print("-------------------------------------------------------------------")
    return ddp
