import sys
sys.path.append('.')

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)
import crocoddyl
import pinocchio as pin

from dam import DAMSoftContactDynamics, DADSoftContactDynamics
from core_mpc.pin_utils import load_robot_wrapper

robot = load_robot_wrapper('iiwa')
model = robot.model
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.random.rand(nv)
tau = np.random.rand(nq)
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
frameId = model.getFrameId('contact')

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
# terminalCostModel = crocoddyl.CostModelSum(state)


# Create cost terms 
  # Control regularization cost
uResidual = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # endeff frame translation cost
endeff_frame_id = model.getFrameId("contact")
# endeff_translation = robot.data.oMf[endeff_frame_id].translation.copy()
endeff_translation = np.array([-0.4, 0.3, 0.7]) # move endeff +10 cm along x in WORLD frame
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)


# Add costs
runningCostModel.addCost("stateReg", xRegCost, 1e-1)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("translation", frameTranslationCost, 10)
# terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
# terminalCostModel.addCost("translation", frameTranslationCost, 10)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = DAMSoftContactDynamics(state, actuation, runningCostModel, frameId, pinRefFrame=pin.WORLD)
running_DAD = running_DAM.createData()

# Numdiff verson
running_DAM_ND = crocoddyl.DifferentialActionModelNumDiff(running_DAM, False)
running_DAM_ND.disturbance = 1e-6
running_DAD_ND = running_DAM_ND.createData()


running_DAM_ND.calc(running_DAD_ND, x0, tau)
running_DAM.calc(running_DAD, x0, tau)
# print(running_DAD.xout)
# print(running_DAD_ND.xout)
# print(running_DAD.cost)
# print(running_DAD_ND.cost)


# Numerical difference function
def numdiff(f,x0,h=1e-6):
    f0 = f(x0).copy()
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = x0[ix]
    return np.array(Fx).T


# croco nd
running_DAM_ND.calcDiff(running_DAD_ND, x0, tau)
# analytic
running_DAM.calcDiff(running_DAD, x0, tau)
# print(running_DAD_ND.Fx)
# custom nd
d = running_DAM.createData()
fx_nd = numdiff(lambda x_:running_DAM.calc(d, x_, tau), x0)

RTOL            = 1e-3 #1e-3
ATOL            = 1e-4 #1e-5
# print(running_DAD.Fx)
print("analytic vs custom nd : \n", np.isclose(running_DAD.Fx, fx_nd, RTOL, ATOL))
print("custom nd vs croco nd : \n", np.isclose(running_DAD_ND.Fx, fx_nd, RTOL, ATOL))
print("analytic vs croco nd : \n", np.isclose(running_DAD_ND.Fx, running_DAD.Fx, RTOL, ATOL))
# print(np.isclose(running_DAD.Fx, running_DAD_ND.Fx, RTOL, ATOL))
# print(np.isclose(running_DAD.Fu, running_DAD_ND.Fu, RTOL, ATOL))
# assert(np.linalg.norm(running_DAD.Fx - running_DAD_ND.Fx) < 1e-2)

# terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

# # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
# dt = 1e-2
# runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
# terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# # Optionally add armature to take into account actuator's inertia
# runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# # Create the shooting problem
# T = 250
# problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# # Create solver + callbacks
# ddp = crocoddyl.SolverFDDP(problem)
# ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                 crocoddyl.CallbackVerbose()])
# # Warm start : initial state + gravity compensation
# xs_init = [x0 for i in range(T+1)]
# us_init = ddp.problem.quasiStatic(xs_init[:-1])

# # Solve
# ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
