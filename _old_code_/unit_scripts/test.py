import crocoddyl
import pinocchio
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
np.set_printoptions(precision=4, linewidth=180)

# Load URDF
urdf_path = '/home/skleff/robot_properties_kuka/urdf/iiwa.urdf'
mesh_path = '/home/skleff/robot_properties_kuka'
robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_path) 
model = robot.model
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv)
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)

# Setup OCP 
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)
# contact model 6D
contact_frame_id = model.getFrameId("contact")
# Contact placement ref
contact_placement = robot.data.oMf[contact_frame_id].copy()
contact6d = crocoddyl.ContactModel6D(state, contact_frame_id, contact_placement, np.array([0., 0.])) 
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel.addContact("contact", contact6d, active=True)
# Cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
pose_frame_id = model.getFrameId("contact")
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, pose_frame_id, robot.data.oMf[pose_frame_id])
frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pinocchio.Force(np.array([0., 0., -20., 0., 0., 0.])), 6, actuation.nu)
uResidual = crocoddyl.ResidualModelControl(state)
xResidual = crocoddyl.ResidualModelControl(state)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e-3)
runningCostModel.addCost("contactForce", contactForceCost, 1.)
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True), dt)
runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, inv_damping=0., enable_force=True), 0.)
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
x0 = np.concatenate([q0, v0])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
# Warm start
xs_init = [x0 for i in range(T+1)]
us_init = ddp.problem.quasiStatic(ddp.xs[:-1])
# Solve
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

