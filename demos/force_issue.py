import crocoddyl
import pinocchio
import numpy as np
import example_robot_data


robot = example_robot_data.load('talos_arm')
model = robot.model
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv


q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
v0 = np.zeros(nv)

state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)
# contact model 6D
contact_frame_id = model.getFrameId("gripper_left_motor_single_link")
contact6d = crocoddyl.ContactModel6D(state, contact_frame_id, robot.data.oMf[contact_frame_id], np.array([0., 0.])) 
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel.addContact("contact", contact6d, active=True)

# Cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
pose_frame_id = model.getFrameId("gripper_left_joint")
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, pose_frame_id, robot.data.oMf[pose_frame_id])
frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pinocchio.Force(np.array([0., 0., 20., 0., 0., 0.])), 6, actuation.nu)
uResidual = crocoddyl.ResidualModelControl(state)
xResidual = crocoddyl.ResidualModelControl(state)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
runningCostModel.addCost("gripperPose", goalTrackingCost, 0.1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e-3)
runningCostModel.addCost("contactForce", contactForceCost, 1.)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)

dt = 1e-3
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True), dt)
runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, inv_damping=0., enable_force=True), 0.)
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
x0 = np.concatenate([q0, v0])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem)

ddp.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])

ddp.solve()

from utils import data_utils, plot_utils, pin_utils
import matplotlib.pyplot as plt
ddp_data = data_utils.extract_ddp_data(ddp, CONTACT=True)
fig, ax = plot_utils.plot_ddp_force(ddp_data, SHOW=False)
f = pin_utils.get_f_lambda(np.array(ddp.xs)[:,:nq], np.array(ddp.xs)[:,nq:], np.array(ddp.us), model, contact_frame_id, REG=0.)
for i in range(3):
    ax[i,0].plot(np.linspace(0 ,T*dt, T), f[:,i], '-.', label='calculated')
    ax[i,1].plot(np.linspace(0, T*dt, T), f[:,3+i], '-.', label='calculated')
plt.show()