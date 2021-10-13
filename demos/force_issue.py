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
runningCostModel.addCost("contactForce", contactForceCost, 10.)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 0.1)

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


# Plot force
import matplotlib.pyplot as plt
import pinocchio as pin
import eigenpy

 # Extract Croco data 
datas = [ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'] for i in range(T)]
ee_forces = np.array([data.jMf.actInv(data.f).vector for data in datas])
ee_force_ref = np.array([ddp.problem.runningModels[i].differential.costs.costs['contactForce'].cost.residual.reference.vector for i in range(T)])
 # Manual computation using pinocchio
qs = np.array(ddp.xs)[:,:nq]; vs = np.array(ddp.xs)[:,nq:]; us = np.array(ddp.us)
fs = np.zeros((T, 6))
REG = 0.
for i in range(T):
    # Get Jacobian and spatial acceleration at EE frame
    pin.forwardKinematics(robot.model, robot.data, qs[i,:], vs[i,:], np.zeros(nq))
    pin.updateFramePlacements(robot.model, robot.data)
    gamma = -pin.getFrameAcceleration(model, robot.data, contact_frame_id, pin.ReferenceFrame.LOCAL)
    pin.computeJointJacobians(robot.model, robot.data)
    J = pin.getFrameJacobian(robot.model, robot.data, contact_frame_id, pin.ReferenceFrame.LOCAL) 
    # Joint space inertia and its inverse + NL terms
    Minv = pin.computeMinverse(robot.model, robot.data, qs[i,:])
    h = pin.nonLinearEffects(robot.model, robot.data, qs[i,:], vs[i,:])
    # Contact force using f = (JMiJ')^+ ( JMi (b-tau) + gamma )
    LDLT = eigenpy.LDLT(J @ Minv @ J.T + REG*np.eye(6))
    fs[i,:]  = LDLT.solve(J @ Minv @ (h - us[i,:]) + gamma.vector)

fig, ax = plt.subplots(3, 2, sharex='col')
tspan = np.linspace(0, T*dt, T)
xyz = ['x', 'y', 'z']
for i in range(3):
    # translation
    ax[i,0].plot(tspan, ee_forces[:,i], linestyle='-', label='Croco')
    ax[i,0].plot(tspan, fs[:,i], linestyle='-.', label='manual')
    ax[i,0].plot(tspan, ee_force_ref[:,i], linestyle='-.', color='k', label='reference', alpha=0.5)
    ax[i,0].set_ylabel('$\\lambda^{lin}_%s$ (N)'%xyz[i], fontsize=16)
    # rotation
    ax[i,1].plot(tspan, ee_forces[:,3+i], linestyle='-', label='Croco')
    ax[i,1].plot(tspan, fs[:,3+i], linestyle='-.', label='manual')
    ax[i,1].plot(tspan, ee_force_ref[:,3+i], linestyle='-.', color='k', label='reference', alpha=0.5)
    ax[i,1].set_ylabel('$\\lambda^{ang}_%s$ (Nm)'%xyz[i], fontsize=16)
ax[i,0].set_xlabel('t (s)', fontsize=16)
ax[i,1].set_xlabel('t (s)', fontsize=16)
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 16})
fig.suptitle('End-effector forces: linear and angular', size=18)
plt.show()