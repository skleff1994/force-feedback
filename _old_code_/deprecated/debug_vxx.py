import numpy as np
import pinocchio as pin
import crocoddyl
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from core_mpc import utils 
import pybullet as p

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
    # Create a Pybullet simulation environment + set simu freq
simu_freq = 20e3    
dt_simu = 1./simu_freq
env = BulletEnvWithGround(p.GUI, dt=dt_simu)
pybullet_simulator = IiwaRobot()
env.add_robot(pybullet_simulator)
    # Create a robot instance. This initializes the simulator as well.
robot = pybullet_simulator.pin_robot
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
    # Reset robot to initial state in PyBullet + update pinocchio data
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
dq0 = np.zeros(nv)
pybullet_simulator.reset_state(q0, dq0)
pybullet_simulator.forward_robot(q0, dq0)
    # Get initial frame placement
M_ee = robot.data.oMf[id_endeff]
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")

# # # # # # # # #
### SETUP OCP ### 
# # # # # # # # #
  # OCP parameters 
dt = 0.03                           # OCP integration step (s)               
N_h = 30                            # Number of knots in the horizon 
x0 = np.concatenate([q0, dq0])      # Initial state 
plan_freq = 10e3
  # Construct cost function terms
   # State and actuation models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
   # State regularization
stateRegWeights = np.array([1., 1., 1., 1., 1., 1., 1., 5., 5., 5., 5., 5., 5., 5.])
x_reg_ref = x0    
xRegCost = crocoddyl.CostModelState(state, 
                                    crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                    x_reg_ref, 
                                    actuation.nu)
print("[OCP] Created state reg cost.")
   # Control regularization
ctrlRegWeights = np.array([1., 1., 1., 1., 1., 1., 1.])
u_grav = pin.rnea(robot.model, robot.data, x0[:nq], np.zeros((nv,1)), np.zeros((nq,1))) #
uRegCost = crocoddyl.CostModelControl(state, 
                                      crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                      u_grav)
print("[OCP] Created ctrl reg cost.")
desiredFramePlacement = M_ee.copy() 
p_ref = desiredFramePlacement.translation.copy()
framePlacementWeights = np.array([1., 1., 1., 1., 1., 1.])
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
  runningModels[i].differential.costs.addCost("placement", framePlacementCost, 51200)
  runningModels[i].differential.costs.addCost("stateReg", xRegCost, 1.953125e-5)
  runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, 3.90625e-5)
  # Add armature
  runningModels[i].differential.armature = np.array([.1, .1, .1, .1, .1, .1, .01])
  # Terminal IAM + set armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                        actuation, 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
   # Add cost models
terminalModel.differential.costs.addCost("placement", framePlacementCost, 1000)
terminalModel.differential.costs.addCost("stateReg", xRegCost, 1e-2)
terminalModel.differential.costs.addCost("velocity", frameVelocityCost, 1e4)
  # Add armature
terminalModel.differential.armature = np.array([.1, .1, .1, .1, .1, .1, .01])
print("[OCP] Created IAMs.")


# Load data 
d = utils.load_data('/home/skleff/force-feedback/data/DATASET3_change_task_increase_freq/10000/tracking=False_10000Hz__exp_9.npz')
# Select a state at right times
ta = 0.5 
tb = 1.0
k_simu_a = int(simu_freq*ta)
k_simu_b = int(simu_freq*tb)
k_plan_a = int(plan_freq*ta)
k_plan_b = int(plan_freq*tb)
x0a = np.concatenate([d['q_mea'][k_simu_a, :], d['v_mea'][k_simu_a, :]])
x0b = np.concatenate([d['q_mea'][k_simu_b, :], d['v_mea'][k_simu_b, :]])
lambda_a = d['Vxx_eigval'][k_plan_a, 0]
lambda_b = d['Vxx_eigval'][k_plan_b, 0]

# Create the shooting problem
problem_a = crocoddyl.ShootingProblem(x0a, runningModels, terminalModel)
problem_b = crocoddyl.ShootingProblem(x0b, runningModels, terminalModel)
# Creating the DDP solver 
ddp_a = crocoddyl.SolverFDDP(problem_a)
ddp_b = crocoddyl.SolverFDDP(problem_b)
# solve for each point
ddp_a.setCallbacks([crocoddyl.CallbackLogger(),
                   crocoddyl.CallbackVerbose()])
ddp_b.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose()])
ddp_a.solve(ddp_a.xs, ddp_a.us, maxiter=10, isFeasible=False)
ddp_b.solve(ddp_b.xs, ddp_b.us, maxiter=10, isFeasible=False)


#################################
### EXTRACT SOLUTION AND PLOT ###
#################################
print("Extracting solution...")
# Extract solution trajectories
qa = np.empty((N_h+1, nq))
va = np.empty((N_h+1, nv))
qb = np.empty((N_h+1, nq))
vb = np.empty((N_h+1, nv))
p_eea = np.empty((N_h+1, 3))
p_eeb = np.empty((N_h+1, 3))

for i in range(N_h+1):
    qa[i,:] = ddp_a.xs[i][:nq].T
    va[i,:] = ddp_a.xs[i][nv:].T
    qb[i,:] = ddp_b.xs[i][:nq].T
    vb[i,:] = ddp_b.xs[i][nv:].T
    pin.forwardKinematics(robot.model, robot.data, qa[i])
    pin.updateFramePlacements(robot.model, robot.data)
    p_eea[i,:] = robot.data.oMf[id_endeff].translation.T
    pin.forwardKinematics(robot.model, robot.data, qb[i])
    pin.updateFramePlacements(robot.model, robot.data)
    p_eeb[i,:] = robot.data.oMf[id_endeff].translation.T
ua = np.empty((N_h, actuation.nu))
ub = np.empty((N_h, actuation.nu))

for i in range(N_h):
    ua[i,:] = ddp_a.us[i].T
    ub[i,:] = ddp_b.us[i].T

import matplotlib.pyplot as plt #; plt.ion()
# Create time spans for X and U + figs and subplots
tspan_x = np.linspace(0, N_h*dt, N_h+1)
tspan_u = np.linspace(0, N_h*dt, N_h)
fig_x, ax_x = plt.subplots(nq, 2)
fig_u, ax_u = plt.subplots(nq, 1)
fig_p, ax_p = plt.subplots(3,1)
# Plot joints pos, vel , acc, torques
for i in range(nq):
    # Positions
    ax_x[i,0].plot(tspan_x, qa[:,i], 'b', label='BEFORE')
    ax_x[i,0].plot(tspan_x, qb[:,i], 'r', label='AFTER')
    ax_x[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
    ax_x[i,0].grid()
    # Velocities
    ax_x[i,1].plot(tspan_x, va[:,i], 'b', label='BEFORE')
    ax_x[i,1].plot(tspan_x, vb[:,i], 'r', label='AFTER')
    ax_x[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
    ax_x[i,1].grid()
    # Torques
    ax_u[i].plot(tspan_u, ua[:,i], 'b', label='BEFORE') # feedforward term
    ax_u[i].plot(tspan_u, ub[:,i], 'r', label='AFTER') # feedforward term
    ax_u[i].set_ylabel(ylabel='$u_%d$'%i, fontsize=16)
    ax_u[i].grid()
    # Remove xticks labels for clarity 
    if(i != nq-1):
        for j in range(2):
            ax_x[i,j].set_xticklabels([])
        ax_u[i].set_xticklabels([])
    # Set xlabel on bottom plot
    if(i == nq-1):
        for j in range(2):
            ax_x[i,j].set_xlabel('t (s)', fontsize=16)
        ax_u[i].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
    fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
    handles_u, labels_u = ax_u[i].get_legend_handles_labels()
    fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
# Plot EE
ylabels_p = ['Px', 'Py', 'Pz']
for i in range(3):
    ax_p[i].plot(tspan_x, p_eea[:,i], 'b', label='BEFORE')
    ax_p[i].plot(tspan_x, p_eeb[:,i], 'r', label='AFTER')
    ax_p[i].set_ylabel(ylabel=ylabels_p[i], fontsize=16)
    ax_p[i].grid()
    handles_p, labels_p = ax_p[i].get_legend_handles_labels()
    fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
ax_p[-1].set_xlabel('t (s)', fontsize=16)
# Align labels + set titles
fig_x.align_ylabels()
fig_u.align_ylabels()
fig_p.align_ylabels()
fig_x.suptitle('Joint trajectories', size=16)
fig_u.suptitle('Joint torques', size=16)
fig_p.suptitle('End-effector trajectory', size=16)
plt.show()
