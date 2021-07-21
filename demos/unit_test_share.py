import numpy as np
import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper
np.set_printoptions(precision=4, linewidth=180)
import pinocchio as pin

'''
Unit test to reproduce the bug of 'peaks in Vxx' . 
an be shared using the data_debug_vxx.npz data file + robot_properties_kuka (urdf and mesh)
'''

# Load robot model
simu_freq = 20e3
plan_freq = 10e3 
  # PATH TO URDF and MESH
from utils import path_utils
urdf_path = '/home/skleff/robot_properties_kuka/urdf/iiwa.urdf'
mesh_path = '/home/skleff/robot_properties_kuka' 
robot = RobotWrapper.BuildFromURDF(path_utils.get_urdf_path('iiwa'), path_utils.get_mesh_dir())
nq, nv = robot.model.nq, robot.model.nv
nu = nq
q0 =  np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
dq0 = np.zeros(7)
robot.forwardKinematics(q0, dq0)
robot.framesForwardKinematics(q0)

# Load data file
SAVE_PATH = '/home/skleff/force-feedback/data/data_debug_vxx.npz'
dat = np.load(SAVE_PATH, allow_pickle=True)
data = dat['data'][()]
print("xa mea  : ", data['x0a'], " taken at ka = ", data['ka'])
print("xa plan : ", data['x0b'], " taken at kb = ", data['kb'])

# OCP parameters 
dt = 0.03               # OCP integration step (s)               
N_h = 30                # Number of knots in the horizon 
# Model params
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]
nq, nv = robot.model.nq, robot.model.nv
# Construct cost function terms
    # State and actuation models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
    # State regularization
stateRegWeights = np.array([1., 1., 1., 1., 1., 1., 1., 5., 5., 5., 5., 5., 5., 5.])
x_reg_ref = np.concatenate([q0, dq0]) 
xRegCost = crocoddyl.CostModelState(state, 
                                    crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                    x_reg_ref, 
                                    actuation.nu)
    # Control regularization
ctrlRegWeights = np.array([1., 1., 1., 1., 1., 1., 1.])
u_grav = pin.rnea(robot.model, robot.data, x_reg_ref[:nq], np.zeros((nv,1)), np.zeros((nq,1))) 
uRegCost = crocoddyl.CostModelControl(state, 
                                    crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                    u_grav)
    # End-effector placement 
desiredFramePlacement = M_ee.copy() 
framePlacementWeights = np.array([1., 1., 1., 1., 1., 1.])
framePlacementCost = crocoddyl.CostModelFramePlacement(state, 
                                                    crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                    crocoddyl.FramePlacement(id_endeff, desiredFramePlacement), 
                                                    actuation.nu) 
    # End-effector velocity 
desiredFrameMotion = pin.Motion(np.array([0.,0.,0.,0.,0.,0.]))
frameVelocityWeights = np.ones(6)
frameVelocityCost = crocoddyl.CostModelFrameVelocity(state, 
                                                    crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                    crocoddyl.FrameMotion(id_endeff, desiredFrameMotion), 
                                                    actuation.nu) 

# Create IAMs
runningModels = []
for i in range(N_h):
    # Create IAM 
    runningModels.append(crocoddyl.IntegratedActionModelEuler( 
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                        actuation, 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu)), dt ) )
    # Add cost models
    runningModels[i].differential.costs.addCost("placement", framePlacementCost, 160000.)
    runningModels[i].differential.costs.addCost("stateReg", xRegCost, 6.25e-06)
    runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, 1.25e-05)
    # Add armature
    runningModels[i].differential.armature = np.array([.1, .1, .1, .1, .1, .1, .01])
# Terminal IAM + set armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                        actuation, 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
# Add cost models
terminalModel.differential.costs.addCost("placement", framePlacementCost, 1000.)
terminalModel.differential.costs.addCost("stateReg", xRegCost, 0.01)
terminalModel.differential.costs.addCost("velocity", frameVelocityCost, 10000.)
# Add armature
terminalModel.differential.armature = np.array([.1, .1, .1, .1, .1, .1, .01])


# Create the shooting problems and solvers
ddp_a = crocoddyl.SolverFDDP(crocoddyl.ShootingProblem(data['x0a'], runningModels, terminalModel))
ddp_b = crocoddyl.SolverFDDP(crocoddyl.ShootingProblem(data['x0b'], runningModels, terminalModel))
ddp_a.setCallbacks([crocoddyl.CallbackLogger(),
                   crocoddyl.CallbackVerbose()])
ddp_b.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose()])

# Solve both DDPs 
ddp_a.solve(data['xwsa'], data['uwsa'], maxiter=10, isFeasible=False)
ddp_b.solve(data['xwsb'], data['uwsb'], maxiter=10, isFeasible=False)

# Id of the eigenvalue / dimension we wanna look at
vals_a = np.sort(np.linalg.eigvals(ddp_a.Vxx[0]))[::-1]
vals_b = np.sort(np.linalg.eigvals(ddp_b.Vxx[0]))[::-1]
print("New spectrum a = \n", vals_a)
print("Old spectrum a = \n", data['vxx_eig_a'])
print("New spectrum b = \n", vals_b)
print("Old spectrum b = \n", data['vxx_eig_b'])

# Compare state predictions
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt
dt_plan = float(1./plan_freq)
tspan = np.linspace(data['ka']*dt_plan, ( data['ka']+N_h )*dt_plan , N_h +1)
for i in range(nq):
    plt.plot(tspan, np.array(data['xsa'])[:, i], linestyle='-', alpha=0.5)
    plt.plot(tspan, np.array(ddp_a.xs)[:, i], linestyle='-.')
    plt.grid(True)
for i in range(nq):
    plt.plot(tspan, np.array(data['xsa'])[:, nq+i], linestyle='-', alpha=0.5)
    plt.plot(tspan, np.array(ddp_a.xs)[:, nq+i], linestyle='-.')
    plt.grid(True)
plt.show()
