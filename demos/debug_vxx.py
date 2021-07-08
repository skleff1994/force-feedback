import numpy as np
import crocoddyl
from utils import path_utils, data_utils, ocp_utils, plot_utils
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin

# Read config file
config = path_utils.load_config_file('static_reaching_task3')
simu_freq = 20e3
# Robot pin wrapper
robot = RobotWrapper.BuildFromURDF(path_utils.get_urdf_path('iiwa'), path_utils.get_mesh_dir())
nq, nv = robot.model.nq, robot.model.nv
nu = nq
q0 = np.asarray(config['q0'])
dq0 = np.asarray(config['dq0'])
robot.forwardKinematics(q0, dq0)
robot.framesForwardKinematics(q0)
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]

# Load data 
d = data_utils.load_data('/home/skleff/force-feedback/data/DATASET3_change_task_increase_freq/10000/tracking=False_10000Hz__exp_9.npz')
plan_freq = 10e3
# Change costs as in recorded simulation
config['frameWeight'] = 51200
config['xRegWeight'] = 1.953125e-5
config['uRegWeight'] = 3.90625e-5
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
# Check VP values
print(lambda_a)
print(lambda_b)
# Creating the DDP solver 
ddp_a = ocp_utils.init_DDP(robot, config, x0a)
ddp_b = ocp_utils.init_DDP(robot, config, x0b)
# solve for each point
ddp_a.setCallbacks([crocoddyl.CallbackLogger(),
                   crocoddyl.CallbackVerbose()])
ddp_b.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose()])
ddp_a.solve(ddp_a.xs, ddp_a.us, maxiter=10, isFeasible=False)
ddp_b.solve(ddp_b.xs, ddp_b.us, maxiter=10, isFeasible=False)

# Plot results
# plot_utils.plot_ddp_ricatti(ddp_a)
plot_utils.plot_ddp_results([ddp_a, ddp_b], robot, id_endeff, which_plots=['K', 'vxx'])
