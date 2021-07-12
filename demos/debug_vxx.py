import numpy as np
import crocoddyl
from numpy.core.multiarray import concatenate
from utils import path_utils, data_utils, ocp_utils, plot_utils
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)

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
d = data_utils.load_data('/home/skleff/force-feedback/data/DATASET6_change_task_increase_freq/10000/tracking=False_10000Hz__exp_0.npz')
plan_freq = 10e3

# Collect all the 'peak' point in data 
# Change costs as in recorded simulation
config['frameWeight'] = 51200
config['xRegWeight'] = 1.953125e-5
config['uRegWeight'] = 3.90625e-5
print("Cost weights ratio = ", str(config['frameWeight']/config['xRegWeight']))
# Select a state at right times
ta = 0.6658 #0.5
tb = 0.6659 #1.0 .66
k_simu_a = int(simu_freq*ta)
k_simu_b = int(simu_freq*tb)
k_plan_a = 6658 #int(plan_freq*ta)
k_plan_b = 6659 #int(plan_freq*tb)
x0a = np.concatenate([d['q_mea'][k_simu_a, :], d['v_mea'][k_simu_a, :]])
x0b = np.concatenate([d['q_mea'][k_simu_b, :], d['v_mea'][k_simu_b, :]])
x0a_ = np.concatenate([d['q_pred'][k_plan_a, 0, :], d['v_pred'][k_plan_a, 0, :]])
x0b_ = np.concatenate([d['q_pred'][k_plan_b, 0, :], d['v_pred'][k_plan_b, 0, :]])
print("xa mea  : ", x0a, " taken at k_simu_a  = ", k_simu_a)
print("xa plan : ", x0a_, " taken at k_plan_a = ", k_plan_a)
print("xb mea  : ", x0b, " taken at k_simu_b  = ", k_simu_b)
print("xb plan : ", x0b_, " taken at k_plan_b = ", k_plan_b)
id_eig = 0
lambda_a = d['Vxx_eigval'][k_plan_a, id_eig]
lambda_b = d['Vxx_eigval'][k_plan_b, id_eig]
# Check VP values
print("Eigenvals of Vxx(xa) in MPC : ", lambda_a)
print("Eigenvals of Vxx(xb) in MPC : ", lambda_b)
# Creating the DDP solver 
ddp_a = ocp_utils.init_DDP(robot, config, x0a_)
ddp_b = ocp_utils.init_DDP(robot, config, x0b_)
# solve for each point
ddp_a.setCallbacks([crocoddyl.CallbackLogger(),
                   crocoddyl.CallbackVerbose()])
ddp_b.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose()])

# Set warm start like in MPC    
    # X (xa)
xs_init_a = []
for i in range(1, d['N_h']+1):
    qi = d['q_pred'][k_plan_a-1, i, :]
    vi = d['v_pred'][k_plan_a-1, i, :]
    xi = np.concatenate([qi, vi])
    xs_init_a.append(xi)
qNp1 = d['q_pred'][k_plan_a-1, -1, :]
vNp1 = d['v_pred'][k_plan_a-1, -1, :]
xNp1 = np.concatenate([qNp1, vNp1])
xs_init_a.append(xNp1)
xs_init_a[0] = x0a_
    # U (xa)
us_init_a = []
for i in range(1, d['N_h']):
    ui = d['u_pred'][k_plan_a-1, i, :]
    us_init_a.append(ui)
uN = d['u_pred'][k_plan_a-1, -1, :]
us_init_a.append(uN)

xs_init_b = []
for i in range(1, d['N_h']+1):
    qi = d['q_pred'][k_plan_b-1, i, :]
    vi = d['v_pred'][k_plan_b-1, i, :]
    xi = np.concatenate([qi, vi])
    xs_init_b.append(xi)
qNp1 = d['q_pred'][k_plan_b-1, -1, :]
vNp1 = d['v_pred'][k_plan_b-1, -1, :]
xNp1 = np.concatenate([qNp1, vNp1])
xs_init_b.append(xNp1)
xs_init_b[0] = x0b_
    # U (xa)
us_init_b = []
for i in range(1, d['N_h']):
    ui = d['u_pred'][k_plan_b-1, i, :]
    us_init_b.append(ui)
uN = d['u_pred'][k_plan_b-1, -1, :]
us_init_b.append(uN)

# Solve
# ddp_a.solve(ddp_a.xs, ddp_a.us, maxiter=10, isFeasible=False)
# ddp_b.solve(ddp_b.xs, ddp_b.us, maxiter=10, isFeasible=False)
ddp_a.solve(xs_init_a, us_init_a)#, maxiter=10, isFeasible=False)
ddp_b.solve(xs_init_b, us_init_b)#, maxiter=10, isFeasible=False)

vals_a, vecs_a = np.linalg.eig(ddp_a.Vxx[0])
vals_b, vecs_b = np.linalg.eig(ddp_b.Vxx[0])
print("ddp_a.Vxx["+str(id_eig)+"] eig val : ", vals_a[id_eig], " (vs "+str(lambda_a)+" in data)")
print("ddp_b.Vxx["+str(id_eig)+"] eig val : ", vals_b[id_eig], " (vs "+str(lambda_b)+" in data)")
print("ddp_a.Vxx["+str(id_eig)+"] eig vec : ", vecs_a[id_eig])
print("ddp_b.Vxx["+str(id_eig)+"] eig vec : ", vecs_b[id_eig])
#  Calculate cost at selected states

#  Plot results
# plot_utils.plot_ddp_ricatti(ddp_a)
# plot_utils.plot_ddp_results([ddp_a, ddp_b], robot, id_endeff)#, which_plots=['x', 'u', 'p'])
