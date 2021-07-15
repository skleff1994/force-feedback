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
# d = data_utils.load_data('/home/skleff/force-feedback/data/DATASET3_change_task_increase_freq/10000/tracking=False_10000Hz__exp_9.npz')
# Change costs as in recorded simulation (d3_exp9)
# config['frameWeight'] = 51200
# config['xRegWeight'] = 1.953125e-5
# config['uRegWeight'] = 3.90625e-5

# Load data 
d = data_utils.load_data('/home/skleff/force-feedback/data/DATASET5_change_task_increase_freq_more_noise/10000/tracking=False_10000Hz__exp_3.npz')
# Change costs as in recorded simulation (d5_exp6)
# config['frameWeight'] = 320000
# config['xRegWeight'] = 3.125e-6
# config['uRegWeight'] = 6.25e-6
# Change costs as in recorded simulation (d5_exp3)
config['frameWeight'] = 40000
config['xRegWeight'] = 2.5e-5
config['uRegWeight'] = 5e-5

plan_freq = 10e3

# Collect all the 'peak' point in data 
print("Cost weights ratio = ", str(config['frameWeight']/config['xRegWeight']))
id_eig = 1
# Collect first peak point in data , either automatically or manually
def find_peak(data, N_end=100000, eps=100):
    '''
    Returns planning step index at which Vxx drops
    '''
    for i in range(N_end):
        if(data['Vxx_eig'][i, id_eig] - data['Vxx_eig'][N_end, id_eig] <= eps):
            print("Found peak at t=", i, "( eigenval = ", data['Vxx_eig'][i, id_eig], " )")
            break
    return i

# k_plan_b = 23944 # find_peak(d) # peak 
# k_plan_a = 23943 # k_plan_b-1   # just before
# ta = k_plan_a/plan_freq
# tb = k_plan_b/plan_freq
# k_simu_a = int(simu_freq*ta)
# k_simu_b = int(simu_freq*tb)

ta = 2.3943
tb = 2.3944
k_plan_a = int(float(plan_freq)*float(ta))
k_plan_b = int(float(plan_freq)*float(tb))
# ta = k_simu_a/simu_freq
# tb = k_simu_b/simu_freq
k_simu_a = int(simu_freq*ta)
k_simu_b = int(simu_freq*tb)

x0a = np.concatenate([d['q_mea'][k_simu_a, :], d['v_mea'][k_simu_a, :]])
x0b = np.concatenate([d['q_mea'][k_simu_b, :], d['v_mea'][k_simu_b, :]])
x0a_ = np.concatenate([d['q_pred'][k_plan_a, 0, :], d['v_pred'][k_plan_a, 0, :]])
x0b_ = np.concatenate([d['q_pred'][k_plan_b, 0, :], d['v_pred'][k_plan_b, 0, :]])
print("xa mea  : ", x0a, " taken at k_simu_a = ", k_simu_a)
print("xa plan : ", x0a_, " taken at k_plan_a = ", k_plan_a)
print("xb mea  : ", x0b, " taken at k_simu_b = ", k_simu_b)
print("xb plan : ", x0b_, " taken at k_plan_b = ", k_plan_b)
# Check VP values
lambda_a = d['Vxx_eig'][k_plan_a, id_eig]
lambda_b = d['Vxx_eig'][k_plan_b, id_eig]
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
    # X (xb)
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
    # U (xb)
us_init_b = []
for i in range(1, d['N_h']):
    ui = d['u_pred'][k_plan_b-1, i, :]
    us_init_b.append(ui)
uN = d['u_pred'][k_plan_b-1, -1, :]
us_init_b.append(uN)

# Solve
ddp_a.solve(xs_init_a, us_init_a, maxiter=10, isFeasible=False)
ddp_b.solve(xs_init_b, us_init_b, maxiter=10, isFeasible=False)

print("New Vxx diag \n", ddp_a.Vxx[0].diagonal())
print("Old Vxx diag \n", d['Vxx_diag'][k_plan_a, :])

# # vals_a, vecs_a = np.linalg.eig(ddp_a.Vxx[0])
# # vals_b, vecs_b = np.linalg.eig(ddp_b.Vxx[0])
vals_a = np.sort(np.linalg.eigvals(ddp_a.Vxx[0]))[::-1]
# vals_b = np.sort(np.linalg.eigvals(ddp_b.Vxx[0]))[::-1]
# print("ddp_a.Vxx["+str(id_eig)+"] eig val : ", vals_a[id_eig], " (vs "+str(lambda_a)+" in data)")
# print("ddp_b.Vxx["+str(id_eig)+"] eig val : ", vals_b[id_eig], " (vs "+str(lambda_b)+" in data)")

print("New spectrum = \n", vals_a)
print("Old spectrum = \n", d['Vxx_eig'][k_plan_a])

# Compare state predictions
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt
tspan = np.linspace(k_plan_a*d['dt_plan'], ( k_plan_a+d['N_h'] )*d['dt_plan'] , d['N_h'] +1)

# for i in range(d['N_h']+1):
#     print("Old : ", d['q_pred'][k_plan_a, i, :], " vs. New : ", ddp_a.xs[i][:nq])
# for i in range(nq):
#     plt.plot(tspan, d['q_pred'][k_plan_a, :, i], linestyle='-', alpha=0.5)
#     plt.plot(tspan, np.array(ddp_a.xs)[:, i], linestyle='-.')
#     plt.grid(True)
# for i in range(d['N_h']+1):
#     print("Old : ", d['v_pred'][k_plan_a, i, :], " vs. New : ", ddp_a.xs[i][nv:])
# for i in range(nq):
#     plt.plot(tspan, d['v_pred'][k_plan_a, :, i], linestyle='-', alpha=0.5)
#     plt.plot(tspan, np.array(ddp_a.xs)[:, nq+i], linestyle='-.')
#     plt.grid(True)
# plt.show()

 
# print("ddp_a.Vxx["+str(id_eig)+"] eig vec : ", vecs_a[id_eig])
# print("ddp_b.Vxx["+str(id_eig)+"] eig vec : ", vecs_b[id_eig])

#  Plot results
# plot_utils.plot_ddp_ricatti(ddp_a)
# plot_utils.plot_ddp_results([ddp_a, ddp_b], robot, id_endeff)#, which_plots=['x', 'u', 'p'])
