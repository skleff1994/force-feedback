"""
@package force_feedback
@file iiwa_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import crocoddyl
import numpy as np  
from utils import path_utils, ocp_utils, data_utils, pin_utils, plot_utils
from robot_properties_kuka.config import IiwaConfig

np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_ocp')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper
robot = IiwaConfig.buildRobotWrapper()
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
INIT_STATES = []
N_STATES = 100
np.random.seed(1)
state = crocoddyl.StateMultibody(robot.model)

# Sampling conservative range for the state : 95% q limits and [-0.5, +0.5] v limits
def get_samples(nb_samples:int):
    '''
    Samples initial states x = (q,v)within conservative state range
     95% of q limits
     [-1,+1] for v
    '''
    samples = []
    q_max = 0.85*np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    v_max = 0.1*np.ones(nv) #np.array([1.4835, 1.4835, 1.7453, 1.309 , 2.2689, 2.3562, 2.3562])  #np.zeros(nv) 
    x_max = np.concatenate([q_max, v_max])   
    for i in range(nb_samples):
        samples.append( np.random.uniform(low=-x_max, high=+x_max, size=(nx,)))
    return samples

import pinocchio as pin
def get_samples_IK(nb_samples:int, q0, p_des, v_des, id_endeff, eps_p=0.1, eps_v=0.1):
    # Sample several states 
    N_SAMPLES = nb_samples
    TSK_SPACE_SAMPLES = []
    JNT_SPACE_SAMPLES = []
    # Define bounds in cartesian space to sample (p_EE,v_EE) around (p_des,0)
    p_min = p_des - eps_p; p_max = p_des + eps_p
    v_min = v_des - eps_v; v_max = v_des + eps_v
    y_min = np.concatenate([p_min, v_min])
    y_max = np.concatenate([p_max, v_max])
    # Generate samples (uniform)
    for i in range(N_SAMPLES):
        # Sample
        y_EE = np.random.uniform(low=y_min, high=y_max, size=(6,))
        TSK_SPACE_SAMPLES.append( y_EE )
        # print(" Task sample  = ", y_EE)
        # IK
        q, _, _ = pin_utils.IK_position(robot, q0, id_endeff, y_EE[:3],
                                        DISPLAY=False, LOGS=False, DT=1e-1, IT_MAX=1000, EPS=1e-6)
        pin.computeJointJacobians(robot.model, robot.data, q)
        robot.framesForwardKinematics(q)
        J_q = pin.getFrameJacobian(robot.model, robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
        vq = np.linalg.pinv(J_q)[:,:3].dot(y_EE[3:]) 
        x = np.concatenate([q, vq])
        JNT_SPACE_SAMPLES.append( x )
    return JNT_SPACE_SAMPLES


# INIT_STATES = get_samples(N_STATES)
p_des = np.asarray(config['p_des'])
v_des = np.zeros(3)    
INIT_STATES = get_samples_IK(N_STATES, q0, p_des, v_des, id_endeff, eps_p=0.05, eps_v=0.01)

#################
### OCP SETUP ###
#################


# Horizons to be tested
COSTS = []
REJECTED_DATA = []
DDP_DATA = []
N_h = config['N_h']
dt = config['dt']
rejected = 0
sample_number = 0
TOL_SOLVER_REG = 1e-1
for x0 in INIT_STATES:
    print("Sample ", sample_number, " / ", N_STATES)
    q0 = x0[:nq]
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)

    # Create solver with custom horizon
    ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=False, 
                                            which_costs=['translation', 
                                                         'ctrlReg', 
                                                         'stateReg',
                                                         'stateLim'],
                                            dt = None, N_h=None) 
    # Warm-start
    ug = pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    # Solve
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    if(ddp.x_reg >= TOL_SOLVER_REG or ddp.u_reg >= TOL_SOLVER_REG ):
        rejected+=1
        ddp_data = data_utils.extract_ddp_data(ddp)
        REJECTED_DATA.append(ddp_data)
        print("  !!! REJECT !!! COST = ", ddp.cost, " | [ X_REG = ", ddp.x_reg, " ] [ U_REG = ", ddp.u_reg, " ]" )
    else:
        print("  ACCEPT: COST = ", ddp.cost, " | [ X_REG = ", ddp.x_reg, " ] [ U_REG = ", ddp.u_reg, " ]" )
        COSTS.append(ddp.cost)
        ddp_data = data_utils.extract_ddp_data(ddp)
        DDP_DATA.append(ddp_data)
    sample_number+=1

print("Rejected : ", rejected)
# Plot results
val_max = np.max(COSTS)
index_max = COSTS.index(val_max)
print("MAX COST = ", val_max, " at index ", index_max)

fig, ax = plot_utils.plot_ddp_results(DDP_DATA, which_plots=['x','u','p'], SHOW=True, sampling_plot=2)
if(rejected!=0):
    fig, ax = plot_utils.plot_ddp_results(REJECTED_DATA, which_plots=['x','u','p'], SHOW=True, sampling_plot=10)