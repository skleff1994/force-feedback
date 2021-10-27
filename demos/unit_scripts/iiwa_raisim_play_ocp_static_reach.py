"""
@package force_feedback
@file iiwa_raisim_play_ocp_static_reach.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
Just to test out the Pinocchio-Rai wrapper for Iiwa robot in raisim simulator
Using simulator as a player of the OCP trajectory (sending torques)
'''


import numpy as np
import time
from utils import raisim_utils, path_utils, ocp_utils, pin_utils, plot_utils, data_utils
np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_ocp')
# Load Kuka config from URDF
urdf_path = "/home/skleff/robot_properties_kuka_RAISIM/iiwa_test.urdf"
mesh_path = "/home/skleff/robot_properties_kuka_RAISIM"
iiwa_config = raisim_utils.IiwaMinimalConfig(urdf_path, mesh_path)

# Load Raisim environment
LICENSE_PATH = '/home/skleff/.raisim/activation.raisim'
env = raisim_utils.RaiEnv(LICENSE_PATH, dt=1e-3)
robot = env.add_robot(iiwa_config, init_config=None)
env.launch_server()

# Initialize simulation
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv; nu = nq
# Update robot model with initial state
robot.reset_state(q0, v0)
robot.forward_robot(q0, v0)
print(robot.get_state())
M_ee = robot.data.oMf[id_endeff]
print("Initial placement : \n")
print(M_ee)


#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']

ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, WHICH_COSTS=config['WHICH_COSTS']) 
ug = pin_utils.get_u_grav(q0, robot)

# Solve and extract solution trajectories
xs_init = [x0 for i in range(N_h+1)]
us_init = [ug  for i in range(N_h)]
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

#  Plot
# ddp_data = data_utils.extract_ddp_data(ddp)
# fig, ax = plot_utils.plot_ddp_results(ddp_data, which_plots=['x', 'u', 'p'], markers=['.'], colors=['b'], SHOW=True)

xs = np.array(ddp.xs)
us = np.array(ddp.us)

# Initial sim state
q,v, = robot.get_state()
for i in range(N_h-1):
    print('Time step '+str(i)+'/'+str(N_h))
    robot.send_joint_command(us[i,:] - ddp.K[i] @ (np.concatenate([q,v])  - xs[i+1]))
    robot.forward_robot(q,v)
    q,v, = robot.get_state()
    env.step()
    time.sleep(0.01)

time.sleep(1000)
env.server.killServer()