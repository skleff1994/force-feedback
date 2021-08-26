"""
@package force_feedback
@file mpc_iiwa_ocp_LPF.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl (feedback from stateLPF x=(q,v,tau))
The goal of this script is to setup OCP (play with weights)
'''

import crocoddyl
import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils
from robot_properties_kuka.config import IiwaConfig

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_lpf')
# Create a Pybullet simulation environment + set simu freq
simu_freq = config['simu_freq']  
dt_simu = 1./simu_freq
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper
robot = IiwaConfig.buildRobotWrapper()
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq

#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
# u0 = np.asarray(config['tau0'])
ug = pin_utils.get_u_grav(q0, robot) 
y0 = np.concatenate([x0, ug])
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, f_c=config['f_c'], callbacks=True)
# Solve and extract solution trajectories
# xs_init = [y0 for i in range(N_h+1)]
us_init = [ug for i in range(N_h)] #ddp.problem.quasiStatic(xs_init[:-1])

# # Test integration (rollout)
# xs = ddp.problem.rollout(us_init)
# crocoddyl.plotOCSolution(xs, us_init)

# Integrate with joint PD to reach q,v = 0
# Kp = np.diag(np.array([1., 2., 1., 1., 1., 1., 1.]))*1.
# Kd = np.diag(np.array([1., 1., 1., 1., 1., .5, .5]))*1.5
# # Ku = np.diag(np.array([1., 1., 1., 1., 1., .5, .5]))*0.01
# xref = np.zeros(nx+nq)
# NSTEPS = 1000
# xs = [y0]
# us = []
# for i in range(NSTEPS):
#     x = xs[i]
#     q = x[:nq]
#     v = x[nq:nq+nv]
#     tau = x[nx:]
#     # print(q, v, tau)
#     u = -Kp.dot(q) - Kd.dot(v) #- Ku.dot(tau - ug)
#     us.append(u)
#     m = ddp.problem.runningModels[0]
#     d = m.createData()
#     m.calc(d, x, u)
#     xs.append(d.xnext.copy())
# import matplotlib.pyplot as plt
# plt.grid()
# crocoddyl.plotOCSolution(xs, us)


    # ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
# xs = np.array(ddp.xs) # optimal (q,v,u) traj
# us = np.array(ddp.us) # optimal   (w)   traj
# Plot
# plot_utils.plot_ddp_results_LPF(ddp, robot, id_endeff)

