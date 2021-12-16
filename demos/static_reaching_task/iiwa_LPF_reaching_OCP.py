"""
@package force_feedback
@file iiwa_LPF_reaching_OCP.py
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

import sys
sys.path.append('.')

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig
import matplotlib.pyplot as plt 
np.set_printoptions(precision=4, linewidth=180)
import time

import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('iiwa_LPF_reaching_OCP')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper + set model to init state
robot = IiwaConfig.buildRobotWrapper()
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
# Get initial frame placement + dimensions of joint space
id_endeff = robot.model.getFrameId('contact')
M_ee = robot.data.oMf[id_endeff]
nq = robot.model.nq; nv = robot.model.nv; nx = nq+nv; nu = nq

#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
ug = pin_utils.get_u_grav(q0, robot.model) 
y0 = np.concatenate([x0, ug])
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=True, 
                                                w_reg_ref='gravity', #np.zeros(nq),#
                                                TAU_PLUS=False, 
                                                LPF_TYPE=config['LPF_TYPE'],
                                                WHICH_COSTS=config['WHICH_COSTS'] ) 
# Solve and extract solution trajectories
xs_init = [y0 for i in range(N_h+1)]
us_init = [ug for i in range(N_h)]

# ddp.reg_max = 1e-3
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False) # regInit=0.)

VISUALIZE = False
if(VISUALIZE):
    robot.initDisplay(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer
    # viewer.gui.addFloor('world/floor')
    # viewer.gui.refresh()
    log_rate = int(N_h/10)
    logger.info("Visualizing...")
    for i in range(N_h):
        # Iter log
        viewer.gui.refresh()
        robot.display(ddp.xs[i][:nq])
        if(i%log_rate==0):
            logger.info("Display config n°"+str(i))
        time.sleep(.05)

PLOT = True
if(PLOT):
    #  Plot
    ddp_data = data_utils.extract_ddp_data_LPF(ddp)
    fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, which_plots=['all'], 
                                                        colors=['r'], 
                                                        markers=['.'], 
                                                        SHOW=True)

# tau_filtered = np.zeros((N_h+1, nq))
# tau_filtered[0,:] = ug
# # alpha=0.9
# for i in range(N_h):
#     tau_filtered[i+1,:] = alpha*tau_filtered[i,:] + (1-alpha)*ddp.us[i]
# for i in range(nq):
#     ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h+1), tau_filtered[:,i], 'b.', alpha=0.7)
#     ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h), np.array(ddp.us)[:,i], 'g', linestyle='-', marker='.', alpha=0.5, label='Control')
# plt.show()

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
#     # logger.info(q, v, tau)
#     u = -Kp.dot(q) - Kd.dot(v) #- Ku.dot(tau - ug)
#     us.append(u)
#     m = ddp.problem.runningModels[0]
#     d = m.createData()
#     m.calc(d, x, u)
#     xs.append(d.xnext.copy())
# import matplotlib.pyplot as plt
# plt.grid()
# crocoddyl.plotOCSolution(xs, us)



