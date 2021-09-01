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
config = path_utils.load_config_file('static_reaching_task_lpf_ocp')
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
ddp = ocp_utils.init_DDP_LPF(robot, config, 
                             y0, f_c=config['f_c'], callbacks=True, cost_w=1e-4, tau_plus=True) #1e-4

# Schedule weights for target reaching
for k,m in enumerate(ddp.problem.runningModels):
    m.differential.costs.costs['placement'].weight = ocp_utils.cost_weight_tanh(k, N_h, max_weight=100., alpha=5., alpha_cut=0.65)
    m.differential.costs.costs['stateReg'].weight = ocp_utils.cost_weight_parabolic(k, N_h, min_weight=0.01, max_weight=config['xRegWeight'])
    # print("IAM["+str(k)+"].ee = "+str(m.differential.costs.costs['placement'].weight)+
    # " | IAM["+str(k)+"].xReg = "+str(m.differential.costs.costs['stateReg'].weight))

# Solve and extract solution trajectories
xs_init = [y0 for i in range(N_h+1)]
us_init = [ug for i in range(N_h)]# ddp.problem.quasiStatic(xs_init[:-1])
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

# Plot
fig, ax = plot_utils.plot_ddp_results_LPF(ddp, robot, SHOW=False)


# Debug by passing the unfiltered torque into the LPF
tau_s = np.array(ddp.xs)[:,:nu]
w_s = np.array(ddp.us)
tau_integrated_s = np.zeros(tau_s.shape)
alpha = 1./float(1+2*np.pi*config['f_c']*dt)
tau_integrated_s[0,:] = ug 
for i in range(N_h):
    tau_integrated_s[i+1,:] = alpha*tau_integrated_s[i,:] + (1-alpha)*w_s[i,:]
for i in range(nq):
    ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h+1), tau_integrated_s[:,i], 'r-', label='Integrated')
import matplotlib.pyplot as plt
handles_x, labels_x = ax['y'][i,2].get_legend_handles_labels()
fig['y'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
plt.show()




# import time
# robot.initDisplay(loadModel=True)
# robot.display(q0)
# viewer = robot.viz.viewer
# viewer.gui.addFloor('world/floor')
# viewer.gui.refresh()

# print("Visualizing...")
# for i in range(N_h):
#     # Iter log
#     print("Display config n°"+str(i))
#     viewer.gui.refresh()
#     robot.display(ddp.xs[i][:nq])
#     time.sleep(.1)


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



