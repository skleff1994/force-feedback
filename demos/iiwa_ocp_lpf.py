"""
@package force_feedback
@file iiwa_ocp_lpf.py
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

np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_lpf_ocp')
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
# Update robot model with initial state
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)

#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
# u0 = np.asarray(config['tau0'])
ug = pin_utils.get_u_grav(q0, robot) 
y0 = np.concatenate([x0, ug])
print("Gravity torque = ", ug)

LPF_TYPE = 2

if(LPF_TYPE==0):
    alpha = np.exp(-2*np.pi*config['f_c']*dt)
if(LPF_TYPE==1):
    alpha = 1./float(1+2*np.pi*config['f_c']*dt)
if(LPF_TYPE==2):
    y = np.cos(2*np.pi*config['f_c']*dt)
    alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 

ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=True, 
                                                cost_w_reg=1e-12, 
                                                cost_w_lim=1.,
                                                tau_plus=True, 
                                                lpf_type=LPF_TYPE,
                                                which_costs=['stateReg', 'ctrlReg', 'placement'] ) 

# for i in range(N_h-1):
# #   if(i<=int(N_h/10)):
# #     ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].weight = 100
#   if(i>=5*N_h/10):
#     ddp.problem.runningModels[i].differential.costs.costs['stateReg'].weight /= 1.1

# Solve and extract solution trajectories
xs_init = [y0 for i in range(N_h+1)]
us_init = [ug for i in range(N_h)]# ddp.problem.quasiStatic(xs_init[:-1])
print("Warm start (ys, ws) with = \n")
print("  q0    = ", y0[:nq])
print("  v0    = ", y0[nq:nq+nv])
print("  tau_0 = ", y0[-nq:])
print("  w0    = ", ug)
print("Quasi-static torque ws =\n ")
us_qs = [ddp.problem.runningModels[0].quasiStatic(ddp.problem.runningDatas[0], y0)] * N_h
print("  ", us_qs[0])
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)





# import time
# robot.initDisplay(loadModel=True)
# robot.display(q0)
# viewer = robot.viz.viewer
# # viewer.gui.addFloor('world/floor')
# # viewer.gui.refresh()
# log_rate = int(N_h/10)
# print("Visualizing...")
# for i in range(N_h):
#     # Iter log
#     viewer.gui.refresh()
#     robot.display(ddp.xs[i][:nq])
#     if(i%log_rate==0):
#       print("Display config n°"+str(i))
#     time.sleep(.02)



#  Plot
fig, ax = plot_utils.plot_ddp_results_LPF(ddp, robot, SHOW=False)


# Debug by passing the unfiltered torque into the LPF
tau_s = np.array(ddp.xs)[:,:nu]
w_s = np.array(ddp.us)
tau_integrated_s = np.zeros(tau_s.shape)

# print()
tau_integrated_s[0,:] = ug 
for i in range(N_h):
    tau_integrated_s[i+1,:] = alpha*tau_integrated_s[i,:] + (1-alpha)*w_s[i,:]
for i in range(nq):
    # Plot a posteriori integration to check IAM
    ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h+1), tau_integrated_s[:,i], 'r-', label='Integrated')
    # Plot gravity torque
    ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h+1), ug[i]*np.ones(N_h+1), 'k--', label='Gravity')
    ax['w'][i].plot(np.linspace(0, N_h*dt, N_h), ug[i]*np.ones(N_h), 'k--', label='Gravity')
import matplotlib.pyplot as plt
handles_x, labels_x = ax['y'][i,2].get_legend_handles_labels()
fig['y'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
plt.show()



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



