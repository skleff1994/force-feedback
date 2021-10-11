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

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig
import matplotlib.pyplot as plt 
np.set_printoptions(precision=4, linewidth=180)
import time


# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
print("--------------------------------------")
print("              LOAD CONFIG             ")
print("--------------------------------------")
# Read config file
config = path_utils.load_config_file('static_reaching_task_lpf_ocp')
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
print("Initial frame translation = ", M_ee.translation.copy())

#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
ug = pin_utils.get_u_grav(q0, robot) 
y0 = np.concatenate([x0, ug])

LPF_TYPE = 1
# Approx. LPF obtained from Z.O.H. discretization on CT LPF 
if(LPF_TYPE==0):
    alpha = np.exp(-2*np.pi*config['f_c']*dt)
# Approx. LPF obtained from 1st order Euler int. on CT LPF
if(LPF_TYPE==1):
    alpha = 1./float(1+2*np.pi*config['f_c']*dt)
# Exact LPF obtained from E.M.A model (IIR)
if(LPF_TYPE==2):
    y = np.cos(2*np.pi*config['f_c']*dt)
    alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 
print("--------------------------------------")
print("              INIT OCP                ")
print("--------------------------------------")
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=True, 
                                                cost_w_reg=0., 
                                                cost_w_lim=10.,
                                                tau_plus=True, 
                                                lpf_type=LPF_TYPE,
                                                WHICH_COSTS=config['WHICH_COSTS'] ) 
# for i in range(N_h-1):
#   if(i<=int(N_h/10)):
#     ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].weight = 100
#   if(i>=5*N_h/10):
#     ddp.problem.runningModels[i].differential.costs.costs['stateReg'].weight /= 1.1


# Solve and extract solution trajectories
xs_init = [y0 for i in range(N_h+1)]
us_init = [ug for i in range(N_h)]

INIT_LOGS = True
if(INIT_LOGS):
    print("--------------------------------------")
    print("              WARM START              ")
    print("--------------------------------------")
    print("Warm start (ys, ws) with : ")
    print("  y_0 : q_0    = ", y0[:nq])
    print("        v_0    = ", y0[nq:nq+nv])
    print("        tau_0  = ", y0[nu:])
    print("  w_0 :        = ", ug)
    # print("Quasi-static torque ws = ")
    # us_qs = [ddp.problem.runningModels[0].quasiStatic(ddp.problem.runningDatas[0], y0)] * N_h
    # print("  ", us_qs[0])
    print("--------------------------------------")
    print("              DDP SOLVE               ")
    print("--------------------------------------")
    print("--------------------------------------")
    print("              ANALYSIS                ")
    print("--------------------------------------")
    print("Cumulative absolute error w.r.t. warm start : ")
    print("norm(qs-q_0)   = ", np.linalg.norm(np.array(ddp.xs)[:,:nq] - y0[:nq]))#/N_h)
    print("norm(vs-q_0)   = ", np.linalg.norm(np.array(ddp.xs)[:,nq:nx] - y0[nq:nx]))#/N_h)
    print("norm(taus-u_g) = ", np.linalg.norm(np.array(ddp.xs)[:,-nu:] - ug))#/N_h)
    print("norm(us-u_g)   = ", np.linalg.norm(np.array(ddp.us - ug)))#/N_h)

ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

VISUALIZE = False
if(VISUALIZE):
    print("--------------------------------------")
    print("              VISUALIZE               ")
    print("--------------------------------------")
    robot.initDisplay(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer
    # viewer.gui.addFloor('world/floor')
    # viewer.gui.refresh()
    log_rate = int(N_h/10)
    print("Visualizing...")
    for i in range(N_h):
        # Iter log
        viewer.gui.refresh()
        robot.display(ddp.xs[i][:nq])
        if(i%log_rate==0):
            print("Display config n°"+str(i))
        time.sleep(.05)

PLOT = True
if(PLOT):
    print("-----------------------------------")
    print("              PLOTS                ")
    print("-----------------------------------")
    #  Plot
    ddp_data = data_utils.extract_ddp_data_LPF(ddp)
    fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, which_plots=['all'], colors=['r'], markers=['.'], SHOW=False)

tau_filtered = np.zeros((N_h+1, nq))
tau_filtered[0,:] = ug
# alpha=0.9
for i in range(N_h):
    tau_filtered[i+1,:] = alpha*tau_filtered[i,:] + (1-alpha)*ddp.us[i]
for i in range(nq):
    ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h+1), tau_filtered[:,i], 'b.', alpha=0.7)
    ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h), np.array(ddp.us)[:,i], 'g', linestyle='-', marker='.', alpha=0.5, label='Control')
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



