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

LPF_TYPE = 0
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
                                                cost_w_reg=1e-4, 
                                                cost_w_lim=1.,
                                                tau_plus=True, 
                                                lpf_type=LPF_TYPE,
                                                WHICH_COSTS=config['WHICH_COSTS'] ) 

ddp2 = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=True, 
                                                cost_w_reg=1e-4, 
                                                cost_w_lim=1.,
                                                tau_plus=True, 
                                                lpf_type=LPF_TYPE,
                                                WHICH_COSTS=config['WHICH_COSTS'] ) 


ddp3 = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=True, 
                                                cost_w_reg=1e-4, 
                                                cost_w_lim=1.,
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
us_init = [ug for i in range(N_h)]# ddp.problem.quasiStatic(xs_init[:-1])
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
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

ddp3.solve([np.zeros(nx+nu) for i in range(N_h+1)], [np.zeros(nu) for i in range(N_h)], maxiter=config['maxiter'], isFeasible=False)

ddp2.solve([np.ones(nx+nu) for i in range(N_h+1)], [np.ones(nu) for i in range(N_h)], maxiter=config['maxiter'], isFeasible=False)
print("--------------------------------------")
print("              ANALYSIS                ")
print("--------------------------------------")
print("Cumulative absolute error w.r.t. warm start : ")
print("norm(qs-q_0)   = ", np.linalg.norm(np.array(ddp.xs)[:,:nq] - y0[:nq]))#/N_h)
print("norm(vs-q_0)   = ", np.linalg.norm(np.array(ddp.xs)[:,nq:nx] - y0[nq:nx]))#/N_h)
print("norm(taus-u_g) = ", np.linalg.norm(np.array(ddp.xs)[:,-nu:] - ug))#/N_h)
print("norm(us-u_g)   = ", np.linalg.norm(np.array(ddp.us - ug)))#/N_h)

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

DDP_DATA = [data_utils.extract_ddp_data(ddp),
            data_utils.extract_ddp_data(ddp2),
            data_utils.extract_ddp_data(ddp3)]

PLOT = True
if(PLOT):
    print("-----------------------------------")
    print("              PLOTS                ")
    print("-----------------------------------")
    #  Plot
    # ddp_data = data_utils.extract_ddp_data()
    fig, ax = plot_utils.plot_ddp_results_LPF(DDP_DATA, SHOW=True)
    # fig, ax = plot_utils.plot_ddp_results_LPF(ddp, robot, SHOW=False)
    
    # plot_utils.plot_refs_LPF(fig, ax, ddp_data, config, SHOW=True)
    # p_des = np.asarray(config['p_des'])  #M_ee.translation.copy() #
    # for i in range(3):
    #     ax['p'][i,0].plot(np.linspace(0, N_h*config['dt'], N_h+1), [p_des[i]]*(N_h+1), 'k-.', label='Desired', alpha=0.5)
    #     ax['p'][i,1].plot(np.linspace(0, N_h*config['dt'], N_h+1), [0.]*(N_h+1), 'k-.', label='Desired', alpha=0.5)
    # handles_x, labels_x = ax['p'][0,0].get_legend_handles_labels()
    # fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

    # # Debug by passing the unfiltered torque into the LPF
    # tau_s = np.array(ddp.xs)[:,:nu]
    # w_s = np.array(ddp.us)
    # tau_integrated_s = np.zeros(tau_s.shape)
    # tau_integrated_s[0,:] = ug 
    # ureg_ref = np.zeros((N_h,nu))
    # for i in range(N_h):
    #     ureg_ref[i,:] = pin_utils.get_u_grav_(np.array(ddp.xs)[i,:nq], robot.model)
    # for i in range(N_h):
    #     tau_integrated_s[i+1,:] = alpha*tau_integrated_s[i,:] + (1-alpha)*w_s[i,:]
    # for i in range(nq):
    #     # Plot a posteriori integration to check IAM
    #     ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h+1), tau_integrated_s[:,i], 'r-', label='Integrated')
    #     # Plot gravity torque
    #     ax['y'][i,2].plot(np.linspace(0, N_h*dt, N_h), ureg_ref[:,i], 'k--', label='Gravity')
    #     ax['w'][i].plot(np.linspace(0, N_h*dt, N_h), ureg_ref[:,i], 'k--', label='Gravity')
    #     ax['w'][i]
    # handles_x, labels_x = ax['y'][i,2].get_legend_handles_labels()
    # fig['y'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
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



