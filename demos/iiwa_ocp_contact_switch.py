"""
@package force_feedback
@file iiwa_ocp_contact_switch.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for contact task with the KUKA iiwa 
"""

'''
The robot is tasked with approaching then contacting an object
with its EE applying a constant normal force  
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig

np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('switch_contact_task_ocp')
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
# Update robot model with initial state
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
M_ee = robot.data.oMf[id_endeff].copy()
print("EE frame placement : \n")
print(M_ee)


#################
### OCP SETUP ###
#################
N_h = config['N_h']
dt = config['dt']
# Contact frame placement 
M_ct = M_ee.copy()
M_ct.translation = M_ee.act(np.array([0., 0., .05]))
print("Contact frame placement : \n")
print(M_ct)


# Warm start and reg
import pinocchio as pin
f_ext = []
for i in range(nq+1):
    # CONTACT --> WORLD
    W_X_ct = M_ct.action
    # WORLD --> JOINT
    j_X_W  = robot.data.oMi[i].actionInverse
    # CONTACT --> JOINT
    j_X_ee = W_X_ct.dot(j_X_W)
    # ADJOINT INVERSE (wrenches)
    f_joint = j_X_ee.T.dot(np.asarray(config['f_des']))
    f_ext.append(pin.Force(f_joint))
# print(f_ext)
u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model)
ug = pin_utils.get_u_grav(q0, robot)
print("u0 = ", u0)
print("ug = ", ug)

ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True,
                                            WHICH_COSTS=config['WHICH_COSTS'],
                                            CONTACT=True,
                                            contact_placement=M_ct,
                                            u_reg_ref=u0) 


# Custom phases in the OCP 
N_switch = N_h//4
# Schedule cost weights for the task (time-based switch)
for i in range(N_h):
    if(i<=N_switch):
        # First phase = reach EE placement
        ddp.problem.runningModels[i].differential.contacts.changeContactStatus("contact", False)
        ddp.problem.runningModels[i].differential.costs.changeCostStatus("force", False)
        ddp.problem.runningModels[i].differential.costs.changeCostStatus("ctrlReg", False) # only grav first
        ddp.problem.runningModels[i].differential.costs.costs["placement"].weight = 1.
       #   ddp.problem.runningModels[i].differential.costs.costs["stateReg"].ref = 10.
    else:
        # Second phase = apply constant normal force
        ddp.problem.runningModels[i].differential.contacts.changeContactStatus("contact", True)
        ddp.problem.runningModels[i].differential.costs.changeCostStatus("force", True)
        ddp.problem.runningModels[i].differential.costs.changeCostStatus("placement", False)
        ddp.problem.runningModels[i].differential.costs.changeCostStatus("ctrlReg", True)
        ddp.problem.runningModels[i].differential.costs.changeCostStatus("ctrlRegGrav", False)
        # ddp.problem.runningModels[i].differential.costs.costs["stateReg"].weight = 10.

# Solve and extract solution trajectories
xs_init = [x0 for i in range(N_switch)] + [x0 for i in range(N_switch, N_h+1)]
us_init = [ug for i in range(N_switch)] + [u0 for i in range(N_switch, N_h)]
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

#  Plot
PLOT = True
if(PLOT):
    ddp_data = data_utils.extract_ddp_data(ddp, CONTACT=True)
    fig, ax = plot_utils.plot_ddp_results(ddp_data, which_plots=['all'], SHOW=True)
    

VISUALIZE = True
pause = 0.01 # in s
if(VISUALIZE):
    import time
    import pinocchio as pin
    robot.initViewer(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer; gui = viewer.gui
    
    # Display force if any
    if('force' in config['WHICH_COSTS']):
        # Display placement of contact in WORLD frame
        M_contact = M_ct.copy()
        offset = np.array([0., 0., 0.03])
        M_contact.translation = M_contact.act(offset)
        tf_contact = list(pin.SE3ToXYZQUAT(M_contact))
        if(gui.nodeExists('world/contact_point')):
            gui.deleteNode('world/contact_point', True)
            gui.deleteLandmark('world/contact_point')
        gui.addSphere('world/contact_point', .01, [1. ,0 ,0, 1.])
        gui.addLandmark('world/contact_point', .3)
        gui.applyConfiguration('world/contact_point', tf_contact)
        # Display contact force
        f_des_LOCAL = np.asarray(config['f_des'])
        M_contact_aligned = M_contact.copy()
            # Because applying tf on arrow makes arrow coincide with x-axis of tf placement
            # but force is along z axis in local frame so need to transform x-->z , i.e. -90° around y
        M_contact_aligned.rotation = M_contact_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))#.dot(M_contact_aligned.rotation) 
        tf_contact_aligned = list(pin.SE3ToXYZQUAT(M_contact_aligned))
        arrow_length = 0.02*np.linalg.norm(f_des_LOCAL)
        print(arrow_length)
        if(gui.nodeExists('world/ref_wrench')):
            gui.deleteNode('world/ref_wrench', True)
        gui.addArrow('world/ref_wrench', .01, arrow_length, [.5, 0., 0., 1.])
        gui.applyConfiguration('world/ref_wrench', tf_contact_aligned )
        # tf = viewer.gui.getNodeGlobalTransform('world/pinocchio/visuals/contact_0')
    # viewer.gui.addFloor('world/floor')
    
    # Display friction cones if any
    if('friction' in config['WHICH_COSTS']):
        mu = config['mu']
        frictionConeColor = [1., 1., 0., 0.3]
        m_generatrices = np.matrix(np.empty([3, 4]))
        m_generatrices[:, 0] = np.matrix([mu, mu, 1.]).T
        m_generatrices[:, 0] = m_generatrices[:, 0] / np.linalg.norm(m_generatrices[:, 0])
        m_generatrices[:, 1] = m_generatrices[:, 0]
        m_generatrices[0, 1] *= -1.
        m_generatrices[:, 2] = m_generatrices[:, 0]
        m_generatrices[:2, 2] *= -1.
        m_generatrices[:, 3] = m_generatrices[:, 0]
        m_generatrices[1, 3] *= -1.
        generatrices = m_generatrices

        v = [[0., 0., 0.]]
        for k in range(m_generatrices.shape[1]):
            v.append(m_generatrices[:3, k].T.tolist()[0])
        v.append(m_generatrices[:3, 0].T.tolist()[0])
        result = robot.viewer.gui.addCurve('world/cone', v, frictionConeColor)
        robot.viewer.gui.setCurveMode('world/cone', 'TRIANGLE_FAN')
        for k in range(m_generatrices.shape[1]):
            l = robot.viewer.gui.addLine('world/cone_ray/' + str(k), [0., 0., 0.],
                                                m_generatrices[:3, k].T.tolist()[0], frictionConeColor)
        robot.viewer.gui.setScale('world/cone', [.5, .5, .5])
        robot.viewer.gui.setVisibility('world/cone', "ALWAYS_ON_TOP")
        # robot.viewer.gui.setVisibility(lineGroup, "ALWAYS_ON_TOP")

    viewer.gui.refresh()
    log_rate = int(N_h/10)
    f = [ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(N_h)]
    print("Visualizing...")

    # Clean arrows if any
    if(gui.nodeExists('world/force')):
        gui.deleteNode('world/force', True)
    if('force' in config['WHICH_COSTS']):
        gui.addArrow('world/force', .02, arrow_length, [.0, 0., 0.5, 0.3])

    time.sleep(1.)
    for i in range(N_h):
        # Iter log
        robot.display(ddp.xs[i][:nq])

        # Display force
        if('force' in config['WHICH_COSTS']):
            gui.resizeArrow('world/force', 0.02, 0.02*np.linalg.norm(f[i]))
            gui.applyConfiguration('world/force', tf_contact_aligned )
        
        # Display the friction cones
        if('friction' in config['WHICH_COSTS']):
            position = M_contact
            position.rotation = M_contact.rotation
            robot.viewer.gui.applyConfiguration('world/cone', list(np.array(pin.SE3ToXYZQUAT(position)).squeeze()))
            robot.viewer.gui.setVisibility('world/cone', "ON")

        viewer.gui.refresh()
        # if(i%log_rate==0):
        print("Display config n°"+str(i))
        time.sleep(pause)



# # Check forces
# import pinocchio as pin
# q = np.array(ddp.xs)[:,:nq]
# v = np.array(ddp.xs)[:,nq:] 
# u = np.array(ddp.us)
# f = pin_utils.get_f_(q, v, u, robot.model, id_endeff, REG=0.)
# import matplotlib.pyplot as plt
# for i in range(3):
#     ax['f'][i,0].plot(np.linspace(0,N_h*dt, N_h), f[:,i], '-.', label="(JMiJ')+")
#     ax['f'][i,1].plot(np.linspace(0,N_h*dt, N_h), f[:,3+i], '-.', label="(JMiJ')+")
# plt.legend()
# plt.show()