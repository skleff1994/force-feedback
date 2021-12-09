"""
@package force_feedback
@file iiwa_contact_circle_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
The robot is tasked with exerting a constant normal force at its EE
while drawing a circle on the contact surface
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''


import sys
sys.path.append('.')

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig
np.set_printoptions(precision=4, linewidth=180)


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
config = path_utils.load_config_file('iiwa_contact_circle_OCP')
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
ee_frame_placement = robot.data.oMf[id_endeff].copy()




# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
N_h = config['N_h']
dt = config['dt']
# Setup Croco OCP and create solver
ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, 
                                            WHICH_COSTS=config['WHICH_COSTS']) 
# Setup tracking problem with circle ref EE trajectory
models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
RADIUS = config['frameCircleTrajectoryRadius'] 
OMEGA  = config['frameCircleTrajectoryVelocity']
for k,m in enumerate(models):
    # Ref
    t = min(k*config['dt'], 2*np.pi/OMEGA)
    p_ee_ref = ocp_utils.circle_point_WORLD(t, ee_frame_placement, 
                                               radius=RADIUS,
                                               omega=OMEGA)
    # Cost translation
    m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
    # Contact model 1D update z ref (WORLD frame)
    # m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref[2]
    # m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref

# Warm start state = IK of circle trajectory
WARM_START_IK = True
if(WARM_START_IK):
    logger.info("Computing warm-start using Inverse Kinematics...")
    xs_init = [] 
    us_init = []
    q_ws = q0
    for k,m in enumerate(models):
        # Get ref placement
        p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
        Mref = ee_frame_placement.copy()
        Mref.translation = p_ee_ref
        # Get corresponding forces at each joint
        f_ext = pin_utils.get_external_joint_torques(Mref, config['frameForceRef'], robot)
        # Get joint state from IK
        q_ws, v_ws, _ = pin_utils.IK_position(robot, q_ws, id_endeff, p_ee_ref, DT=1e-2, IT_MAX=100)
        xs_init.append(np.concatenate([q_ws, v_ws]))
        if(k<N_h):
            us_init.append(pin_utils.get_tau(q_ws, v_ws, np.zeros((nq,1)), f_ext, robot.model))
# Classical warm start using initial config
else:
    f_ext = pin_utils.get_external_joint_torques(ee_frame_placement, config['frameForceRef'], robot)
    u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model)
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [u0 for i in range(config['N_h'])]

# Solve initial
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)




#  Plot
PLOT = False
if(PLOT):
    ddp_data = data_utils.extract_ddp_data(ddp)
    fig, ax = plot_utils.plot_ddp_results( ddp_data, which_plots=['all'], markers=['.'], SHOW=True)





VISUALIZE = True
pause = 0.01 # in s
if(VISUALIZE):
    import time
    import pinocchio as pin

    # Init viewer
    robot.initViewer(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer; gui = viewer.gui

    draw_rate = int(N_h/50)
    log_rate  = int(N_h/10)
    
    ref_color  = [1., 0., 0., 1.]
    real_color = [0., 0., 1., 0.3]
    ct_color   = [0., 1., 0., 1.]
    
    ref_size    = 0.01
    real_size   = 0.02
    ct_size     = 0.02
    wrench_coef = 0.02

    # Display contact point as sphere + landmark
    if('contactModelFrameName' in config.keys()):

        # Placement of contact in WORLD frame = EE placement + tennis ball radius offset
        ct_frame_placement = ee_frame_placement.copy()
        offset = 0.036
        ct_frame_placement.translation = ct_frame_placement.act(np.array([0., 0., offset])) 
        tf_contact = list(pin.SE3ToXYZQUAT(ct_frame_placement))

        # Delete contact point node if already displayed
        if(gui.nodeExists('world/contact_point')):
            gui.deleteNode('world/contact_point', True)
            gui.deleteLandmark('world/contact_point')
        # Display contact point node as green sphere
        gui.addSphere('world/contact_point', ct_size, ct_color)
        gui.addLandmark('world/contact_point', .25)
        gui.applyConfiguration('world/contact_point', tf_contact)
        
        viewer.gui.refresh()

        # Display reference contact wrench as red arrow
        if('force' in config['WHICH_COSTS']):
            # Display contact force as arrow
            f_des_LOCAL = np.asarray(config['frameForceRef'])
            ct_frame_placement_aligned = ct_frame_placement.copy()
                # Because applying tf on arrow makes arrow coincide with x-axis of tf placement
                # but force is along z axis in local frame so need to transform x-->z , i.e. -90° around y
            ct_frame_placement_aligned.rotation = ct_frame_placement_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))
            tf_contact_aligned = list(pin.SE3ToXYZQUAT(ct_frame_placement_aligned))
            arrow_length = wrench_coef*np.linalg.norm(f_des_LOCAL)
            # Remove force arrow if already displayed
            if(gui.nodeExists('world/ref_wrench')):
                gui.deleteNode('world/ref_wrench', True)
            # Display force arrow
            gui.addArrow('world/ref_wrench', ref_size, arrow_length, ref_color)
            gui.applyConfiguration('world/ref_wrench', tf_contact_aligned )
            
            viewer.gui.refresh()

        # Display friction cones 
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

            viewer.gui.refresh()

    # Display reference trajectory as red spheres
    if('translation' in config['WHICH_COSTS']):

        # Remove circle ref traj and EE traj if already displayed
        for i in range(N_h):
            if(viewer.gui.nodeExists('world/EE_ref'+str(i))):
                viewer.gui.deleteNode('world/EE_ref'+str(i), True)
        
        viewer.gui.refresh()

    # Display EE trajectory as blue spheres
    for i in range(N_h):      
        if(viewer.gui.nodeExists('world/EE_'+str(i))):
            viewer.gui.deleteNode('world/EE_'+str(i), True)

    viewer.gui.refresh()
    
    logger.info("Visualizing...")

    # Clean force arrow if already displayed
    if(gui.nodeExists('world/force')):
        gui.deleteNode('world/force', True)
    # Display force arrow
    if('force' in config['WHICH_COSTS']):
        gui.addArrow('world/force', real_size, arrow_length, real_color)

    time.sleep(1.)

    for i in range(N_h):
        # Display robot in config q
        q = ddp.xs[i][:nq]
        robot.display(q)

        # Display EE traj and ref circle traj
        if(i%draw_rate==0):
            if('translation' in config['WHICH_COSTS']):
                # EE ref circle trajectory
                m_ee_ref = ee_frame_placement.copy()
                m_ee_ref.translation = models[i].differential.costs.costs['translation'].cost.residual.reference
                tf_ee_ref = list(pin.SE3ToXYZQUAT(m_ee_ref))
                viewer.gui.addSphere('world/EE_ref'+str(i), ref_size, ref_color)
                viewer.gui.applyConfiguration('world/EE_ref'+str(i), tf_ee_ref)
            # EE trajectory
            robot.framesForwardKinematics(q)
            m_ee = robot.data.oMf[id_endeff].copy()
            tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
            viewer.gui.addSphere('world/EE_'+str(i), real_size, real_color)
            viewer.gui.applyConfiguration('world/EE_'+str(i), tf_ee)
        
        # Move contact point 
        m_ct = m_ee.copy()
        m_ct.translation = m_ct.act(np.array([0., 0., offset])) 
        tf_ct = list(pin.SE3ToXYZQUAT(m_ct))
        gui.applyConfiguration('world/contact_point', tf_ct)

        # Display force (magnitude and placement)
        if('force' in config['WHICH_COSTS']):
            # Display wrench
            wrench = ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector
            gui.resizeArrow('world/force', real_size, wrench_coef*np.linalg.norm(wrench[2]))
            m_ct_aligned = m_ct.copy()
                # Because applying tf on arrow makes arrow coincide with x-axis of tf placement
                # but force is along z axis in local frame so need to transform x-->z , i.e. -90° around y
            m_ct_aligned.rotation = m_ct_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))
            gui.applyConfiguration('world/force', list(pin.SE3ToXYZQUAT(m_ct_aligned)) )
            # Move reference wrench 
            m_ct_ref_aligned = m_ee_ref.copy()
            m_ct_ref_aligned.translation = m_ct_ref_aligned.act(np.array([0., 0., offset])) 
            m_ct_ref_aligned.rotation = m_ct_ref_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))
            tf_ct_ref_aligned = list(pin.SE3ToXYZQUAT(m_ct_ref_aligned))
            gui.applyConfiguration('world/ref_wrench', tf_ct_ref_aligned)

        # Display the friction cones
        if('friction' in config['WHICH_COSTS']):
            position = m_ee
            position.rotation = m_ee.rotation
            robot.viewer.gui.applyConfiguration('world/cone', list(np.array(pin.SE3ToXYZQUAT(position)).squeeze()))
            robot.viewer.gui.setVisibility('world/cone', "ON")

        viewer.gui.refresh()

        if(i%log_rate==0):
            logger.info("Display config n°"+str(i))

        time.sleep(pause)
