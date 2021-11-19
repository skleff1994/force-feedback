"""
@package force_feedback
@file iiwa_LPF_cirlce_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for tracking EE circle with the KUKA iiwa 
"""

'''
The robot is tasked with tracking a circle EE trajectory
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig
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
config = path_utils.load_config_file('iiwa_LPF_circle_OCP')
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
M_ee = robot.data.oMf[id_endeff]



# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
N_h = config['N_h']
dt = config['dt']
# Setup Croco OCP and create solver
ug = pin_utils.get_u_grav(q0, robot.model) 
y0 = np.concatenate([x0, ug])
LPF_TYPE = 1
ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=True, 
                                                w_reg_ref='gravity',
                                                TAU_PLUS=False, 
                                                LPF_TYPE=LPF_TYPE,
                                                WHICH_COSTS=config['WHICH_COSTS'] ) 
# Create circle trajectory (WORLD frame) and setup tracking problem
EE_ref = ocp_utils.circle_trajectory_WORLD(M_ee.copy(), dt=config['dt'], 
                                                        radius=config['frameCircleTrajectoryRadius'], 
                                                        omega=config['frameCircleTrajectoryVelocity'])
# ocp_utils.set_ee_tracking_problem(ddp, EE_ref)
# Set EE translation cost model references (i.e. setup tracking problem)
models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
for k,m in enumerate(models):
    if(k<EE_ref.shape[0]):
        ref = EE_ref[k]
    else:
        ref = EE_ref[-1]
    m.differential.costs.costs['translation'].cost.residual.reference = ref

# Warm start state = IK of circle trajectory
WARM_START_IK = True
if(WARM_START_IK):
    logger.info("Computing warm-start using Inverse Kinematics...")
    xs_init = [] 
    us_init = []
    q_ws = q0
    for k,m in enumerate(list(ddp.problem.runningModels) + [ddp.problem.terminalModel]):
        # if('placement' in m.differential.costs.costs.todict().keys()):
        #     # M_ee_ref = M_ee.copy()
        #     # M_ee_ref.translation = m.differential.costs.costs['placement'].cost.residual.reference.translation 
        #     # q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, M_ee_ref, DT=1e-1, IT_MAX=2)
        #     p_ee_ref = m.differential.costs.costs['placement'].cost.residual.reference.translation 
        #     q_ws, v_ws, eps = pin_utils.IK_position(robot, q_ws, id_endeff, p_ee_ref, DT=1e-2, IT_MAX=100)
        #     # print(q_ws, v_ws)
        if('translation' in m.differential.costs.costs.todict().keys()):
            p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
            q_ws, v_ws, eps = pin_utils.IK_position(robot, q_ws, id_endeff, p_ee_ref, DT=1e-2, IT_MAX=100)
        tau_ws = pin_utils.get_u_grav(q_ws, robot.model)
        xs_init.append(np.concatenate([q_ws, v_ws, tau_ws]))
        if(k<N_h):
            us_init.append(tau_ws)

# Classical warm start using initial config
else:
    xs_init = [y0 for i in range(config['N_h']+1)]
    us_init = [ug for i in range(config['N_h'])]

# Solve 
ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

#  Plot
PLOT = True
if(PLOT):
    ddp_data = data_utils.extract_ddp_data_LPF(ddp)
    fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, which_plots=['all'], markers=['.'], colors=['b'], SHOW=True)



VISUALIZE = True
pause = 0.02 # in s
if(VISUALIZE):
    import time
    import pinocchio as pin
    models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    # Init viewer
    robot.initViewer(loadModel=True)
    robot.display(q0)
    viewer = robot.viz.viewer; gui = viewer.gui

    draw_rate = int(N_h/50)
    log_rate  = int(N_h/10)
    
    ref_color  = [1., 0., 0., 1.]
    real_color = [0., 0., 1., 0.3]
    
    ref_size    = 0.01
    real_size   = 0.02
    wrench_coef = 0.02

    # Display reference trajectory as red spheres
    if('translation' or 'placement' in config['WHICH_COSTS']):

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

    time.sleep(1.)

    for i in range(N_h):
        # Display robot in config q
        q = ddp.xs[i][:nq]
        robot.display(q)

        # Display EE traj and ref circle traj
        if(i%draw_rate==0):
            if('translation' or 'placement' in config['WHICH_COSTS']):
                # EE ref circle trajectory
                m_ee_ref = M_ee.copy()
                if('translation' in config['WHICH_COSTS']):
                    m_ee_ref.translation = models[i].differential.costs.costs['translation'].cost.residual.reference
                elif('placement' in config['WHICH_COSTS']):
                    m_ee_ref = models[i].differential.costs.costs['placement'].cost.residual.reference.copy()
                tf_ee_ref = list(pin.SE3ToXYZQUAT(m_ee_ref))
                viewer.gui.addSphere('world/EE_ref'+str(i), ref_size, ref_color)
                viewer.gui.applyConfiguration('world/EE_ref'+str(i), tf_ee_ref)
            # EE trajectory
            robot.framesForwardKinematics(q)
            m_ee = robot.data.oMf[id_endeff].copy()
            tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
            viewer.gui.addSphere('world/EE_'+str(i), real_size, real_color)
            viewer.gui.applyConfiguration('world/EE_'+str(i), tf_ee)
        

        viewer.gui.refresh()

        if(i%log_rate==0):
            logger.info("Display config n°"+str(i))

        time.sleep(pause)
