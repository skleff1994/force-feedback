"""
@package force_feedback
@file rotation_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for tracking EE rotation
"""

'''
The robot is tasked with rotation its EE about normal axis
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import sys
sys.path.append('.')

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils
from robot_properties_kuka.config import IiwaConfig
np.set_printoptions(precision=4, linewidth=180)
import time
import pinocchio as pin

import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



TASK = 'rotation'


def main(robot_name='iiwa', PLOT=False, VISUALIZE=True):

    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config = path_utils.load_config_file(robot_name+'_'+TASK+'_OCP')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper
    robot = pin_utils.load_robot_wrapper(robot_name)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frame_of_interest']
    id_endeff = robot.model.getFrameId(frame_name)
    nq, nv = robot.model.nq, robot.model.nv
    nx = nq+nv; nu = nq
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
    ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=True, 
                                                WHICH_COSTS=config['WHICH_COSTS']) 
    # Setup tracking problem with oritantation ref for EE trajectory
    models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    OMEGA  = config['frameRotationTrajectoryVelocity']
    for k,m in enumerate(models):
        # Ref
        t = min(k*config['dt'], 2*np.pi/OMEGA)
        # Desired RPY in WORLD frame
        R_ee_ref_WORLD = ocp_utils.rotation_orientation_WORLD(t, M_ee.copy(), 
                                                                 omega=OMEGA, 
                                                                 LOCAL_AXIS=config['ROTATION_LOCAL_AXIS'])
        # Cost translation
        m.differential.costs.costs['rotation'].cost.residual.reference = R_ee_ref_WORLD


    # Warm start state = IK of circle trajectory
    WARM_START_IK = False
    if(WARM_START_IK):
        logger.info("Computing warm-start using Inverse Kinematics...")
        xs_init = [] 
        us_init = []
        q_ws = q0
        for k,m in enumerate(models):
            Mref = M_ee.copy()
            Mref.rotation = m.differential.costs.costs['rotation'].cost.residual.reference
            q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
            xs_init.append(np.concatenate([q_ws, v_ws]))
        us_init = [pin_utils.get_u_grav(xs_init[i][:nq], robot.model) for i in range(N_h)]

    # Classical warm start using initial config
    else:
        ug  = pin_utils.get_u_grav(q0, robot.model)
        xs_init = [x0 for i in range(config['N_h']+1)]
        us_init = [ug for i in range(config['N_h'])]

    # Solve 
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

    #  Plot
    if(PLOT):
        ddp_data = data_utils.extract_ddp_data(ddp, frame_of_interest=config['frame_of_interest'])
        fig, ax = plot_utils.plot_ddp_results(ddp_data, which_plots=['all'], markers=['.'], colors=['b'], SHOW=True)


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
        if('translation' in config['WHICH_COSTS'] or 'placement' in config['WHICH_COSTS'] or 'rotation' in config['WHICH_COSTS']):

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
                    elif('rotation' in config['WHICH_COSTS']):
                        m_ee_ref.rotation = models[i].differential.costs.costs['rotation'].cost.residual.reference
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



if __name__=='__main__':
    if(len(sys.argv) < 2 or len(sys.argv) > 4):
        print("Usage: python rotation_OCP.py [arg1: robot_name (str)] [arg2: PLOT (bool)] [arg3: VISUALIZE (bool)]")
        sys.exit(0)
    elif(len(sys.argv)==2):
        sys.exit(main(str(sys.argv[1])))
    elif(len(sys.argv)==3):
        sys.exit(main(str(sys.argv[1]), bool(sys.argv[2])))
    elif(len(sys.argv)==4):
        sys.exit(main(str(sys.argv[1]), bool(sys.argv[2]), bool(sys.argv[3])))
