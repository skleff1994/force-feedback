"""
@package force_feedback
@file reaching_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE position task 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl (state x=(q,v))
The goal of this script is to setup the OCP (a.k.a. play with weights)
'''

import sys
sys.path.append('.')

import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils, misc_utils



def main(robot_name, PLOT, VISUALIZE):

    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config, _ = path_utils.load_config_file(__file__, robot_name)
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
    # Warmstart and solve
    ug = pin_utils.get_u_grav(q0, robot.model, config['armature'])
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)


    #  Plot
    if(PLOT):
        ddp_data = data_utils.extract_ddp_data(ddp, frame_of_interest=frame_name)
        _, _ = plot_utils.plot_ddp_results(ddp_data, which_plots=['all'], markers=['.'], colors=['b'], SHOW=True)



    # Visualize motion in Gepetto-viewer
    if(VISUALIZE):
        import time
        import pinocchio as pin
        N_h = config['N_h']
        # Init viewer
        viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer()
        viz.loadViewerModel()
        viz.display(q0)
        viewer = viz.viewer; gui = viewer.gui
        draw_rate = int(N_h/50)
        log_rate  = int(N_h/10)    
        ref_color  = [1., 0., 0., 1.]
        real_color = [0., 0., 1., 0.3]
        ref_size    = 0.03
        real_size   = 0.02
        pause = 0.05
        # cleanup
            # clean ref
        if(gui.nodeExists('world/EE_ref')):
            gui.deleteNode('world/EE_ref', True)
        for i in range(N_h):
            # clean DDP sol
            if(gui.nodeExists('world/EE_sol_'+str(i))):
                gui.deleteNode('world/EE_sol_'+str(i), True)
                gui.deleteLandmark('world/EE_sol_'+str(i))
        # Get initial EE placement + tf
        ee_frame_placement = M_ee.copy()
        tf_ee = list(pin.SE3ToXYZQUAT(ee_frame_placement))
        # Get ref EE placement + tf
        m_ee_ref = ee_frame_placement.copy()
        m_ee_ref.translation = np.asarray(config['frameTranslationRef'])
        tf_ee_ref = list(pin.SE3ToXYZQUAT(m_ee_ref))
        # Display ref
        gui.addSphere('world/EE_ref', ref_size, ref_color)
        gui.applyConfiguration('world/EE_ref', tf_ee_ref)
        # Display sol init + landmark
        gui.addSphere('world/EE_sol_', real_size, real_color)
        gui.addLandmark('world/EE_sol_', 0.25)
        gui.applyConfiguration('world/EE_sol_', tf_ee)
        # Refresh and wait
        viewer.gui.refresh()
        logger.info("Visualizing...")
        time.sleep(1.)
        # Animate
        for i in range(N_h):
            # Display robot in config q
            q = ddp.xs[i][:nq]
            viz.display(q)
            # Display EE traj and ref circle traj
            if(i%draw_rate==0):
                # EE trajectory
                robot.framesForwardKinematics(q)
                m_ee = robot.data.oMf[id_endeff].copy()
                tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
                viewer.gui.addSphere('world/EE_sol_'+str(i), real_size, real_color)
                viewer.gui.applyConfiguration('world/EE_sol_'+str(i), tf_ee)
                viewer.gui.applyConfiguration('world/EE_sol_', tf_ee)
            viewer.gui.refresh()
            if(i%log_rate==0):
                logger.info("Display config n°"+str(i))
            time.sleep(pause)



if __name__=='__main__':
    args = misc_utils.parse_OCP_script(sys.argv[1:])
    main(args.robot_name, args.PLOT, args.VISUALIZE)