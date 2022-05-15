"""
@package force_feedback
@file LPF_reaching_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE position task (with Low-Pass-Filter)
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl (stateLPF x=(q,v,tau))
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import sys
sys.path.append('.')

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from utils import path_utils, pin_utils, plot_utils, misc_utils

from lpf_mpc.ocp import OptimalControlProblemLPF
from lpf_mpc.data import DDPDataParserLPF

def main(robot_name, PLOT, DISPLAY):


    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config, _ = path_utils.load_config_file(__file__, robot_name)
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper + set model to init state
    robot = pin_utils.load_robot_wrapper(robot_name)
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frameTranslationFrameName']
    id_endeff = robot.model.getFrameId(frame_name)
    M_ee = robot.data.oMf[id_endeff]
    nq = robot.model.nq; nv = robot.model.nv; nx = nq+nv; nu = nq
    # print("ID ENDEFF = ", id_endeff)

    # for elt in robot.model.frames:
    #     print('frame name = '+str(elt.name) + ' | id = ', robot.model.getFrameId(elt.name))

    #################
    ### OCP SETUP ###
    #################
    N_h = config['N_h']
    dt = config['dt']
    ug = pin_utils.get_u_grav(q0, robot.model, config['armature']) 
    y0 = np.concatenate([x0, ug])
    ddp = OptimalControlProblemLPF(robot, config).initialize(y0, callbacks=True)
    # Solve and extract solution trajectories
    xs_init = [y0 for i in range(N_h+1)]
    us_init = [ug for i in range(N_h)]

    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False) 
    
    if(PLOT):
        #  Plot
        ddp_data = DDPDataParserLPF(ddp).extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, which_plots=config['WHICH_PLOTS'], 
                                                            colors=['r'], 
                                                            markers=['.'], 
                                                            SHOW=True)


    if(DISPLAY):

        import time
        import pinocchio as pin

        N_h = config['N_h']
        models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
        # Init viewer
        robot.initViewer(loadModel=True)
        robot.display(q0)
        viewer = robot.viz.viewer; gui = viewer.gui

        draw_rate = int(N_h/50)
        log_rate  = int(N_h/10)
        
        ref_color  = [1., 0., 0., 1.]
        real_color = [0., 0., 1., 0.3]
        ct_color   = [0., 1., 0., 1.]
        
        ref_size    = 0.03
        real_size   = 0.02
        ct_size     = 0.02
        wrench_coef = 0.02

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
        
        viewer.gui.refresh()
        logger.info("Visualizing...")
        time.sleep(1.)

        for i in range(N_h):
            # Display robot in config q
            q = ddp.xs[i][:nq]
            robot.display(q)

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
    main(args.robot_name, args.PLOT, args.DISPLAY)