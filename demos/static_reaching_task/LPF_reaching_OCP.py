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

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils

np.set_printoptions(precision=4, linewidth=180)

import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(robot_name='iiwa', PLOT=True, VISUALIZE=True):


    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config = path_utils.load_config_file(robot_name+'_LPF_reaching_OCP')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper + set model to init state
    robot = pin_utils.load_robot_wrapper(robot_name)
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frame_of_interest']
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
    ug = pin_utils.get_u_grav(q0, robot.model) 
    y0 = np.concatenate([x0, ug])
    ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=True, 
                                                    w_reg_ref=np.zeros(nq),
                                                    TAU_PLUS=False, 
                                                    LPF_TYPE=config['LPF_TYPE'],
                                                    WHICH_COSTS=config['WHICH_COSTS'] ) 
    # Solve and extract solution trajectories
    xs_init = [y0 for i in range(N_h+1)]
    us_init = [ug for i in range(N_h)]

    # ddp.reg_max = 1e-3
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False) # regInit=0.)

    if(VISUALIZE):

        import time
        import pinocchio as pin

        robot.initDisplay(loadModel=True)
        robot.display(q0)
        viewer = robot.viz.viewer

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


    if(PLOT):
        #  Plot
        ddp_data = data_utils.extract_ddp_data_LPF(ddp, frame_of_interest=frame_name)
        fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, which_plots=['all'], 
                                                            colors=['r'], 
                                                            markers=['.'], 
                                                            SHOW=True)

if __name__=='__main__':
    if(len(sys.argv) < 2 or len(sys.argv) > 4):
        print("Usage: python reaching_LPF_OCP.py [arg1: robot_name (str)] [arg2: PLOT (bool)] [arg3: VISUALIZE (bool)]")
        sys.exit(0)
    elif(len(sys.argv)==2):
        sys.exit(main(str(sys.argv[1])))
    elif(len(sys.argv)==3):
        sys.exit(main(str(sys.argv[1]), bool(sys.argv[2])))
    elif(len(sys.argv)==4):
        sys.exit(main(str(sys.argv[1]), bool(sys.argv[2]), bool(sys.argv[3])))
