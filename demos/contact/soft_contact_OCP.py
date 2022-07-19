"""
@package force_feedback
@file soft_contact_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task  
"""

'''
The robot is tasked with applying a constant normal force in contact with a wall
Trajectory optimization using Crocoddyl using the DAMSoftcontact where contact force
is linear visco-elastic (spring damper model)
The goal of this script is to setup OCP (play with weights)
'''

import sys
sys.path.append('.')

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc import path_utils, pin_utils, misc_utils

from soft_mpc.ocp import OptimalControlProblemSoftContact
from soft_mpc.data import DDPDataHanlderSoftContact


def main(robot_name, PLOT, DISPLAY):


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
    oMf = robot.data.oMf[id_endeff]
    # Contact model
    oPc = oMf.translation + np.array([0.,0.,0.05    ])
    Kp = 100
    Kv = 2*np.sqrt(Kp)

    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    # Warm start and reg

    # Define initial state
    import pinocchio as pin

    pinRefFrame = pin.LOCAL
    ov = pin.getFrameVelocity(robot.model, robot.data, id_endeff, pin.WORLD).linear
    of0 = -Kp*(oMf.translation- oPc) - Kv*ov
    oRf = oMf.rotation
    lf0 = oRf.T @ of0
    fext0 = [pin.Force.Zero() for _ in range(robot.model.njoints)]
    fext0[robot.model.frames[id_endeff].parent] = robot.model.frames[id_endeff].placement.act(pin.Force(lf0, np.zeros(3)))
    # f_ext = pin_utils.get_external_joint_torques(oMf, config['frameForceRef'], robot)
    # u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), fext0, robot.model, config['armature'])
        # Setup Croco OCP and create solver
    ddp = OptimalControlProblemSoftContact(robot, config).initialize(x0,    
        id_endeff, Kp, Kv, oPc, pinRefFrame, callbacks=True)
    # Warmstart and solve
    # ug = pin_utils.get_u_grav(q0, robot.model)
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [pin_utils.get_tau(q0, v0, np.zeros(nq), fext0, robot.model, np.zeros(nq)) for i in range(config['N_h'])] 
    # xs_init = [x0 for i in range(config['N_h']+1)]
    # us_init = [u0 for i in range(config['N_h'])]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

    if(PLOT):
        #  Plot
        ddp_handler = DDPDataHanlderSoftContact(ddp)
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        # Extract soft force
        xs = np.array(ddp_data['xs'])
        ps = pin_utils.get_p_(xs[:,:nq], robot.model, id_endeff)
        vs = pin_utils.get_v_(xs[:,:nq], xs[:,nq:], robot.model, id_endeff, ref=pin.WORLD)
        # Force in WORLD aligned frame
        fs_lin = np.array([robot.data.oMf[id_endeff].rotation @ (-Kp*(ps[i,:] - oPc) - Kv*vs[i,:]) for i in range(config['N_h'])])
        fs_ang = np.zeros((config['N_h'], 3))
        ddp_data['fs'] = np.hstack([fs_lin, fs_ang])
        ddp_data['force_ref'] = [np.zeros(6) for i in range(config['N_h']) ]
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], 
                                                            colors=['r'], 
                                                            markers=['.'], 
                                                            SHOW=True)


    # Display solution in Gepetto Viewer
    if(DISPLAY):
        import crocoddyl
        display = crocoddyl.GepettoDisplay(robot, frameNames=[frame_name])
        display.displayFromSolver(ddp, factor=0.1)


if __name__=='__main__':
    args = misc_utils.parse_OCP_script(sys.argv[1:])
    main(args.robot_name, args.PLOT, args.DISPLAY)