"""
@package force_feedback
@file debug_baumgarte_lwa.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for normal force task
"""

'''
The robot is tasked with exerting a constant normal force at its EE
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''


import sys
sys.path.append('.')

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=3, linewidth=180)

from utils import pin_utils #path_utils, ocp_utils, pin_utils, plot_utils, data_utils, misc_utils
import crocoddyl
import pinocchio as pin

WITH_COSTS      = False
ND_DISTURBANCE  = 1e-6
GAUSS_APPROX    = True
RTOL            = 1e-3
ATOL            = 1e-6
RANDOM_SEED     = 1
np.random.seed(1)
CONTACT_FRAME   = pin.LOCAL

def main():

    # # # # # # # # # # # # #
    ### LOAD ROBOT MODEL  ###
    # # # # # # # # # # # # #
    robot = pin_utils.load_robot_wrapper('iiwa')
    model = robot.model
    nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
    q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0])
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    tau = np.random.rand(nq)
    logger.info("tau random = "+str(tau))
    # # # # # # # # # # # # # # #
    ###  SETUP CROCODDYL OCP  ###
    # # # # # # # # # # # # # # #

    # State and actuation model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    # Running and terminal cost models
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    # Contact model 
    contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
        # Create 3D contact on the en-effector frame
    contact_frame_id = model.getFrameId("contact")
    contact_position = robot.data.oMf[contact_frame_id].translation.copy()
    baumgarte_gains  = np.array([0., 50.])
    contact3d = crocoddyl.ContactModel3D(state, contact_frame_id, contact_position, baumgarte_gains, CONTACT_FRAME) 
    nc = 3
        # Populate contact model with contacts
    contactModel.addContact("contact", contact3d, active=True)

    if(WITH_COSTS):
        # Control regularization cost
        uResidual = crocoddyl.ResidualModelContactControlGrav(state)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)
        # State regularization cost
        xResidual = crocoddyl.ResidualModelState(state, x0)
        xRegCost = crocoddyl.CostModelResidual(state, xResidual)
        # End-effector frame force cost
        desired_wrench = np.array([0., 0., -20., 0., 0., 0.])
        frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), nc, actuation.nu)
        contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
        # Populate cost models with cost terms
        runningCostModel.addCost("stateReg", xRegCost, 1e-2)
        runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
        runningCostModel.addCost("force", contactForceCost, 10.)
        terminalCostModel.addCost("stateReg", xRegCost, 1e-2)

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    DAM    = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True)
    DAM_ND = crocoddyl.DifferentialActionModelNumDiff(DAM, GAUSS_APPROX)
    DAD    = DAM.createData()
    DAD_ND = DAM_ND.createData()
    DAM_ND.disturbance = ND_DISTURBANCE
        
    # calc versus ND
    DAM.calc(DAD, x0, tau)
    DAM_ND.calc(DAD_ND, x0, tau)
    logger.debug("--- Test xout vs ND ---")
    logger.info("MODEL.xout   : "+str(DAD.xout))
    logger.info("NUMDIFF.xout : "+str(DAD_ND.xout))
    logger.debug(np.allclose(DAD.xout, DAD_ND.xout, RTOL, ATOL))
    # calc versus pinocchio python
    model = robot.model.copy()
    data  = model.createData()
    contactData = contactModel.createData(data)
    actuationData = actuation.createData()
    pin.computeAllTerms(model, data, q0, v0)
    contactModel.calc(contactData, x0)
    actuation.calc(actuationData, x0, tau)
    logger.debug("   -- a0 test vs pin -- ")
    logger.info("a0 :"+str(contactData.a0))
    logger.debug(np.allclose(contactData.a0, DAD.multibody.contacts.a0, RTOL, ATOL))
    # logger.info("Jc = \n"+str(contactData.Jc))
    logger.debug("   -- Jc test vs pin --")
    logger.debug(np.allclose(contactData.Jc, DAD.multibody.contacts.Jc, RTOL, ATOL))
    logger.debug("   -- tau test vs pin --")
    logger.info("tau = "+str(actuationData.tau))
    logger.debug(np.allclose(actuationData.tau, tau, RTOL, ATOL))
    pin.forwardDynamics(model, data, actuationData.tau, contactData.Jc, contactData.a0)
    xout = data.ddq
    logger.debug("   -- xout test vs pin --")
    logger.info("xout = "+str(xout))
    logger.debug(np.allclose(xout, DAD.xout, RTOL, ATOL))
    logger.debug("   -- lambda_c test vs pin --")
    logger.info("lambda_c = "+str(data.lambda_c))
    logger.debug(np.allclose(data.lambda_c, DAD.multibody.pinocchio.lambda_c, RTOL, ATOL))
    contactModel.updateForce(contactData, data.lambda_c)

    print("\n")
    # calcDiff
        # Fu    
    DAM.calcDiff(DAD, x0, tau)
    DAM_ND.calcDiff(DAD_ND, x0, tau)
    logger.debug("--- Test Fu vs ND ---")
    # logger.info("MODEL.Fu   :\n "+ str(DAD.Fu))
    # logger.info("NUMDIFF.Fu :\n "+ str(DAD_ND.Fu))
    logger.debug(np.allclose(DAD.Fu, DAD_ND.Fu, RTOL, ATOL))
        # Fx
    DAM.calcDiff(DAD, x0, tau)
    DAM_ND.calcDiff(DAD_ND, x0, tau)
    logger.debug("--- Test Fx vs ND---")
    # logger.info("MODEL.Fx   :\n "+ str(DAD.Fx))
    # logger.info("NUMDIFF.Fx :\n "+ str(DAD_ND.Fx))
    logger.debug(np.allclose(DAD.Fx, DAD_ND.Fx, RTOL, ATOL))
    # Calc vs pinocchio analytical 
    pin.computeRNEADerivatives(model, data, q0, v0, xout, contactData.fext)
    Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, contactData.Jc)
    actuation.calcDiff(actuationData, x0, tau)
    contactModel.calcDiff(contactData, x0) 
    a_partial_dtau = Kinv[:nv, :nv]
    a_partial_da   = Kinv[-nv:, -nc:]     
    # f_partial_dtau = Kinv[-nc:, :nq]
    # f_partial_da   = Kinv[-nc:, -nc:]
    Fx = np.zeros((nv, nx))
    Fx[:,:nq] = -a_partial_dtau @ data.dtau_dq
    Fx[:,nq:] = -a_partial_dtau @ data.dtau_dv
    # print(contactData.da0_dx[:nc])
    Fx -= a_partial_da @ contactData.da0_dx[:nc]
    Fx += a_partial_dtau @ actuationData.dtau_dx
    Fu = a_partial_dtau @ actuationData.dtau_du
    # print("Fx = \n")
    logger.debug("   -- Test Fu vs pin --")
    logger.debug(np.allclose(Fu, DAD.Fu, RTOL, ATOL))
    logger.debug("   -- Test Fx vs pin --")
    logger.debug(np.allclose(Fx, DAD.Fx, RTOL, ATOL))




    # pin_utils.
    # # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    # dt = 1e-2
    # runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    # terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

    # # Optionally add armature to take into account actuator's inertia
    # runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
    # terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

    # # Create the shooting problem
    # T = 250
    # problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # # Create solver + callbacks
    # ddp = crocoddyl.SolverFDDP(problem)
    # ddp.setCallbacks([crocoddyl.CallbackLogger(),
    #                 crocoddyl.CallbackVerbose()])
    # # Warm start : initial state + gravity compensation
    # xs_init = [x0 for i in range(T+1)]
    # us_init = ddp.problem.quasiStatic(xs_init[:-1])

    # # Solve
    # ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)


    # ddp_data = data_utils.extract_ddp_data(ddp, ee_frame_name=frame_name, ct_frame_name=frame_name)
    # ddp_data['xs'] = list(xs)
    # ddp_data['us'] = list(us)
    # _, _ = plot_utils.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)


if __name__=='__main__':
    main()