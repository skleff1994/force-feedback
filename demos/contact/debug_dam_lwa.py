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
RANDOM_SEED     = 10
np.random.seed(RANDOM_SEED)
CONTACT_FRAME   = pin.LOCAL

def main():

    # # # # # # # # # # # # #
    ### LOAD ROBOT MODEL  ###
    # # # # # # # # # # # # #
    robot = pin_utils.load_robot_wrapper('iiwa')
    nq = robot.model.nq; nv = robot.model.nv; nu = nq; nx = nq+nv
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
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
    # Running and terminal cost models
    runningCostModel = crocoddyl.CostModelSum(state)
    # terminalCostModel = crocoddyl.CostModelSum(state)
    # Contact model 
    contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
        # Create 3D contact on the en-effector frame
    contact_frame_id = robot.model.getFrameId("contact")
    contact_position = robot.data.oMf[contact_frame_id].translation.copy()
    baumgarte_gains  = np.array([0., 50.])
    contact3d = crocoddyl.ContactModel3D(state, contact_frame_id, contact_position, baumgarte_gains) #, CONTACT_FRAME) 
    nc = 3
        # Populate contact model with contacts
    contactModel.addContact("contact", contact3d, active=True)

    # if(WITH_COSTS):
    #     # Control regularization cost
    #     uResidual = crocoddyl.ResidualModelContactControlGrav(state)
    #     uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    #     # State regularization cost
    #     xResidual = crocoddyl.ResidualModelState(state, x0)
    #     xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    #     # End-effector frame force cost
    #     desired_wrench = np.array([0., 0., -20., 0., 0., 0.])
    #     frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), nc, actuation.nu)
    #     contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
    #     # Populate cost models with cost terms
    #     runningCostModel.addCost("stateReg", xRegCost, 1e-2)
    #     runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    #     runningCostModel.addCost("force", contactForceCost, 10.)
    #     terminalCostModel.addCost("stateReg", xRegCost, 1e-2)

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    DAM    = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True)
    DAM_ND = crocoddyl.DifferentialActionModelNumDiff(DAM, GAUSS_APPROX)
    DAD    = DAM.createData()
    DAD_ND = DAM_ND.createData()
    DAM_ND.disturbance = ND_DISTURBANCE

    # TO allocate new data (sanity check)
    robot2 = pin_utils.load_robot_wrapper('iiwa')
    robot2.framesForwardKinematics(q0)
    robot2.computeJointJacobians(q0)
    state2 = crocoddyl.StateMultibody(robot2.model)
    actuation2 = crocoddyl.ActuationModelFull(state2)
    runningCostModel2 = crocoddyl.CostModelSum(state2)
    contactModel2 = crocoddyl.ContactModelMultiple(state2, actuation2.nu)
    contact3d2 = crocoddyl.ContactModel3D(state2, contact_frame_id, contact_position, baumgarte_gains) #, CONTACT_FRAME) 
    contactModel2.addContact("contact", contact3d2, active=True)




    # calc versus ND
    DAM.calc(DAD, x0, tau)
    DAM_ND.calc(DAD_ND, x0, tau)
    logger.debug("--- TEST CALC FUNCTION ---")
    logger.debug("   -- xout (model vs numdiff) --")
    logger.info("MODEL.xout   : "+str(DAD.xout))
    logger.info("NUMDIFF.xout : "+str(DAD_ND.xout))
    logger.debug(np.allclose(DAD.xout, DAD_ND.xout, RTOL, ATOL))
    # calc versus pinocchio python
    model = robot2.model.copy()
    data  = model.createData()
    contactData = contactModel2.createData(data)
    actuationData = actuation2.createData()
    costData = runningCostModel2.createData(crocoddyl.DataCollectorAbstract())
    pin.computeAllTerms(model, data, q0, v0)
    pin.computeCentroidalMomentum(model, data)

    actuation2.calc(actuationData, x0, tau)
    contactModel2.calc(contactData, x0)
    
    logger.debug("   -- a0 (model vs python) -- ")
    logger.info("a0 :"+str(contactData.a0))
    logger.debug(np.allclose(contactData.a0, DAD.multibody.contacts.a0, RTOL, ATOL))
    # logger.info("Jc = \n"+str(contactData.Jc))
    logger.debug("   -- Jc (model vs python) --")
    logger.debug(np.allclose(contactData.Jc, DAD.multibody.contacts.Jc, RTOL, ATOL))
    logger.debug("   -- tau (model vs python) --")
    logger.info("tau = "+str(actuationData.tau))
    logger.debug(np.allclose(actuationData.tau, tau, RTOL, ATOL))
    pin.forwardDynamics(model, data, actuationData.tau, contactData.Jc, contactData.a0)
    xout = data.ddq
    logger.debug("   -- xout (model vs python) --")
    logger.info("xout = "+str(xout))
    logger.debug(np.allclose(xout, DAD.xout, RTOL, ATOL))
    logger.debug("   -- lambda_c (model vs python) --")
    logger.info("lambda_c = "+str(data.lambda_c))
    logger.debug(np.allclose(data.lambda_c, DAD.multibody.pinocchio.lambda_c, RTOL, ATOL))
    
    contactModel2.updateAcceleration(contactData, xout)
    contactModel2.updateForce(contactData, data.lambda_c)
    runningCostModel2.calc(costData, x0, tau)

    print("\n")
    # calcDiff
    logger.debug("--- TEST CALCDIFF FUNCTION ---")
        # Fu    
    DAM.calcDiff(DAD, x0, tau)
    DAM_ND.calcDiff(DAD_ND, x0, tau)
    logger.debug("   -- Test Fu (model vs numdiff) --")
    # logger.info("MODEL.Fu   :\n "+ str(DAD.Fu))
    # logger.info("NUMDIFF.Fu :\n "+ str(DAD_ND.Fu))
    logger.debug(np.allclose(DAD.Fu, DAD_ND.Fu, RTOL, ATOL))
        # Fx
    DAM.calcDiff(DAD, x0, tau)
    DAM_ND.calcDiff(DAD_ND, x0, tau)
    logger.debug("   -- Test Fx (model vs numdiff) --")
    # logger.info("MODEL.Fx   :\n "+ str(DAD.Fx))
    # logger.info("NUMDIFF.Fx :\n "+ str(DAD_ND.Fx))
    logger.debug(np.allclose(DAD.Fx, DAD_ND.Fx, RTOL, ATOL))
    
    # Calc vs pinocchio analytical 
    pin.computeRNEADerivatives(model, data, q0, v0, xout, contactData.fext)
    Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, contactData.Jc[:nc])
    logger.debug("   -- Test KKT (model vs python) --")
    # logger.info("PIN.KKTinv   :\n "+ str(Kinv))
    # logger.info("MODEL.KKTinv :\n "+ str(DAD.Kinv))
    logger.debug(np.allclose(Kinv, DAD.Kinv, RTOL, ATOL))
    
    actuation2.calcDiff(actuationData, x0, tau)
    contactModel2.calcDiff(contactData, x0) 

    logger.debug("   -- Test dtau_dq (model vs python) --")
    # logger.info("dtau_dq :\n"+str(data.dtau_dq))
    # logger.info("dtau_dq :\n"+str(DAD.multibody.pinocchio.dtau_dq))
    logger.debug(np.allclose(data.dtau_dq, DAD.multibody.pinocchio.dtau_dq, RTOL, ATOL))
    logger.debug("   -- Test dtau_dv (model vs python) --")
    # logger.info("dtau_dv :\n"+str(data.dtau_dv))
    # logger.info("dtau_dv :\n"+str(DAD.multibody.pinocchio.dtau_dv))
    logger.debug(np.allclose(data.dtau_dv, DAD.multibody.pinocchio.dtau_dv, RTOL, ATOL))

    logger.debug("   -- Test actuation.dtau_dx (model vs python) --")
    # logger.info("dtau_dx :\n"+str(actuationData.dtau_dx))
    # logger.info("dtau_dx :\n"+str(DAD.multibody.actuation.dtau_dx))
    logger.debug(np.allclose(actuationData.dtau_dx, DAD.multibody.actuation.dtau_dx, RTOL, ATOL))
    logger.debug("   -- Test actuation.dtau_du (model vs python) --")
    # logger.info("dtau_du :\n"+str(actuationData.dtau_du))
    # logger.info("dtau_du :\n"+str(DAD.multibody.actuation.dtau_du))
    logger.debug(np.allclose(actuationData.dtau_du, DAD.multibody.actuation.dtau_du, RTOL, ATOL))

    logger.debug("   -- Test contact.da0_dx (model vs python) --")
    # logger.info("da0_dx :\n"+str(contactData.da0_dx))
    # logger.info("da0_dx :\n"+str(DAD.multibody.contacts.da0_dx))
    logger.debug(np.allclose(contactData.da0_dx, DAD.multibody.contacts.da0_dx, RTOL, ATOL))

    a_partial_dtau = Kinv[:nv, :nv]
    a_partial_da   = Kinv[-nv:, -nc:]     
    f_partial_dtau = Kinv[-nc:, :nq]
    f_partial_da   = Kinv[-nc:, -nc:]
    Fx = np.zeros((nv, nx))
    Fx[:,:nq] = -a_partial_dtau @ data.dtau_dq
    Fx[:,nq:] = -a_partial_dtau @ data.dtau_dv
    Fx -= a_partial_da @ contactData.da0_dx[:nc]
    Fx += a_partial_dtau @ actuationData.dtau_dx
    Fu = a_partial_dtau @ actuationData.dtau_du

    df_dx = np.zeros((nc, nx))
    df_du = np.zeros((nc, nu))

    df_dx[:nc, :nv]  = f_partial_dtau @ data.dtau_dq
    df_dx[:nc, -nv:] = f_partial_dtau @ data.dtau_dv
    df_dx[:nc, :]   += f_partial_da @ contactData.da0_dx[:nc]
    df_dx[:nc, :]   -= f_partial_dtau @ actuationData.dtau_dx

    df_du[:nc, :] = -f_partial_dtau @ actuationData.dtau_du

    contactModel2.updateAccelerationDiff(contactData, Fx[-nv:,:])
    contactModel2.updateForceDiff(contactData, df_dx[:nc, :], df_du[:nc, :])

    logger.debug("   -- Test df_dx (model vs python) --")
    logger.info("PIN.df_dx   :\n "+ str(contactData.df_dx))
    logger.info("MODEL.df_dx :\n "+ str(DAD.multibody.contacts.df_dx))
    logger.debug(np.allclose(contactData.df_dx, DAD.multibody.contacts.df_dx, RTOL, ATOL))

    logger.debug("   -- Test df_du (model vs python) --")
    logger.info("PIN.df_du   :\n "+ str(contactData.df_du))
    logger.info("MODEL.df_du :\n "+ str(DAD.multibody.contacts.df_du))
    logger.debug(np.allclose(contactData.df_du, DAD.multibody.contacts.df_du, RTOL, ATOL))


    logger.debug("   -- Test ddv_dx (model vs python) --")
    logger.info("PIN.ddv_dx   :\n "+ str(contactData.ddv_dx))
    logger.info("MODEL.ddv_dx :\n "+ str(DAD.multibody.contacts.ddv_dx))
    logger.debug(np.allclose(contactData.ddv_dx, DAD.multibody.contacts.ddv_dx, RTOL, ATOL))

    runningCostModel2.calcDiff(costData, x0, tau)

    # print("Fx = \n")
    logger.debug("   -- Test Fu (model vs python) --")
    logger.debug(np.allclose(Fu, DAD.Fu, RTOL, ATOL))
    # logger.debug("   -- Test Fu (numdiff vs python) --")
    # logger.debug(np.allclose(Fu, DAD_ND.Fu, RTOL, ATOL))
    logger.debug("   -- Test Fx (model vs python) --")
    logger.info("PIN.Fx   :\n "+ str(Fx))
    logger.info("MODEL.Fx :\n "+ str(DAD.Fx))
    logger.debug(np.allclose(Fx, DAD.Fx, RTOL, ATOL))
    # logger.debug("\n"+str(np.isclose(Fx, DAD.Fx, RTOL, ATOL)))
    # logger.debug("   -- Test Fx (numdiff vs python) --")
    # logger.info("PIN.Fx   :\n "+ str(Fx))
    # logger.info("NUMDIFF.Fx :\n "+ str(DAD_ND.Fx))
    # logger.debug(np.allclose(Fx, DAD_ND.Fx, RTOL, ATOL))
    # logger.debug("\n"+str(np.isclose(Fx, DAD_ND.Fx, RTOL, ATOL)))


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