
import numpy as np
import pinocchio as pin
import eigenpy

def get_p(q, pin_robot, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        robot     : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    return get_p_(q, pin_robot.model, id_endeff)

def get_p_(q, model, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    p = np.empty((N,3))
    data = model.createData()
    for i in range(N):
        pin.forwardKinematics(model, data, q[i])
        pin.updateFramePlacements(model, data)
        p[i,:] = data.oMf[id_endeff].translation.T
    return p


def get_v(q, dq, pin_robot, id_endeff):
    '''
    Returns end-effector velocities given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    return get_v_(q, dq, pin_robot.model, id_endeff)

def get_v_(q, dq, model, id_endeff):
    '''
    Returns end-effector velocities given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        model     : pinocchio model
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    v = np.empty((N,3))
    data = model.createData()
    for i in range(N):
        # Get jacobian + compute vel
        J = pin.computeFrameJacobian(model, data, q[i,:], id_endeff)
        v[i,:] = J.dot(dq[i])[:3] # expressed in LOCAL frame 
    return v


# def get_f(q, v, tau, pin_robot, id_endeff, dt=1e-2):
#     '''
#     Returns contact force in LOCAL frame based on FD estimate of joint acc
#         q         : joint positions
#         v         : joint velocities
#         a         : joint acceleration
#         tau       : joint torques
#         pin_robot : Pinocchio wrapper
#         id_endeff : id of EE frame
#         dt        : step size for FD estimate of joint acceleration
#     '''
#     return get_f_(q, v, tau, pin_robot.model, id_endeff, dt=dt)

def get_f_(q, v, tau, model, id_endeff, armature=[.1, .1, .1, .1, .1, .1, .0], REG=0.):
    '''
    Returns contact force in LOCAL frame based on FD estimate of joint acc
        q         : joint positions
        v         : joint velocities
        a         : joint acceleration
        tau       : joint torques
        pin_robot : Pinocchio wrapper
        id_endeff : id of EE frame
        dt        : step size for FD estimate of joint acceleration
    '''
    data = model.createData()
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Get spatial acceleration at EE frame
        pin.forwardKinematics(model, data, q[i,:], v[i,:], np.zeros(q.shape[1]))
        pin.updateFramePlacements(model, data)
        gamma = -pin.getFrameAcceleration(model, data, id_endeff, pin.ReferenceFrame.LOCAL)
        pin.computeJointJacobians(model, data)
        J = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
        # Joint space inertia and its inverse + NL terms
        pin.computeAllTerms(model, data, q[i,:], v[i,:])
        data.M += np.diag(armature)
        Minv = np.linalg.inv(data.M)
        h = pin.nonLinearEffects(model, data, q[i,:], v[i,:])
        # Contact force
        # f = (JMiJ')^+ ( JMi (b-tau) + gamma )
        REGMAT = REG*np.eye(6)
        LDLT = eigenpy.LDLT(J @ Minv @ J.T + REGMAT)
        f[i,:]  = LDLT.solve(J @ Minv @ (h - tau[i,:]) + gamma.vector)
        # f[i,:] = np.linalg.solve( J @ Minv @ J.T + REGMAT,  J @ Minv @ (h - tau[i,:]) + gamma.vector )
    return f

def get_f_lambda(q, v, tau, model, id_endeff, armature=[.1, .1, .1, .1, .1, .1, .0], REG=0.):
    '''
    Returns contact force in LOCAL frame based on FD estimate of joint acc
        q         : joint positions
        v         : joint velocities
        a         : joint acceleration
        tau       : joint torques
        pin_robot : Pinocchio wrapper
        id_endeff : id of EE frame
        dt        : step size for FD estimate of joint acceleration
    '''
    data = model.createData()
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Get spatial acceleration at EE frame
        pin.computeJointJacobians(model, data, q[i,:])
        pin.framesForwardKinematics(model, data, q[i,:])
        J = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
          # Forward kinematics & placements
        pin.forwardKinematics(model, data, q[i,:], v[i,:], np.zeros(q.shape[1]))
        pin.updateFramePlacements(model, data)
        gamma = pin.getFrameAcceleration(model, data, id_endeff, pin.ReferenceFrame.LOCAL)
        # Joint space inertia and its inverse + NL terms
        # pin.computeAllTerms(model, data, q[i,:], v[i,:])
        data.M += np.diag(armature)
        pin.forwardDynamics(model, data, q[i,:], v[i,:], tau[i,:], J[:6,:], gamma.vector, REG)
        # Contact force
        f[i,:] = data.lambda_c
    return f


def get_f_kkt(q, v, tau, model, id_endeff, armature=[.1, .1, .1, .1, .1, .1, .0], REG=0.):
    '''
    Returns contact force in LOCAL frame based on FD estimate of joint acc
        q         : joint positions
        v         : joint velocities
        a         : joint acceleration
        tau       : joint torques
        pin_robot : Pinocchio wrapper
        id_endeff : id of EE frame
        dt        : step size for FD estimate of joint acceleration
    '''
    data = model.createData()
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Get spatial acceleration at EE frame
        pin.computeJointJacobians(model, data, q[i,:])
        pin.framesForwardKinematics(model, data, q[i,:])
        J = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
          # Forward kinematics & placements
        pin.forwardKinematics(model, data, q[i,:], v[i,:], np.zeros(q.shape[1]))
        pin.updateFramePlacements(model, data)
        gamma = pin.getFrameAcceleration(model, data, id_endeff, pin.ReferenceFrame.LOCAL)
        # Joint space inertia and its inverse + NL terms
        h = pin.nonLinearEffects(model, data, q[i,:], v[i,:])
        rhs = np.vstack([np.array([h - tau[i,:]]).T, np.array([gamma.vector]).T ])
        f[i,:] = pin.computeKKTContactDynamicMatrixInverse(model, data, q[i,:], J).dot(rhs)[-6:,0]
    return f


def get_u_grav(q, model, armature=[.1, .1, .1, .1, .1, .1, .0]):
    '''
    Return gravity torque at q
    '''
    data = model.createData()
    data.M += np.diag(armature)
    return pin.computeGeneralizedGravity(model, data, q)


def get_tau(q, v, a, f, model, armature=[.1, .1, .1, .1, .1, .1, .0]):
    '''
    Return torque using rnea
    '''
    data = model.createData()
    data.M += np.diag(armature)
    return pin.rnea(model, data, q, v, a, f)



from numpy.linalg import pinv
import time
import matplotlib.pyplot as plt

def IK_position(robot, q, frame_id, p_des, LOGS=False, DISPLAY=False, DT=1e-2, IT_MAX=1000, EPS=1e-6, sleep=0.01):
    '''
    Inverse kinematics: returns q, v to reach desired position p
    '''
    errs =[]
    for i in range(IT_MAX):  
        if(i%10 == 0 and LOGS==True):
            print("Step "+str(i)+"/"+str(IT_MAX))
        pin.framesForwardKinematics(robot.model, robot.data, q)  
        oMtool = robot.data.oMf[frame_id]          
        oRtool = oMtool.rotation                  
        tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, frame_id)
        o_Jtool3 = oRtool.dot( tool_Jtool[:3,:] )         # 3D Jac of EE in W frame
        o_TG = oMtool.translation - p_des                 # translation err in W frame 
        vq = -pinv(o_Jtool3).dot(o_TG)                    # vel in negative err dir
        q = pin.integrate(robot.model,q, vq * DT)         # take step
        if(DISPLAY):
            robot.display(q)                                   
            time.sleep(sleep)
        errs.append(o_TG)
        if(i%10 == 0 and LOGS==True):
            print(np.linalg.norm(o_TG))
        if np.linalg.norm(o_TG) < EPS:
            break    
    return q, vq, errs

def IK_placement(robot, q0, frame_id, oMf_des, DT=1e-2, IT_MAX=1000, EPS=1e-6, DAMP=1e-6):
    '''
    Inverse kinematics: returns q, v to reach desired placement M 
    '''
    data = robot.data 
    model = robot.model
    q = q0.copy()
    vq = np.zeros(model.nq)
    pin.framesForwardKinematics(model, data, q)
    oMf = data.oMf[frame_id]
    errs = []
    # Loop on an inverse kinematics for 200 iterations.
    for i in range(IT_MAX): 
        pin.framesForwardKinematics(model, data, q)  
        dMi = oMf_des.actInv(oMf)
        err = pin.log(dMi).vector
        errs.append(err)
        if np.linalg.norm(err) < EPS:
            success = True
            break       
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeFrameJacobian(model, data, q, frame_id)                 
        vq = - J.T.dot(np.linalg.solve(J.dot(J.T) + DAMP * np.eye(6), err))
        q = pin.integrate(model, q, vq * DT)
        robot.display(q)                                   
        time.sleep(0.1)
        i += 1
    return q, vq, errs