
import numpy as np
import pinocchio as pin

def get_p(q, pin_robot, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        robot     : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    p = np.empty((N,3))
    for i in range(N):
        pin.forwardKinematics(pin_robot.model, pin_robot.data, q[i])
        pin.updateFramePlacements(pin_robot.model, pin_robot.data)
        p[i,:] = pin_robot.data.oMf[id_endeff].translation.T
    return p

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
    N = np.shape(q)[0]
    v = np.empty((N,3))
    jac = np.zeros((6,pin_robot.model.nv))
    for i in range(N):
        # Get jacobian
        pin.computeJointJacobians(pin_robot.model, pin_robot.data, q[i,:])
        jac = pin.getFrameJacobian(pin_robot.model, pin_robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
        # Get EE velocity
        v[i,:] = jac.dot(dq[i])[:3]
    return v

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
    jac = np.zeros((6,model.nv))
    for i in range(N):
        # Get jacobian + compute vel
        pin.computeJointJacobians(model, data, q[i,:])
        jac = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
        v[i,:] = jac.dot(dq[i])[:3]
    return v


def get_f(q, v, tau, pin_robot, id_endeff, dt=1e-2):
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
    # Estimate joint accelerations with finite differences on v
    a = np.zeros(q.shape)
    for i in range(q.shape[0]):
        if i>0:
            a[i,:] = (v[i,:] - v[i-1,:])/dt
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Jacobian (in LOCAL coord)
        pin.computeJointJacobians(pin_robot.model, pin_robot.data, q[i,:])
        jac = pin.getFrameJacobian(pin_robot.model, pin_robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
        # Joint space inertia and its inverse + NL terms
        pin.crba(pin_robot.model, pin_robot.data, q[i,:])
        pin.computeMinverse(pin_robot.model, pin_robot.data, q[i,:])
        M = pin_robot.data.M
        Minv = pin_robot.data.Minv
        h = pin.nonLinearEffects(pin_robot.model, pin_robot.data, q[i,:], v[i,:])
        # Contact force
        f[i,:] = np.linalg.inv( jac.dot(Minv).dot(jac.T) ).dot( jac.dot(Minv).dot( h - tau[i,:] + M.dot(a[i,:]) ) )
    return f

def get_u_grav(q, pin_robot):
    '''
    Return gravity torque at q
    '''
    return pin.computeGeneralizedGravity(pin_robot.model, pin_robot.data, q)
    # return pin.rnea(pin_robot.model, pin_robot.data, q, np.zeros((pin_robot.model.nv,1)), np.zeros((pin_robot.model.nq,1)))

def get_u_grav_(q, model):
    '''
    Return gravity torque at q (from model, not pin)
    '''
    data = model.createData()
    return pin.computeGeneralizedGravity(model, data, q)


def get_u_mea(q, v, pin_robot):
    '''
    Return gravity torque at q
    '''
    return pin.rnea(pin_robot.model, pin_robot.data, q, v, np.zeros((pin_robot.model.nq,1)))


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