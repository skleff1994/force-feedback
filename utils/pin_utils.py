
import numpy as np
import pinocchio as pin

# Post-process trajectories with pinocchio
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

# Post-process trajectories with pinocchio
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

def get_q(p_ee, model, frame_id):
    '''
    Returns configurations corresponding to p_ee end-effector
    '''
    IT_MAX = 100
    DT     = 1e-1
    data = model.createData()
    oMgoal = pin.SE3(np.eye(3), p_ee)
    q = pin.neutral(model).copy()
    herr = [] # Log the value of the error between tool and goal.
    # Loop on an inverse kinematics for 200 iterations.
    for i in range(IT_MAX):  # Integrate over 2 second of robot life
        pin.framesForwardKinematics(model, data, q)  # Compute frame placements
        oMtool = data.oMf[frame_id]                  # Placement from world frame o to frame f oMtool
        oRtool = oMtool.rotation                     # Rotation from world axes to tool axes oRtool 
        tool_Jtool = pin.computeFrameJacobian(model, data, q, frame_id)  # 6D jacobian in local frame
        o_Jtool3 = oRtool.dot(tool_Jtool[:3,:])          # 3D jacobian in world frame
        o_TG = oMtool.translation-oMgoal.translation  # vector from tool to goal, in world frame
        vq = -pinv(o_Jtool3).dot(o_TG)
        q = pin.integrate(model, q, vq * DT)
        herr.append(o_TG)
    return q,vq

    
    # oMdes = pinocchio.SE3(np.eye(3), np.array([1., 0., 1.]))
    
    # q      = pinocchio.neutral(model)
    # eps    = 1e-4
    # IT_MAX = 1000
    # DT     = 1e-1
    # damp   = 1e-12
    
    # i=0
    # while True:
    #     pinocchio.framesForwardKinematics(model,data,q)
    #     dMi = oMdes.actInv(data.oMi[frame_id])
    #     err = pinocchio.log(dMi).vector
    #     if norm(err) < eps:
    #         success = True
    #         break
    #     if i >= IT_MAX:
    #         success = False
    #         break
    #     J = pinocchio.computeFramesJacobian(model, data, q, frame_id, )
    #     v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    #     q = pinocchio.integrate(model,q,v*DT)
    #     if not i % 10:
    #         print('%d: error = %s' % (i, err.T))
    #     i += 1