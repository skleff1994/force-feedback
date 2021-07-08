
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
