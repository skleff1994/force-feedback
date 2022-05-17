import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import example_robot_data 
import pinocchio as pin

# Load robot and setup params
robot = example_robot_data.load('talos_arm')
nq = robot.model.nq; nv = robot.model.nv; nu = nq; nx = nq+nv
# q0 = np.random.rand(nq) 
q0 = np.array([.5,-1,1.5,0,0,-0.5,0])
v0 = np.random.rand(nv)
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
tau = np.random.rand(nq)
print("x0  = "+str(x0))
print("tau = "+str(tau))


# Numerical difference function
def numdiff(f,x0,h=1e-6):
    f0 = f(x0).copy()
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = x0[ix]
    return np.array(Fx).T

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def fdyn_local(model, data, frameId, x, tau, Kp, Kv, lP):
    '''
    compute joint acc using LOCAL Force
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oRf = data.oMf[frameId].rotation
    lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    pdot = lJ[:3] @ v
    lv = pin.getFrameVelocity(model, data, frameId, pin.LOCAL).linear
    assert(np.linalg.norm(lv - pdot) < 1e-4)
    force = -Kp * oRf.T @ ( data.oMf[frameId].translation - lP ) - Kv*lv
    
    force2 = force_local(model, data, frameId, x, Kp, Kv, lP)
    assert(np.linalg.norm(force2 - force) < 1e-4)
    
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(force, np.zeros(3)))
    aq = pin.aba(model, data, q, v, tau, fext)
    # print("acc = \n")
    # print(aq)
    return aq

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def fdyn_world(model, data, frameId, x, tau, Kp, Kv, oP):
    '''
    compute joint acc using WORLD force
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oRf = data.oMf[frameId].rotation
    oJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
    pdot = oJ[:3] @ v
    ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
    assert(np.linalg.norm(ov - pdot) < 1e-4)
    force = -Kp * ( data.oMf[frameId].translation - oP ) - Kv*ov

    force2 = force_world(model, data, frameId, x, Kp, Kv, oP)
    assert(np.linalg.norm(force2 - force) < 1e-4)

    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(oRf.T @ force, np.zeros(3)))
    aq = pin.aba(model, data, q, v, tau, fext)
    # print("acc = \n")
    # print(aq)
    return aq

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def force_local(model, data, frameId, x, Kp, Kv, lP):
    '''
    compute the contact force in LOCAL
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oRf = data.oMf[frameId].rotation
    lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    pdot = lJ[:3] @ v
    lv = pin.getFrameVelocity(model, data, frameId, pin.LOCAL).linear
    assert(np.linalg.norm(lv - pdot) < 1e-4)
    ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
    assert(np.linalg.norm(ov - oRf @ lv) < 1e-4)
    force = -Kp * oRf.T @ (data.oMf[frameId].translation - lP) - Kv*lv
    return force

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def force_world(model, data, frameId, x, Kp, Kv, oP):
    '''
    compute the contact force in WORLD
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
    pdot = oJ[:3] @ v
    ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
    assert(np.linalg.norm(ov - pdot) < 1e-4)
    force = -Kp*(data.oMf[frameId].translation - oP) - Kv*ov
    return force




# TESTS 
contactFrameName = "gripper_left_fingertip_1_link"
model = robot.model
data = robot.model.createData()
frameId = model.getFrameId(contactFrameName)
Kp = 1
Kv = 2*np.sqrt(Kp)
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
oRf = data.oMf[frameId].rotation
# anchor point (local and world)
oPc = np.zeros(3)
lPc = oRf.T @ np.zeros(3)

# Compute visco-elastic contact force 
lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
oJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
lv = pin.getFrameVelocity(model, data, frameId, pin.LOCAL).linear
ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
assert(np.linalg.norm(oJ[:3] - oRf @ lJ[:3]) < 1e-4)
assert(np.linalg.norm(lv - lJ[:3] @ v0) < 1e-4)
assert(np.linalg.norm(ov - oJ[:3] @ v0) < 1e-4)
assert(np.linalg.norm(ov - oRf @ lv) < 1e-4)

# LOCAL force and joint acc
lf = -Kp* oRf.T @ ( data.oMf[frameId].translation - lPc) - Kv*lv
lfext = [pin.Force.Zero() for _ in range(model.njoints)]
lfext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(lf, np.zeros(3)))
laq = pin.aba(model, data, q0, v0, tau, lfext)
assert(np.linalg.norm(laq - fdyn_local(model, data, frameId, x0, tau, Kp, Kv, lPc)) <1e-4)
# WORLD force and joint acc
of = -Kp* ( data.oMf[frameId].translation - oPc) - Kv*ov
assert(np.linalg.norm(of - oRf @ lf) < 1e-4)
ofext = [pin.Force.Zero() for _ in range(model.njoints)]
ofext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(oRf.T @ of, np.zeros(3)))
oaq = pin.aba(model, data, q0, v0, tau, ofext)
assert(np.linalg.norm(oaq - fdyn_world(model, data, frameId, x0, tau, Kp, Kv, oPc)) <1e-4)
assert(np.linalg.norm(oaq - laq) < 1e-4)
# check force at joint level are the same
jfext_lin = lfext[model.frames[frameId].parent].linear
jfext_ang = lfext[model.frames[frameId].parent].angular
ofext_lin = ofext[model.frames[frameId].parent].linear
ofext_ang = ofext[model.frames[frameId].parent].angular
assert(np.linalg.norm(jfext_lin - ofext_lin) < 1e-4)
assert(np.linalg.norm(jfext_ang - ofext_ang) < 1e-4)
# Check derivatives of the force against numdiff
    # ND
dlf_dx_ND = numdiff(lambda x_:force_local(model, data, frameId, x_, Kp, Kv, lPc), x0)
dof_dx_ND = numdiff(lambda x_:force_world(model, data, frameId, x_, Kp, Kv, oPc), x0)
    # Analytic
        # local 
dlf_dx = np.zeros((3,nx))
lv_partial_dq, lv_partial_dv = pin.getFrameVelocityDerivatives(model, data, frameId, pin.LOCAL) 
dlf_dx[:,:nq] = -Kp * (lJ[:3] + pin.skew(oRf.T @ (data.oMf[frameId].translation - oPc)) @ lJ[3:]) - Kv*lv_partial_dq[:3]
dlf_dx[:,nq:] = -Kv*lv_partial_dv[:3]
assert(np.linalg.norm(dlf_dx - dlf_dx_ND) < 1e-4)
#         # world ### FAILS because of frameVelDerivatives in LWA ?
# dof_dx = np.zeros((3,nx))
# ov_partial_dq, ov_partial_dv = pin.getFrameVelocityDerivatives(model, data, frameId, pin.LOCAL_WORLD_ALIGNED) 
# dof_dx[:,:nq] = -Kp * oJ[:3] - Kv*ov_partial_dq[:3]
# dof_dx[:,nq:] = -Kv*ov_partial_dv[:3]
# assert(np.linalg.norm(dof_dx - dof_dx_ND) < 1e-4)       
        # world 2 (rotate local)
dof_dx = np.zeros((3,nx))
dof_dx[:,:nq] = oRf @ dlf_dx[:,:nq] - pin.skew(oRf @ lf) @ oJ[3:]
dof_dx[:,nq:] = oRf @ dlf_dx[:,nq:]
assert(np.linalg.norm(dof_dx - dof_dx_ND) < 1e-4)
# Compute the derivative of joint acceleration
    # local 
laba_dq, lada_dv, laba_dtau = pin.computeABADerivatives(model, data, q0, v0, tau, lfext)
ldaq_dx = np.zeros((nq, nx))
ldaq_du = np.zeros((nq, nq))
ldaq_dx[:,:nq] = laba_dq + data.Minv @ lJ[:3].T @ dlf_dx[:,:nq]
ldaq_dx[:,nq:] = lada_dv + data.Minv @ lJ[:3].T @ dlf_dx[:,nq:]
ldaq_du = laba_dtau
    # compare numdiff
ldaq_dx_ND = numdiff(lambda x_:fdyn_local(model, data, frameId, x_, tau, Kp, Kv, lPc), x0)
print("analytic = \n")
print(ldaq_dx)
print("nd = \n")
print(ldaq_dx_ND)
assert(np.linalg.norm(ldaq_dx - ldaq_dx_ND) < 1e-3)
    # world
oaba_dq, oada_dv, oaba_dtau = pin.computeABADerivatives(model, data, q0, v0, tau, ofext)
odaq_dx = np.zeros((nq, nx))
odaq_du = np.zeros((nq, nu))
odaq_dx[:,:nq] = oaba_dq + data.Minv @ oJ[:3].T @ dof_dx[:,:nq]
odaq_dx[:,nq:] = oaba_dq + data.Minv @ oJ[:3].T @ dof_dx[:,nq:]
odaq_du = oaba_dtau
    # compare numdiff
odaq_dx_ND = numdiff(lambda x_:fdyn_world(model, data, frameId, x_, tau, Kp, Kv, oPc), x0)
print("analytic = \n")
print(odaq_dx)
print("nd = \n")
print(odaq_dx)
assert(np.linalg.norm(odaq_dx - odaq_dx) < 1e-3)
