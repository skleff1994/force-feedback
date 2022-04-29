'''
Debugging calc and calcDiff of DAMContactFwdDyn
# Solution 2 : Perform all calculations directly in the LWA frame 
# I.e. express everything in WORLD frame at the contact level 
# It is "cleaner" in theory , but right now doubt about pin.frameAccDerivatives in LWA 
(its a_da not equal LWA_Jc) so contact computes LOCAL then rotates 
'''


from unicodedata import name
import numpy as np
np.set_printoptions(precision=3, linewidth=180, suppress=True)

import example_robot_data 
import pinocchio as pin
import crocoddyl 


class bcolors:
    DEBUG = '\033[1m'+'\033[96m'
    ERROR = '\033[1m'+'\033[91m'
    ENDC = '\033[0m'
testcolormap = {False: bcolors.ERROR , True: bcolors.DEBUG}


ND_DISTURBANCE  = 1e-6
GAUSS_APPROX    = True
RTOL            = 1e-3 
ATOL            = 1e-3 
RANDOM_SEED     = 1
np.random.seed(RANDOM_SEED)

# Test parameters 
PIN_REFERENCE_FRAME         = pin.LOCAL     
ALIGN_LOCAL_WITH_WORLD      = False
TORQUE_SUCH_THAT_ZERO_FORCE = False
ZERO_JOINT_VELOCITY         = False

print(bcolors.DEBUG + "Reference frame = " + str(PIN_REFERENCE_FRAME) + bcolors.ENDC)


# Load robot and setup params
robot = example_robot_data.load('talos_arm')
nq = robot.model.nq; nv = robot.model.nv; nu = nq; nx = nq+nv
# q0 = np.random.rand(nq) 
q0 = np.array([.5,-1,1.5,0,0,-0.5,0])
if(ZERO_JOINT_VELOCITY): 
    print(bcolors.DEBUG + "Set zero joint velocity" + bcolors.ENDC)
    v0 = np.zeros(nq)  
else: 
    v0 = np.random.rand(nv)
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
# Optionally align LOCAL frame with WORLD frame
if(ALIGN_LOCAL_WITH_WORLD):
    print(bcolors.DEBUG + "Aligned LOCAL frame with WORLD" + bcolors.ENDC)
    # Add a custom frame aligned with WORLD to have oRf = identity
    parentFrameId = robot.model.getFrameId("gripper_left_fingertip_1_link")
    parentFrame = robot.model.frames[parentFrameId]
    W_M_j = robot.data.oMi[parentFrame.parent]
    W_M_c = pin.SE3(np.eye(3), W_M_j.act(parentFrame.placement.translation))
    # Add a frame
    customFrame = pin.Frame('contact_frame', parentFrame.parent, parentFrameId, W_M_j.actInv(W_M_c), pin.OP_FRAME)
    robot.model.addFrame(customFrame)
    contactFrameName = customFrame.name 
    # Update data
    robot.data = robot.model.createData() 
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
else:
    contactFrameName = "gripper_left_fingertip_1_link"
contactFrameId = robot.model.getFrameId(contactFrameName)
# Optionally pick torque s.t. force is zero
if(TORQUE_SUCH_THAT_ZERO_FORCE):
    print(bcolors.DEBUG + "Select tau s.t. contact force = 0" + bcolors.ENDC)
    # Compute rnea( q=q0, vq=v0, aq=J^+ * gamma, fext=0 )
    fext = [pin.Force.Zero() for i in range(robot.model.njoints)]
    pin.computeAllTerms(robot.model, robot.data, q0, v0)
    J = pin.getFrameJacobian(robot.model, robot.data, contactFrameId, pin.LOCAL)[:3,:]
    gamma = -pin.getFrameClassicalAcceleration(robot.model, robot.data, contactFrameId, pin.LOCAL)
    aq    = np.linalg.pinv(J) @ gamma.vector[:3]
    tau   = pin.rnea(robot.model, robot.data, q0, v0, aq, fext)
else:
    tau = np.random.rand(nq)
print("x0  = "+str(x0))
print("tau = "+str(tau))
print("Contact frame placement oRf : \n"+str(robot.data.oMf[contactFrameId]))

nc = 3



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

# contact calc : acceleration drift + Jacobian
def contactCalc1(model, data, frameId, x, a):
    '''
    approach 1 : everything in LOCAL
    '''
    pin.computeAllTerms(model, data, x[:nq], x[nq:])
    pin.forwardKinematics(model, data, x[:nq], x[nq:], a)
    pin.updateFramePlacement(model, data, frameId)
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL)
    a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear
    assert(np.linalg.norm(a.linear + np.cross(v.angular, v.linear) - a0) <= 1e-6 )
    return a0

# Contact calcDiff : acceleration derivatives
def contactCalcDiff1(model, data, frameId, x):
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
    assert(np.linalg.norm(a_partial_da - fJf) <= 1e-6 )
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    vv = v.linear ; vw = v.angular
    da0_dx = np.zeros((nc,nx))
    assert(np.linalg.norm(da0_dx) <= 1e-6 )
    da0_dx[:,:nv] = a_partial_dq[:3,:]
    da0_dx[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
    da0_dx[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
    da0_dx[:,nv:] = a_partial_dv[:3,:]
    da0_dx[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
    da0_dx[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
    return da0_dx

# Checking calc and calcDiff with zero acceleration (no forces)
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq))
pin.computeJointJacobians(model, data, q0)
pin.updateFramePlacements(model, data)
pin.computeRNEADerivatives(model, data, q0, v0, np.zeros(nq)) 
da0_dx_ND_1 = numdiff(lambda x_:contactCalc1(model, data, contactFrameId, x_, np.zeros(nq)), x0)
da0_dx_1 = contactCalcDiff1(model, data, contactFrameId, x0)
test_da0_dx_1 = np.allclose(da0_dx_1, da0_dx_ND_1, RTOL, ATOL)
print(testcolormap[test_da0_dx_1] + "   -- Test da0_dx numdiff (1) drift : " + str(test_da0_dx_1) + bcolors.ENDC)
# if(not test_da0_dx_1):
#     print("analytic = \n", da0_dx_1)
#     print("numdiff \n", da0_dx_ND_1)

print("\n")

# If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# since the constraint is resolved 
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
pin.computeAllTerms(model, data, q0, v0)
# Compute forward dynamics with drift obtained from contact model 
    # Drift 
a0 = contactCalc1(model, data, frameId, x0, np.zeros(nq))
fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
pin.forwardDynamics(model, data, tau, fJf[:3,:], a0)
    # get force at joint level
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(data.lambda_c, np.zeros(3)))
    # Get derivatives
pin.computeRNEADerivatives(model,data, q0, v0, data.ddq, fext)
    # Check constraint acc = 0
assert(np.linalg.norm(pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear) <= 1e-6 )
# Check numdiff
da0_dx_ND_1 = numdiff(lambda x_:contactCalc1(model, data, contactFrameId, x_, data.ddq), x0)
da0_dx_1 = contactCalcDiff1(model, data, contactFrameId, x0)
test_da0_dx_1 = np.allclose(da0_dx_1, da0_dx_ND_1, RTOL, ATOL)
print(testcolormap[test_da0_dx_1] + "   -- Test da0_dx numdiff (1) with constraint: " + str(test_da0_dx_1) + bcolors.ENDC)
# if(not test_da0_dx_1):
#     print("analytic = \n", da0_dx_1)
#     print("numdiff \n", da0_dx_ND_1)



# contact calc : acceleration drift + Jacobian
def contactCalc2(model, data, frameId, x, a, ref):
    '''
    approach 1 : everything in REF
    '''
    pin.computeAllTerms(model, data, x[:nq], x[nq:])
    pin.forwardKinematics(model, data, x[:nq], x[nq:], a)
    pin.updateFramePlacement(model, data, frameId)
    if(ref == pin.LOCAL):
        v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
        a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL)
        a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
    assert(np.linalg.norm(a.linear + np.cross(v.angular, v.linear) - a0) <= 1e-6 )
    return a0

# Contact calcDiff : acceleration derivatives
def contactCalcDiff2(model, data, frameId, x, ref):
    if(ref == pin.LOCAL):
        fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
        a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL)
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
        v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL_WORLD_ALIGNED) 
        v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
    vv = v.linear ; vw = v.angular
    da0_dx = np.zeros((nc,nx))
    assert(np.linalg.norm(a_partial_da - fJf) <= 1e-6 )
    assert(np.linalg.norm(da0_dx) <= 1e-6 )
    da0_dx[:,:nv] = a_partial_dq[:3,:]
    da0_dx[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
    da0_dx[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
    da0_dx[:,nv:] = a_partial_dv[:3,:]
    da0_dx[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
    da0_dx[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
    # assert
    return da0_dx

# Contact calcDiff : acceleration derivatives
def contactCalcDiff2Bis(model, data, frameId, x, ref):
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    vv = v.linear ; vw = v.angular
    da0_dx = np.zeros((nc,nx))
    assert(np.linalg.norm(a_partial_da - fJf) <= 1e-6 )
    assert(np.linalg.norm(da0_dx) <= 1e-6 )
    da0_dx[:,:nv] = a_partial_dq[:3,:]
    da0_dx[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
    da0_dx[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
    da0_dx[:,nv:] = a_partial_dv[:3,:]
    da0_dx[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
    da0_dx[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        da0_dx_temp = da0_dx.copy()
        R = data.oMf[frameId].rotation
        da0_dx = R @ da0_dx_temp
        a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
        Jw = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
        # print("skew term = ", pin.skew(a0)@Jw)  0 if constraint is satisfied since a=0
        da0_dx[:,:nq] -= pin.skew(a0)@Jw
    return da0_dx


print("\n")

# Checking calc and calcDiff with zero acceleration (no forces)
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq))
pin.computeJointJacobians(model, data, q0)
pin.updateFramePlacements(model, data)
pin.computeRNEADerivatives(model, data, q0, v0, np.zeros(nq)) 
da0_dx_ND_2 = numdiff(lambda x_:contactCalc2(model, data, contactFrameId, x_, np.zeros(nq), PIN_REFERENCE_FRAME), x0)
da0_dx_2    = contactCalcDiff2(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
da0_dx_2bis = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
test_da0_dx_2    = np.allclose(da0_dx_2, da0_dx_ND_2, RTOL, ATOL)
test_da0_dx_2bis = np.allclose(da0_dx_2bis, da0_dx_ND_2, RTOL, ATOL)
test_da0_dx_2crossed = np.allclose(da0_dx_2, da0_dx_2bis, RTOL, ATOL)
print(testcolormap[test_da0_dx_2] + "   -- Test da0_dx numdiff (2) drift : " + str(test_da0_dx_2) + bcolors.ENDC)
# if(not test_da0_dx_2):
#     print("analytic 2 = \n", da0_dx_2)
#     print("numdiff \n", da0_dx_ND_2)
print(testcolormap[test_da0_dx_2bis] + "   -- Test da0_dx numdiff (2bis) drift: " + str(test_da0_dx_2bis) + bcolors.ENDC)
# if(not test_da0_dx_2bis):
#     print("analytic 2bis = \n", da0_dx_2bis)
#     print("numdiff \n", da0_dx_ND_2)
print(testcolormap[test_da0_dx_2crossed] + "   -- Test da0_dx crossed (2 and 2bis) drift: " + str(test_da0_dx_2crossed) + bcolors.ENDC)
# if(not test_da0_dx_2crossed):
#     print("analytic 2 = \n", da0_dx_2)
#     print("analytic 2bis = \n", da0_dx_2bis)


print("\n")

# Contact calcDiff : acceleration derivatives
def a_partial_dx(model, data, frameId, x, ref):
    if(ref == pin.LOCAL):
        _, a_partial_dq, a_partial_dv, _ = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL)
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        _, a_partial_dq, a_partial_dv, _ = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL_WORLD_ALIGNED) 
    a_partial_dx = np.hstack([ a_partial_dq, a_partial_dv ])
    return a_partial_dx

def linear_acc(model, data, frameId, x, a, ref):
    pin.forwardKinematics(model,data,x[:nq],x[nq:nq+nv],a)
    pin.updateFramePlacements(model,data)
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        a = pin.getFrameAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear
    else:
        a = pin.getFrameAcceleration(model,data,frameId,pin.LOCAL).linear
    return a

def classical_acc(model, data, frameId, x, a, ref):
    pin.forwardKinematics(model,data,x[:nq],x[nq:nq+nv],a)
    pin.updateFramePlacements(model,data)
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        a = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear
    else:
        a = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL).linear
    return a

# Check that a_partial_dx matches the derivative of the acceleration
linear_acc_ND = numdiff(lambda x_:linear_acc(model, data, frameId, x_, data.ddq, PIN_REFERENCE_FRAME), x0)
a_partial_dx  = a_partial_dx(model, data, frameId, x0, PIN_REFERENCE_FRAME)[:3,:]
test_partial_dx    = np.allclose(linear_acc_ND, a_partial_dx, RTOL, ATOL)
print(testcolormap[test_partial_dx] + "   -- Test partial dx " + str(test_partial_dx) + bcolors.ENDC)
# print("analytic : \n", a_partial_dx)
# print("numdiff = \n", linear_acc_ND)
# print("\n")

print("\n")

# If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# since the constraint is resolved 
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
pin.computeAllTerms(model, data, q0, v0)
# Compute forward dynamics with drift obtained from contact model 
    # Drift 
a0 = contactCalc1(model, data, frameId, x0, np.zeros(nq))
fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
pin.forwardDynamics(model, data, tau, fJf[:3,:], a0)
    # get force at joint level
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(data.lambda_c, np.zeros(3)))
    # Get derivatives
pin.computeRNEADerivatives(model,data, q0, v0, data.ddq, fext)
    # Check constraint acc = 0
assert(np.linalg.norm(pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear) <= 1e-6 )
da0_dx_ND_2 = numdiff(lambda x_:contactCalc2(model, data, contactFrameId, x_, data.ddq, PIN_REFERENCE_FRAME), x0)
da0_dx_2    = contactCalcDiff2(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
da0_dx_2bis = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
test_da0_dx_2    = np.allclose(da0_dx_2, da0_dx_ND_2, RTOL, ATOL)
test_da0_dx_2bis = np.allclose(da0_dx_2bis, da0_dx_ND_2, RTOL, ATOL)
test_da0_dx_2crossed = np.allclose(da0_dx_2, da0_dx_2bis, RTOL, ATOL)
print(testcolormap[test_da0_dx_2] + "   -- Test da0_dx numdiff (2) with constraint: " + str(test_da0_dx_2) + bcolors.ENDC)
# if(not test_da0_dx_2):
#     print("analytic 2 = \n", da0_dx_2)
#     print("numdiff \n", da0_dx_ND_2)
print(testcolormap[test_da0_dx_2bis] + "   -- Test da0_dx numdiff (2bis) constraint : " + str(test_da0_dx_2bis) + bcolors.ENDC)
# if(not test_da0_dx_2bis):
#     print("analytic 2bis = \n", da0_dx_2bis)
#     print("numdiff \n", da0_dx_ND_2)
print(testcolormap[test_da0_dx_2crossed] + "   -- Test da0_dx crossed (2 and 2bis) constraint : " + str(test_da0_dx_2crossed) + bcolors.ENDC)
# if(not test_da0_dx_2crossed):
#     print("analytic 2 = \n", da0_dx_2)
#     print("analytic 2bis = \n", da0_dx_2bis)





# # Forward dynamics rewritten with forces in world coordinates.
# def fdyn(model, data, id_frame, x, tau, ref):
#     '''
#     fwdyn(x,u) = forward contact dynamics(q,v,tau) 
#     returns the concatenation of configuration acceleration and contact forces expressed in world
#     coordinates.
#     '''
#     q=x[:nq]
#     v=x[nq:]
#     pin.computeAllTerms(model,data,q,v)
#     pin.forwardKinematics(model,data,q,v,v*0)
#     pin.updateFramePlacements(model,data)
#     M = data.M
#     if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
#         J = pin.getFrameJacobian(model,data,id_frame,pin.LOCAL_WORLD_ALIGNED)[:3,:]
#         a0 = pin.getFrameClassicalAcceleration(model,data,id_frame,pin.LOCAL_WORLD_ALIGNED).linear
#     else:
#         J = pin.getFrameJacobian(model, data, id_frame, pin.LOCAL)[:3,:]
#         a0 = pin.getFrameClassicalAcceleration(model,data,id_frame,pin.LOCAL).linear
#     b = data.nle
#     K = np.block([ [M,J.T],[J,np.zeros([3,3])] ])
#     k = np.concatenate([ tau-b, -a0 ])
#     af = np.linalg.inv(K)@k
#     return af 

# # TEST CALC
# DAM.calc(DAD, x0, tau)
# DAM_ND.calc(DAD_ND, x0, tau)
# print(bcolors.DEBUG + "--- TEST CALC FUNCTION ---" + bcolors.ENDC)
# test_xout = np.allclose(DAD.xout, DAD_ND.xout, RTOL, ATOL)
# print(testcolormap[test_xout] + "   -- Test xout : " + str(test_xout) + bcolors.ENDC)

# # TEST CALCDIFF
# print(bcolors.DEBUG + "--- TEST CALCDIFF FUNCTION ---" + bcolors.ENDC)
# DAM.calcDiff(DAD, x0, tau)
# DAM_ND.calcDiff(DAD_ND, x0, tau)

# test_Fu = np.allclose(DAD.Fu, DAD_ND.Fu, RTOL, ATOL)
# print(testcolormap[test_Fu] + "   -- Test Fu : " + str(test_Fu) + bcolors.ENDC)

# test_Fx = np.allclose(DAD.Fx, DAD_ND.Fx, RTOL, ATOL)
# print(testcolormap[test_Fx] + "   -- Test Fx : " + str(test_Fx) + bcolors.ENDC)
# if(test_Fx == False):
#     test_Fq = np.allclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL)
#     print(testcolormap[test_Fq] + "           -- Fq : " + str(test_Fq) + bcolors.ENDC)
#     test_Fv = np.allclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL)
#     print(testcolormap[test_Fv] + "           -- Fv : " + str(test_Fv) + bcolors.ENDC)




# Fx_nd = numdiff(lambda x_:fdyn(robot.model, robot.data, contactFrameId, x_, tau, PIN_REFERENCE_FRAME), x0)
# Fx = np.vstack([DAD.Fx, -DAD.df_dx])
# print(Fx_nd[-3:])
# print(Fx[-3:])
# test_Fxdfdx = np.allclose(Fx, Fx_nd, RTOL, ATOL)
# print(testcolormap[test_Fxdfdx] + "   -- Test Fx and df_dx numdiff : " + str(test_Fxdfdx) + bcolors.ENDC)
