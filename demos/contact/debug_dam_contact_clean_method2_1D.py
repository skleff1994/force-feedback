'''
Debugging calc and calcDiff of DAMContactFwdDyn
# Solution 2 : Perform all calculations directly in the LWA frame 
# I.e. express everything in WORLD frame at the contact level 
# It is "cleaner" in theory , but right now doubt about pin.frameAccDerivatives in LWA 
(its a_da not equal LWA_Jc) so contact computes LOCAL then rotates 
FOR contact 1D
'''


from lib2to3.pgen2.token import AT
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
MASK                        = 2 # along z     
# d->a0[0] = d->a.linear()[mask_] + d->vw[(mask_+1)%3] * d->vv[(mask_+2)%3] - d->vw[(mask_+2)%3] * d->vv[(mask_+1)%3];
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
# print("Contact frame placement oRf : \n"+str(robot.data.oMf[contactFrameId]))

nc = 1

CT_REF = np.zeros(3)
GAINS  = [0., 0.]


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
def contactCalc2bis(model, data, frameId, x, aq, ref):
    '''
    approach 1 : everything in REF
    '''
    pin.computeAllTerms(model, data, x[:nq], x[nq:])
    pin.forwardKinematics(model, data, x[:nq], x[nq:], aq)
    pin.updateFramePlacement(model, data, frameId)
    R = data.oMf[frameId].rotation
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL)
    # print(a.vector)
    a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear
    la0_no_bg = a0.copy()
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    assert(np.linalg.norm(a.linear + np.cross(v.angular, v.linear) - a0) <= 1e-6 )
    # Baumgarte term
    if(GAINS[0] != 0.):
        a0 += GAINS[0] * R.T @ (data.oMf[frameId].translation - CT_REF)
    if(GAINS[1] != 0.):
        a0 += GAINS[1] * v.linear
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        a0_tmp = a0.copy()
        v_world = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        a_world = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        a0_world = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
        assert(np.linalg.norm(a_world.linear + np.cross(v_world.angular, v_world.linear) - a0_world) <= 1e-6 )
        assert(np.linalg.norm(R @ la0_no_bg - a0_world) <= 1e-6 )
        wJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        assert(np.linalg.norm(R @ fJf[:3,:] - wJf[:3,:]) <= 1e-6 ) 
        a0 = R @ a0_tmp
    return a0

# Contact calcDiff : acceleration derivatives
def contactCalcDiff2Bis(model, data, frameId, x, ref):
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    vv = v.linear ; vw = v.angular
    da0_dx_3d = np.zeros((3,nx))
    assert(np.linalg.norm(a_partial_da - fJf) <= 1e-6 )
    assert(np.linalg.norm(da0_dx_3d) <= 1e-6 )
    da0_dx_3d[:,:nv] = a_partial_dq[:3,:]
    da0_dx_3d[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
    da0_dx_3d[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
    da0_dx_3d[:,nv:] = a_partial_dv[:3,:]
    da0_dx_3d[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
    da0_dx_3d[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
    da0_dx = da0_dx_3d[MASK]
    R = data.oMf[frameId].rotation
    # Add Baumgarte term in LOCAL
    if (GAINS[0] != 0.):
        tmp_skew = pin.skew( R.T @ (data.oMf[frameId].translation - CT_REF) )
        da0_dx[:,:nv] += GAINS[0] * ( tmp_skew @ fJf[3:,:] + fJf[:3,:] )
    if (GAINS[1] != 0.):
        da0_dx[:,:nv] += GAINS[1] * v_partial_dq[:3,:]
        da0_dx[:,nv:] += GAINS[1] * fJf[:3,:]
    # Then express in WORLD
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
        # print("contactCalcDiff2Bis.a0 = ", a0)
        da0_dx_temp = da0_dx.copy()     
        da0_dx = R @ da0_dx_temp
        Jw = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
        da0_dx[:,:nq] -= pin.skew(a0)@Jw
    return da0_dx


print("\n")

# # Checking calc and calcDiff with zero acceleration (no forces)
# frameId = contactFrameId
# model = robot.model
# data = robot.model.createData()
# pin.computeAllTerms(model, data, q0, v0)
# pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
# pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq))
# pin.computeJointJacobians(model, data, q0)
# pin.updateFramePlacements(model, data)
# pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq)) 
# pin.computeRNEADerivatives(model, data, q0, v0, np.zeros(nq)) 
# da0_dx_ND_2bis = numdiff(lambda x_:contactCalc2bis(model, data, contactFrameId, x_, np.zeros(nq), PIN_REFERENCE_FRAME), x0)
# da0_dx_2bis = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
# test_da0_dx_2bis = np.allclose(da0_dx_2bis, da0_dx_ND_2bis, RTOL, ATOL)
# print(testcolormap[test_da0_dx_2bis] + "   -- Test da0_dx ND : " + str(test_da0_dx_2bis) + bcolors.ENDC)


# # If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# # and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# # since the constraint is resolved 
# frameId = contactFrameId
# model = robot.model
# data = robot.model.createData()
# pin.computeAllTerms(model, data, q0, v0)
# R = data.oMf[frameId].rotation
# # Compute forward dynamics with drift obtained from contact model 
#     # Drift 
# a0 = contactCalc2bis(model, data, frameId, x0, np.zeros(nq), PIN_REFERENCE_FRAME)
# if(PIN_REFERENCE_FRAME == pin.LOCAL):
#     fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
# else:
#     fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
# pin.forwardDynamics(model, data, tau, fJf[:3,:], a0)
# if(PIN_REFERENCE_FRAME == pin.LOCAL):
#     f = data.lambda_c
# else:
#     f = R.T @ data.lambda_c
#     # get force at joint level
# fext = [pin.Force.Zero() for _ in range(model.njoints)]
# fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(f, np.zeros(3)))
#     # Get derivatives
# pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) 
# pin.computeRNEADerivatives(model,data, q0, v0, data.ddq, fext)
#     # Check constraint acc = 0
# assert(np.linalg.norm(pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear) <= 1e-6 )
# da0_dx_2bis = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
# print(testcolormap[test_da0_dx_2bis] + "   -- Test da0_dx ND with acc. constraint : " + str(test_da0_dx_2bis) + bcolors.ENDC)





# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def fdyn(model, data, frameId, x, tau, ref):
    '''
    fwdyn(x,u) = forward contact dynamics(q,v,tau) 
    returns the concatenation of configuration acceleration and contact forces 
    '''
    q=x[:nq]
    v=x[nq:]
    pin.computeAllTerms(model,data,q,v)
    pin.forwardKinematics(model,data,q,v,v*0)
    pin.updateFramePlacements(model,data)
    M = data.M
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        J = pin.getFrameJacobian(model,data,frameId,pin.LOCAL_WORLD_ALIGNED)[2:3,:]
        a0 = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear[2:3]
    else:
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[2:3,:]
        a0 = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL).linear[2:3]
    b = data.nle
    K = np.block([ [M,J.T],[J,np.zeros([nc,nc])] ])
    k = np.concatenate([ tau-b, -a0 ])
    af = np.linalg.inv(K)@k
    return af 


# METHOD 2: CONTACT LWA (rotated LOCAL) + adjust DAM (skew term in rnea_dx)
def fdyn2bis(model, data, frameId, x, tau, ref): 
    q=x[:nq]
    v=x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model,data)
    # Compute drift + jacobian in ref
    a0 = contactCalc2bis(model, data, frameId, x, np.zeros(nq), ref)
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[:3,:]
        assert(np.linalg.norm(a0 - pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear)<1e-6)
        assert(np.linalg.norm(J - data.oMf[frameId].rotation @ pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:])<1e-6)
    else:
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:]
        assert(np.linalg.norm(a0 - pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL).linear)<1e-6)
    # print("fdyn2bis.a0 = ", a0)
    # Call forward dynamics to get joint acc and force ( in local or lwa frame ) ok     identical to ground truth
    pin.forwardDynamics(model, data, tau, J, a0)
    return np.hstack([data.ddq, -data.lambda_c]) 


# METHOD 2: CONTACT LWA (rotated LOCAL) + adjust DAM (skew term in rnea_dx)
def fdyn_diff2bis(model, data, frameId, x, tau, ref):
    '''
    computes partial derivatives of joint acc (and force)
        using hard-coded contact model 3D calcDiff()
    '''
    q = x[:nv]
    v = x[nv:]
    Fx = np.zeros((nq,nx))
    df_dx = np.zeros((nc,nx))
    R = data.oMf[frameId].rotation
    # Compute RNEA derivatives and KKT inverse using force and joint acc computed in fwdDyn
        # here we need LOCAL
    if(ref == pin.LOCAL):
        f = data.lambda_c
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[2:3,:]  
    else:
        f = R.T @ data.lambda_c 
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[2:3,:]  
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[model.frames[frameId].parent].linear = model.frames[frameId].placement.rotation[:,2] * f[0]
    pin.computeRNEADerivatives(model, data, q, v, data.ddq, fext) 
    drnea_dx = np.hstack([data.dtau_dq, data.dtau_dv])
    Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, J)  
    # Contact derivatives 
    pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) 
    da0_dx = contactCalcDiff2Bis(model, data, frameId, x, ref)
    # print(da0_dx)
    # Check KKT inverse 
    K = np.block([ [data.M,J.T],[J,np.zeros([1,1])] ])
    assert(np.linalg.norm(np.linalg.inv(K) - Kinv) < 1e-6)

    # correct rnea derivatives
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL) 
        # print("skew term = \n", lJ[:3,:].T @ pin.skew(R.T @ data.lambda_c) @ oJ[3:,:])
        drnea_dx[:,:nv] -= lJ[2:3,:].T @ pin.skew(R.T @ data.lambda_c) @ lJ[3:,:]

    # # Check rnea' and a'
    # dy_dx = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x)
    # a0 = pin.getFrameClassicalAcceleration(model,data,frameId, PIN_REFERENCE_FRAME).linear
    # print(a0)
    # k = np.concatenate([ tau - data.nle, -a0 ])
    # dk_dx = np.vstack([drnea_dx, da0_dx])
    # y = -np.linalg.inv(K) @ k
    #     # check a'
    # print(np.isclose(-K@dy_dx, np.vstack([drnea_dx, da0_dx]), RTOL, ATOL))
    # da0_dx_truth = (-K @ dy_dx)[-nc:,:]
    # print("truth = \n", da0_dx_truth)
    # print(da0_dx)
    # print(np.isclose(da0_dx_truth, da0_dx, RTOL, ATOL))
    # assert(np.linalg.norm(da0_dx_truth -  da0_dx) < 1e-4) 
    #     #  Check rnea'
    # drnea_dx_truth = (-K @ dy_dx)[:nv,:]
    # print('after = \n', drnea_dx)
    # print('truth = \n', drnea_dx_truth)
    # print(np.isclose(drnea_dx_truth, drnea_dx, RTOL, ATOL))
    # print(np.isclose(da0_dx_truth, da0_dx, RTOL, ATOL))
    # # assert(np.linalg.norm(drnea_dx -  drnea_dx_before) < 1e-3) 
    # # assert(np.linalg.norm(drnea_dx_truth -  drnea_dx) < 1e-3)


    # Fillout partials of DAM 
    a_partial_dtau = Kinv[:nv, :nv]
    a_partial_da   = Kinv[:nv, -nc:]     
    f_partial_dtau = Kinv[-nc:, :nv]
    f_partial_da   = Kinv[-nc:, -nc:]

    
    Fx = -a_partial_dtau @ drnea_dx 
    # Fx[:,:nv] = -a_partial_dtau @ data.dtau_dq
    # Fx[:,nv:] = -a_partial_dtau @ data.dtau_dv
    Fx -= a_partial_da @ da0_dx[:nc] 
    Fx += a_partial_dtau @ np.zeros((nq,nx))
    # Fu = a_partial_dtau @ np.eye(nq)
    df_dx[:nc,:] = f_partial_dtau @ drnea_dx
    # df_dx[:nc, :nv]  = f_partial_dtau @ data.dtau_dq
    # df_dx[:nc, -nv:] = f_partial_dtau @ data.dtau_dv
    df_dx[:nc, :]   += f_partial_da @ da0_dx[:nc] 
    df_dx[:nc, :]   -= f_partial_dtau @ np.zeros((nq,nx))
    # df_du[:nc, :]  = -f_partial_dtau @ np.eye(nq)
    return np.vstack([Fx, -df_dx])



# # If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# # and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# # since the constraint is resolved 
# frameId = contactFrameId
# model = robot.model
# data = robot.model.createData()
# # Compare fdyn ground truth and fdyn2bis
# af = fdyn(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
# af2bis = fdyn2bis(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
# assert(np.linalg.norm(af - af2bis)<1e-6)
# np.hstack([data.ddq, -data.lambda_c]) 
# daf_dx_ND_0 = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
# daf_dx_ND_2bis = numdiff(lambda x_:fdyn2bis(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
# daf_dx_2bis    = fdyn_diff2bis(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
# test_daf_dx_0    = np.allclose(daf_dx_ND_0, daf_dx_ND_2bis, RTOL, ATOL)
# print(testcolormap[test_daf_dx_0] + "   -- Test fdyn_ND vs fdyn2bis_ND : " + str(test_daf_dx_0) + bcolors.ENDC)
# test_daf_dx_2bis    = np.allclose(daf_dx_2bis, daf_dx_ND_2bis, RTOL, ATOL)
# print(testcolormap[test_daf_dx_2bis] + "   -- Test fdyn_diff2bis vs fdyn2bis_ND : " + str(test_daf_dx_2bis) + bcolors.ENDC)






# Calc a0
frameId = contactFrameId
model = robot.model
data = robot.model.createData()

#   }
#   Data* d = static_cast<Data*>(data.get());
#   d->oRf = d->pinocchio->oMf[id_].rotation();
#   if(type_ == pinocchio::LOCAL) 
#   {
#     data->f.linear() = d->jMf.rotation().col(mask_) * force[0];
#     data->f.angular() = d->jMf.translation().cross(data->f.linear());
#   }
#   if(type_ == pinocchio::LOCAL_WORLD_ALIGNED || type_ == pinocchio::WORLD)
#   {
#     data->f = d->jMf.act(pinocchio::ForceTpl<Scalar>(d->oRf.transpose().col(mask_) * force[0], Vector3s::Zero()));
#   }


# forward dynamics in LOCAL 
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
la0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear[2:3]
lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
lJc = lJ[2:3]
# print(lJ[MASK]) 
pin.forwardDynamics(model, data, tau, lJc, la0)
lK = np.block([ [data.M, lJc.T],[lJc, np.zeros([1,1])] ])
laf = np.linalg.inv(lK) @ np.concatenate([tau - data.nle, -la0])
assert(np.linalg.norm(laf[:nv] - data.ddq) <1e-6)
assert(np.linalg.norm(laf[-nc:] + data.lambda_c) <1e-6)
# Derivatives in LOCAL
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent].linear = model.frames[frameId].placement.rotation[:,2] * data.lambda_c[0]
# print(fext)
# fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(data.lambda_c, np.zeros(3)))
pin.computeRNEADerivatives(model, data, q0, v0, data.ddq, fext) 
ldrnea_dx = np.hstack([data.dtau_dq, data.dtau_dv])
# print("local rnea 1 :  \n", ldrnea_dx)
lKinv = pin.getKKTContactDynamicMatrixInverse(model, data, lJc)  
assert(np.linalg.norm(lKinv - np.linalg.inv(lK)) <1e-6)
assert(np.linalg.norm(fdyn(model, data, frameId, x0, tau, pin.LOCAL) - laf) < 1e-6)
ldaf_dx_ND = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, pin.LOCAL), x0)
ldk_dx = -lK @ ldaf_dx_ND
    # Frame acceleration derivative
v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
vv = v.linear ; vw = v.angular
lda0_dx_3d = np.zeros((3,nx))
lda0_dx_3d[:,:nv] = a_partial_dq[:3,:]
lda0_dx_3d[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
lda0_dx_3d[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
lda0_dx_3d[:,nv:] = a_partial_dv[:3,:]
lda0_dx_3d[:,nv:] += pin.skew(vw) @ lJ[:3,:]
lda0_dx_3d[:,nv:] -= pin.skew(vv) @ lJ[3:,:]
lda0_dx = lda0_dx_3d[2]
assert(np.linalg.norm(lda0_dx - contactCalcDiff2Bis(model, data, frameId, x0, pin.LOCAL))<1e-4)
assert(np.linalg.norm(ldk_dx[-nc:] - lda0_dx) <1e-4)
assert(np.linalg.norm(ldk_dx[:nv] - ldrnea_dx) <1e-4)
assert(np.linalg.norm(ldaf_dx_ND + lKinv @ np.concatenate([ldrnea_dx, lda0_dx])) <1e-3)

# forward dynamics in WORLD
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
R = data.oMf[frameId].rotation
wa0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
wJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[:3,:]
pin.forwardDynamics(model, data, tau, wJf, wa0)
wK = np.block([ [data.M, wJf.T],[wJf, np.zeros([3,3])] ])
waf = np.linalg.inv(wK) @ np.concatenate([tau - data.nle, -wa0])
assert(np.linalg.norm(waf[:nv] - data.ddq) <1e-6)
assert(np.linalg.norm(waf[:nv] - laf[:nv] ) < 1e-6)
assert(np.linalg.norm(waf[-nc:] + data.lambda_c) <1e-6)
# Derivatives in WORLD
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(R.T @ data.lambda_c, np.zeros(3)))
pin.computeRNEADerivatives(model, data, q0, v0, data.ddq, fext) 
wdrnea_dx = np.hstack([data.dtau_dq, data.dtau_dv])
assert(np.linalg.norm(wdrnea_dx -ldrnea_dx) <1e-4)
    # additional term  
lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL) 
wdrnea_dx[:,:nv] -= lJ[:3].T @ pin.skew(R.T @ data.lambda_c) @ lJ[3:]
wKinv = pin.getKKTContactDynamicMatrixInverse(model, data, wJf)  
assert(np.linalg.norm(wKinv - np.linalg.inv(wK))<1e-6)
assert(np.linalg.norm(fdyn(model, data, frameId, x0, tau, pin.LOCAL_WORLD_ALIGNED) - waf) < 1e-6)
wdaf_dx_ND = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, pin.LOCAL_WORLD_ALIGNED), x0)
wdk_dx = -wK @ wdaf_dx_ND
    # Frame acc derivative 
wda0_dx = R @ lda0_dx
Jw = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) # very important 
    # print("linear acc after fwdDyn : ", pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear)
wa = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
assert(np.linalg.norm(wa) <1e-6)
# wda0_dx[:,:nq] -= pin.skew(wa)@Jw # no skew term after because acc = 0
assert(np.linalg.norm(wda0_dx - contactCalcDiff2Bis(model, data, frameId, x0, pin.LOCAL_WORLD_ALIGNED) ) <1e-6 )
assert(np.linalg.norm(wdk_dx[-nc:] - wda0_dx) <1e-4)
    # rnea derivatives
assert(np.linalg.norm(wdk_dx[:nv] - wdrnea_dx)<1e-4)
    # all
wdaf_dx = -wKinv @ np.vstack([wdrnea_dx, wda0_dx])
assert(np.linalg.norm(wdaf_dx_ND - wdaf_dx) <1e-3)

