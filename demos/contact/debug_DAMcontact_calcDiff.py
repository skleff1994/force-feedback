'''
Debugging calc and calcDiff of DAMContactFwdDyn
'''

import numpy as np
np.set_printoptions(precision=3, linewidth=180, suppress=True)

import example_robot_data 
import pinocchio as pin
import crocoddyl 


class bcolors:
    DEBUG = '\033[1m'+'\033[96m'
    ERROR = '\033[1m'+'\033[91m'
    ENDC = '\033[0m'


ND_DISTURBANCE  = 1e-6
GAUSS_APPROX    = True
RTOL            = 1e-3 
ATOL            = 1e-4 
RANDOM_SEED     = 1
np.random.seed(RANDOM_SEED)

# Test parameters 
PIN_REFERENCE_FRAME         = pin.WORLD     
ALIGN_LOCAL_WITH_WORLD      = False
TORQUE_SUCH_THAT_ZERO_FORCE = False
ZERO_JOINT_VELOCITY         = False


# Custom DAD with one 3D contact
class DADContact3D(crocoddyl.DifferentialActionDataContactFwdDynamics):
    def __init__(self, dam): 
        crocoddyl.DifferentialActionDataContactFwdDynamics.__init__(self, dam)
        self.xout = np.zeros(dam.nv)        
        self.Fx = np.zeros((dam.nv, dam.nx))
        self.Fu = np.zeros((dam.nv, dam.nu))
        self.Lx = np.zeros(dam.nx)
        self.Lu = np.zeros(dam.nu)
        self.Lxx = np.zeros((dam.nx, dam.nx))
        self.Lxu = np.zeros((dam.nx, dam.nu))
        self.Luu = np.zeros((dam.nu, dam.nu))
        # Custom contact model 
        self.a0 = np.zeros(dam.nc)
        self.a0_temp = np.zeros(dam.nc)
        self.Jc = np.zeros((dam.nv, dam.nc))
        self.f = pin.Force.Zero()
        self.fext = [pin.Force.Zero() for i in range(dam.rmodel.njoints)]
        self.da0_dx = np.zeros((dam.nc, dam.nx))
        self.da0_dx_temp = np.zeros((dam.nc, dam.nx))
        self.a = pin.Motion.Zero()
        self.v = pin.Motion.Zero()
        self.vv = np.zeros(3)
        self.vw = np.zeros(3)
        self.fJf = np.zeros((dam.nc, dam.nv))


# Custom DAM with one 3D contact
class DAMContact3D(crocoddyl.DifferentialActionModelContactFwdDynamics):
    def __init__(self, state, actuation, costModel, contactFrameId, refPosition=np.zeros(3), gains=np.zeros(2), ref=pin.LOCAL):
        # dummy contact model to size the DAM (overwritten in self.calc and self.calcDiff)
        contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
        contactModel.addContact("dummy_contact3D", crocoddyl.ContactModel3D(state, contactFrameId, refPosition, gains), active=True)
        crocoddyl.DifferentialActionModelContactFwdDynamics.__init__(self, state, actuation, contactModel, costModel, inv_damping=0, enable_force=True)
        self.rmodel = self.state.pinocchio
        self.nv = self.state.nv
        self.nx = self.state.nx
        self.contactFrameId = contactFrameId
        self.parentJointId = self.rmodel.frames[self.contactFrameId].parent
        self.refPosition = refPosition
        self.gains = gains
        self.ref = ref
        self.nc = 3
        self.enable_force = True
        self.jMf = self.rmodel.frames[contactFrameId].placement
        self.fXj = self.rmodel.frames[contactFrameId].placement.actionInverse
        
    def calc(self, data, x, u):
        '''
        Computes joint acc 
         using hard-coded contact model 3D calc()
        '''
        rdata = data.pinocchio 
        q = x[:self.nv]
        v = x[self.nv:]
        # oRf = rdata.oMf[self.contactFrameId].rotation

        pin.computeAllTerms(self.rmodel, rdata, q, v)
        pin.computeCentroidalMomentum(self.rmodel, rdata)
        self.actuation.calc(data.multibody.actuation, x, u)
        

        # Hard-coded ontact model calc()
            # Tested against bindings in Python : OK
        pin.updateFramePlacement(self.rmodel, rdata, self.contactFrameId)
        data.fJf = pin.getFrameJacobian(self.rmodel, rdata, self.contactFrameId, pin.LOCAL)
        data.v = pin.getFrameVelocity(self.rmodel, rdata, self.contactFrameId, pin.LOCAL)
        data.a = pin.getFrameAcceleration(self.rmodel, rdata, self.contactFrameId, pin.LOCAL)
        data.vv = data.v.linear ; data.vw = data.v.angular
        # if(self.ref == pin.WORLD or self.ref == pin.LOCAL_WORLD_ALIGNED):
        #     data.a0 = pin.getFrameClassicalAcceleration(self.rmodel, rdata, self.contactFrameId, pin.LOCAL_WORLD_ALIGNED).linear
        #     data.Jc = oRf @ data.fJf[:3,:] 
        # else:
        data.a0 = pin.getFrameClassicalAcceleration(self.rmodel, rdata, self.contactFrameId, pin.LOCAL).linear
        data.Jc = data.fJf[:3,:] 
        assert(np.linalg.norm(data.a.linear + np.cross(data.v.angular, data.v.linear) - data.a0) <= 1e-6 )
        # data.a0 = data.a.linear + np.cross(data.vw, data.vv)
        # data.Jc = data.fJf[:3,:]
        
        # 
        # if(self.ref == pin.WORLD or self.ref == pin.LOCAL_WORLD_ALIGNED):
        #     data.a0 = oRf @ data.a0_temp

        # Call forward dynamics
        pin.forwardDynamics(self.rmodel, rdata, data.multibody.actuation.tau, data.Jc, data.a0)
        # print("force lambda_c LOCAL = \n ", rdata.lambda_c)
        # # Record force at joint level (i.e. contactModel3D.updateForce() in C++)
        # if(self.ref == pin.WORLD or self.ref == pin.LOCAL_WORLD_ALIGNED):
        #     data.f = self.jMf.act(pin.Force(oRf.T @ rdata.lambda_c, np.zeros(3)))
        # elif(self.ref == pin.LOCAL):
        #     data.f = self.jMf.act(pin.Force(rdata.lambda_c, np.zeros(3)))
        # # Fill out forces acting on each joint (i.e. contactModelMultiple.updateForce() in C++)
        # f_WORLD = rdata.oMf[self.contactFrameId].act(self.jMf.actInv(data.f))
        # f_JOINT = rdata.oMi[self.parentJointId].actInv(f_WORLD)
        # if(self.ref == pin.WORLD or self.ref == pin.LOCAL_WORLD_ALIGNED):
        #     data.fext[self.parentJointId] = self.jMf.act(pin.Force(oRf.T @ rdata.lambda_c, np.zeros(3)))
        # else:
        data.fext[self.parentJointId] = self.jMf.act(pin.Force(rdata.lambda_c, np.zeros(3)))
        # print("FEXT (JOINT) = \n", data.fext[self.parentJointId])
        # Record joint acceleration     
        data.xout = rdata.ddq
        
        return data 

    def calcDiff(self, data, x, u):
        '''
        computes partial derivatives of joint acc (and force)
         using hard-coded contact model 3D calcDiff()
        '''
        rdata = data.pinocchio 
        q = x[:self.nv]
        v = x[self.nv:]
        oRf = rdata.oMf[self.contactFrameId].rotation
        # Compute RNEA derivatives and KKT inverse
        pin.computeRNEADerivatives(self.rmodel, rdata, q, v, data.xout, data.fext) # SAME
        # print("xout = \n", data.xout)
        # print("fext = \n", data.fext)
        # print("Jc = \n", data.Jc)
        Kinv = pin.getKKTContactDynamicMatrixInverse(self.rmodel, rdata, data.Jc)  # SAME
        
        # Actuation derivatives
        actuation.calcDiff(data.multibody.actuation, x, tau)

        # Hard-coded contact model derivatives 
            # Tested against numdiff in C++     : OK
            # Tested against bindings in Python : OK
        # parendJointId = self.rmodel.frames[self.contactFrameId].parent   
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(self.rmodel, rdata, self.contactFrameId, pin.LOCAL) 
        vv_skew = pin.skew(data.vv)
        vw_skew = pin.skew(data.vw)
        data.da0_dx[:,:self.nv] = a_partial_dq[:3,:]
        data.da0_dx[:,:self.nv] += vw_skew @ v_partial_dq[:3,:]
        data.da0_dx[:,:self.nv] -= vv_skew @ v_partial_dq[3:,:]
        data.da0_dx[:,self.nv:] = a_partial_dv[:3,:]
        data.da0_dx[:,self.nv:] += vw_skew @ data.fJf[:3,:] 
        data.da0_dx[:,self.nv:] -= vv_skew @ data.fJf[3:,:]
        # print(data.da0_dx)
            # Add Baumgarte gains to data.da0_dx here if necessary
        # data.da0_dx_temp = data.da0_dx.copy()

        # print("da0_dx_temp before (LOCAL) : \n", data.da0_dx_temp)

        # if(self.ref == pin.LOCAL_WORLD_ALIGNED or self.ref == pin.WORLD):
        #     # tmp_skew = pin.skew(data.a0) 
        #     Jw = pin.getFrameJacobian(self.rmodel, rdata, self.contactFrameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
        #     data.da0_dx[:,self.nv:] = oRf @ data.da0_dx_temp[:,self.nv:] - pin.skew(data.a0) @ Jw #oRf @ data.fJf[3:,:]
        #     data.da0_dx[:,:self.nv] = oRf @ data.da0_dx_temp[:,:self.nv]
        # print("DAO_DX = \n", data.da0_dx)
        # derivatives of drift in WORLD 
        # but express dam derivative WORLD as a function of dam deriv LOCAL, which is a function of drift deriv LOCAL
        # Fill out DAM partials
        a_partial_dtau = Kinv[:self.nv, :self.nv]
        a_partial_da   = Kinv[:self.nv, -self.nc:]     
        f_partial_dtau = Kinv[-self.nc:, :self.nv]
        f_partial_da   = Kinv[-self.nc:, -self.nc:]
        data.Fx[:,:self.nv] = -a_partial_dtau @ rdata.dtau_dq
        data.Fx[:,self.nv:] = -a_partial_dtau @ rdata.dtau_dv
        data.Fx -= a_partial_da @ data.da0_dx[:self.nc] 
        data.Fx += a_partial_dtau @ data.multibody.actuation.dtau_dx
        data.Fu = a_partial_dtau @ data.multibody.actuation.dtau_du

        # if(self.enable_force):
        data.df_dx[:self.nc, :self.nv]  = f_partial_dtau @ rdata.dtau_dq
        data.df_dx[:self.nc, -self.nv:] = f_partial_dtau @ rdata.dtau_dv
        data.df_dx[:self.nc, :]   += f_partial_da @ data.da0_dx[:self.nc] 
        data.df_dx[:self.nc, :]   -= f_partial_dtau @ data.multibody.actuation.dtau_dx
        data.df_du[:self.nc, :]  = -f_partial_dtau @ data.multibody.actuation.dtau_du
        
        # # Rotate force
        # df_dx_temp = data.df_dx.copy()
        # print("df_dx before (LOCAL) : \n", df_dx_temp)
        if(self.ref == pin.WORLD or self.ref == pin.LOCAL_WORLD_ALIGNED):
            Jw = pin.getFrameJacobian(self.rmodel, rdata, self.contactFrameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
            data.df_dx[:self.nc,:] = oRf @ data.df_dx[:self.nc,:]
            data.df_dx[:self.nc,:nv] -= pin.skew(rdata.lambda_c)@Jw
        # print("drnea_dq = \n", rdata.dtau_dq)
        print("world : \n", np.vstack([data.Fx, data.df_dx]))
        return data 

    def createData(self):
        data = DADContact3D(self) 
        return data         





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



# State, actuation, cost models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
runningCostModel = crocoddyl.CostModelSum(state) # dummy cost model (no cost is used)

# Create DAM + DAM NumDiff 
DAM = DAMContact3D(state, actuation, runningCostModel, contactFrameId, ref=PIN_REFERENCE_FRAME)
DAM_ND = crocoddyl.DifferentialActionModelNumDiff(DAM, GAUSS_APPROX)
DAD    = DAM.createData()
DAD_ND = DAM_ND.createData()
DAM_ND.disturbance = ND_DISTURBANCE
testcolormap = {False: bcolors.ERROR , True: bcolors.DEBUG}

print("\n")

# TEST CALC
DAM.calc(DAD, x0, tau)
DAM_ND.calc(DAD_ND, x0, tau)
print(bcolors.DEBUG + "--- TEST CALC FUNCTION ---" + bcolors.ENDC)
test_xout = np.allclose(DAD.xout, DAD_ND.xout, RTOL, ATOL)
print(testcolormap[test_xout] + "   -- Test xout : " + str(test_xout) + bcolors.ENDC)

# TEST CALCDIFF
print(bcolors.DEBUG + "--- TEST CALCDIFF FUNCTION ---" + bcolors.ENDC)
DAM.calcDiff(DAD, x0, tau)
DAM_ND.calcDiff(DAD_ND, x0, tau)

test_Fu = np.allclose(DAD.Fu, DAD_ND.Fu, RTOL, ATOL)
print(testcolormap[test_Fu] + "   -- Test Fu : " + str(test_Fu) + bcolors.ENDC)

test_Fx = np.allclose(DAD.Fx, DAD_ND.Fx, RTOL, ATOL)
print(testcolormap[test_Fx] + "   -- Test Fx : " + str(test_Fx) + bcolors.ENDC)
if(test_Fx == False):
    test_Fq = np.allclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL)
    print(testcolormap[test_Fq] + "           -- Fq : " + str(test_Fq) + bcolors.ENDC)
    test_Fv = np.allclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL)
    print(testcolormap[test_Fv] + "           -- Fv : " + str(test_Fv) + bcolors.ENDC)


# print("acc = \n", DAD.xout)
# print("fext = \n", DAD.f)
# print("d(acc) = \n", DAD.Fx)
# print("d(force) = \n", DAD.df_dx)