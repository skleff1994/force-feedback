'''
Debugging calc and calcDiff of DAMContactFwdDyn
'''

import numpy as np
np.set_printoptions(precision=4, linewidth=180)

import example_robot_data 
import pinocchio as pin
import crocoddyl 


class bcolors:
    DEBUG = '\033[1m'+'\033[96m'
    ERROR = '\033[1m'+'\033[91m'
    ENDC = '\033[0m'


ND_DISTURBANCE  = 1e-6
GAUSS_APPROX    = False
RTOL            = 1e-3 #1e-3
ATOL            = 1e-4 #1e-5
RANDOM_SEED     = 1
np.random.seed(RANDOM_SEED)
PIN_REFERENCE_FRAME   = pin.LOCAL

ALIGN_LOCAL_WITH_WORLD      = True
TORQUE_SUCH_THAT_ZERO_FORCE = True
ZERO_JOINT_VELOCITY         = False

# Load robot 
robot = example_robot_data.load('talos_arm')
nq = robot.model.nq; nv = robot.model.nv; nu = nq; nx = nq+nv
q0 = np.random.rand(nq) ; v0 = np.random.rand(nv) #np.zeros(nq)  
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
print("x0  = "+str(x0))


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
print("Contact frame placement oRf : \n"+str(robot.data.oMf[contactFrameId]))

# Optionally pick torque s.t. force is zero
if(TORQUE_SUCH_THAT_ZERO_FORCE):
    print(bcolors.DEBUG + "Select tau s.t. contact force = 0" + bcolors.ENDC)
    # Compute rnea( q=q0, vq=v0, aq=J^+ * gamma, f_ext=0 )
    f_ext = [pin.Force.Zero() for i in range(robot.model.njoints)]
    pin.computeAllTerms(robot.model, robot.data, q0, v0)
    J = pin.getFrameJacobian(robot.model, robot.data, contactFrameId, pin.LOCAL)[:3,:]
    gamma = -pin.getFrameClassicalAcceleration(robot.model, robot.data, contactFrameId, pin.LOCAL)
    aq    = np.linalg.pinv(J) @ gamma.vector[:3]
    tau   = pin.rnea(robot.model, robot.data, q0, v0, aq, f_ext)
else:
    tau = np.random.rand(nq)

print("tau = "+str(tau))


# Custom DAD with one 3D contact
class DADContact3D(crocoddyl.DifferentialActionDataContactFwdDynamics):
    def __init__(self, dam): 
        crocoddyl.DifferentialActionDataContactFwdDynamics.__init__(self, dam)
        self.xout = np.zeros(dam.nv)        
        # self.df_dx = np.zeros((dam.nc, dam.nx))
        # self.df_du = np.zeros((dam.nc, dam.nu))
        self.Fx = np.zeros((dam.nv, dam.nx))
        self.Fu = np.zeros((dam.nv, dam.nu))
        self.Lx = np.zeros(dam.nx)
        self.Lu = np.zeros(dam.nu)
        self.Lxx = np.zeros((dam.nx, dam.nx))
        self.Lxu = np.zeros((dam.nx, dam.nu))
        self.Luu = np.zeros((dam.nu, dam.nu))
        # Custom contact model 
        self.a0 = np.zeros(dam.nc)
        self.Jc = np.zeros((dam.nv, dam.nc))
        self.wrench = pin.Force.Zero()
        self.f_ext = [pin.Force.Zero() for i in range(dam.rmodel.njoints)]
        self.da0_dx = np.zeros((dam.nc, dam.nx))
        self.v = pin.Motion.Zero()
        self.vv = np.zeros(3)
        self.vw = np.zeros(3)
        self.fJf = np.zeros((dam.nc, dam.nv))

# Custom DAM with one 3D contact, empty cost
class DAMContact3D(crocoddyl.DifferentialActionModelContactFwdDynamics):
    def __init__(self, state, actuation, costModel, contactFrameId, refPosition=np.zeros(3), gains=np.zeros(2), ref=pin.LOCAL):
        # dummy contact model to size the DAM (overwritten in self.calc and self.calcDiff)
        contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
        contactModel.addContact("contact", crocoddyl.ContactModel3D(state, contactFrameId, refPosition, gains), active=True)
        crocoddyl.DifferentialActionModelContactFwdDynamics.__init__(self, state, actuation, contactModel, costModel, inv_damping=0, enable_force=True)
        self.rmodel = self.state.pinocchio
        self.nv = self.state.nv
        self.nx = self.state.nx
        self.contactFrameId = contactFrameId
        self.refPosition = refPosition
        self.gains = gains
        self.ref = ref
        self.nc = 3
        self.enable_force = True
        self.jMf = self.rmodel.frames[contactFrameId].placement
        self.fXj = self.rmodel.frames[contactFrameId].placement.actionInverse
        
    def calc(self, data, x, u):
        rdata = data.pinocchio 
        q = x[:self.nv]
        v = x[self.nv:]
        # Compute calc in python
        pin.computeAllTerms(self.rmodel, rdata, q, v)
        pin.computeCentroidalMomentum(self.rmodel, rdata)
        self.actuation.calc(data.multibody.actuation, x, u)
        # Contact model calc
        pin.updateFramePlacement(self.rmodel, rdata, self.contactFrameId)
        data.fJf = pin.getFrameJacobian(self.rmodel, rdata, self.contactFrameId, pin.LOCAL)
        data.v = pin.getFrameVelocity(self.rmodel, rdata, self.contactFrameId, pin.LOCAL)
        data.vv = data.v.linear ; data.vw = data.v.angular
        oRf = rdata.oMf[self.contactFrameId].rotation
        data.a0 = pin.getFrameClassicalAcceleration(self.rmodel, rdata, self.contactFrameId, pin.LOCAL).linear
        data.Jc = data.fJf[:3,:]
        # WORLD ALIGNED
        if(self.ref == pin.WORLD or self.ref == pin.LOCAL_WORLD_ALIGNED):
            data.a0 = oRf @ data.a0.copy()   
            data.Jc = oRf @ data.Jc.copy()  
            # call forward dyn
            pin.forwardDynamics(self.rmodel, rdata, data.multibody.actuation.tau, data.Jc, data.a0)
            # record force at joint level
            data.wrench = pin.Force(oRf.T @ rdata.lambda_c, np.zeros(3))
            for i in range(self.rmodel.njoints):
                # CONTACT --> WORLD --> JOINT
                f_WORLD = data.oMf[self.contactFrameId].act(data.wrench)
                f_JOINT = robot.data.oMi[i].actInv(f_WORLD)
                data.f_ext[i] = pin.Force(f_JOINT)
        # LOCAL
        elif(self.ref == pin.LOCAL):
            pin.forwardDynamics(self.rmodel, rdata, data.multibody.actuation.tau, data.Jc, data.a0)
            # record force at joint level
            data.wrench = pin.Force(rdata.lambda_c, np.zeros(3))
            for i in range(self.rmodel.njoints):
                # CONTACT --> JOINT
                data.f_ext[i] = self.rmodel.frames[contactFrameId].placement.act(data.wrench)
        # Record joint acceleration     
        data.xout = rdata.ddq
        return data 

    def calcDiff(self, data, x, u):
        
        rdata = data.pinocchio 
        q = x[:self.nv]
        v = x[self.nv:]
        oRf = rdata.oMf[self.contactFrameId].rotation
        
        # Compute RNEA derivatives and KKT inverse
        pin.computeRNEADerivatives(self.rmodel, rdata, q, v, data.xout, data.f_ext)
        Kinv = pin.getKKTContactDynamicMatrixInverse(self.rmodel, rdata, data.Jc)
        # Actuation derivatives
        actuation.calcDiff(data.multibody.actuation, x, tau)

        # Contact model derivatives     
        parendJointId = self.rmodel.frames[self.contactFrameId].parent   
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getJointAccelerationDerivatives(self.rmodel, rdata, parendJointId, pin.LOCAL) 
        vv_skew = pin.skew(data.vv)
        vw_skew = pin.skew(data.vw)
        data.da0_dx[:,self.nv:] = (self.fXj @ a_partial_dq)[:3,:]
        data.da0_dx[:,self.nv:] += vw_skew @ (self.fXj @ v_partial_dq)[:3,:]
        data.da0_dx[:,self.nv:] -= vv_skew @ (self.fXj @ v_partial_dq)[3:,:]
        data.da0_dx[:,:self.nv] = (self.fXj @ a_partial_dv)[:3,:]
        data.da0_dx[:,:self.nv] += vw_skew @ data.fJf[:3,:] 
        data.da0_dx[:,:self.nv] -= vv_skew @ data.fJf[3:,:]
            # Add Baumgarte gains here if necessary
        da0_dx_temp_ = data.da0_dx.copy()
        if(self.ref == pin.LOCAL_WORLD_ALIGNED or self.ref == pin.WORLD):
            tmp_skew = pin.skew(oRf @ data.a0_temp_)
            data.da0_dx[:,self.nv:] = oRf @ da0_dx_temp_[:,self.nv:] - data.tmp_skew @ oRf @ data.fJf[3:,:]
            data.da0_dx[:,:self.nv] = oRf @ da0_dx_temp_[:,:self.nv]

        # Check KKT inv
        # KKT = np.zeros((self.nv+self.nc, self.nv+self.nc))
        # KKT[:nq,:nq] = rdata.M         ; KKT[:nq,nq:] = data.Jc.T
        # KKT[nq:,:nq] = data.Jc ; KKT[nq:,nq:] = np.zeros((self.nc, self.nc))
        # print(bcolors.DEBUG + np.allclose(Kinv, np.linalg.inv(KKT), RTOL, ATOL))

        # Fill out DAM partials
        a_partial_dtau = Kinv[:nv, :nv]
        a_partial_da   = Kinv[:nv, -self.nc:]     
        f_partial_dtau = Kinv[-self.nc:, :nv]
        f_partial_da   = Kinv[-self.nc:, -self.nc:]
        data.Fx[:,:nq] = -a_partial_dtau @ rdata.dtau_dq
        data.Fx[:,nq:] = -a_partial_dtau @ rdata.dtau_dv
        data.Fx -= a_partial_da @ data.da0_dx[:self.nc]
        data.Fx += a_partial_dtau @ data.multibody.actuation.dtau_dx
        data.Fu = a_partial_dtau @ data.multibody.actuation.dtau_du

        if(self.enable_force):
            # df_dx = np.zeros((self.nc, self.nx))
            # df_du = np.zeros((self.nc, self.nu))
            data.df_dx[:self.nc, :self.nv]  = f_partial_dtau @ rdata.dtau_dq
            data.df_dx[:self.nc, -self.nv:] = f_partial_dtau @ rdata.dtau_dv
            data.df_dx[:self.nc, :]   += f_partial_da @ data.da0_dx[:self.nc]
            data.df_dx[:self.nc, :]   -= f_partial_dtau @ data.multibody.actuation.dtau_dx

            data.df_du[:self.nc, :] = -f_partial_dtau @ data.multibody.actuation.dtau_du
            # Update acc and force derivatives
            # contactModel.updateAccelerationDiff(contactData, Fx[-self.nv:,:])
            # contactModel.updateForceDiff(contactData, df_dx[:self.nc, :], df_du[:self.nc, :])
        # runningCostModel.calcDiff(data, x0, tau)
        return data 

    def createData(self):
        data = DADContact3D(self) 
        return data         



# State, actuation, cost models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
runningCostModel = crocoddyl.CostModelSum(state)
# Dummy contact model to properly allocate data in custom DAM

DAM = DAMContact3D(state, actuation, runningCostModel, contactFrameId)
# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
DAM_ND = crocoddyl.DifferentialActionModelNumDiff(DAM, GAUSS_APPROX)
DAD    = DAM.createData()
DAD_ND = DAM_ND.createData()
DAM_ND.disturbance = ND_DISTURBANCE
# calc versus ND
DAM.calc(DAD, x0, tau)
DAM_ND.calc(DAD_ND, x0, tau)
print(bcolors.DEBUG + "--- TEST CALC FUNCTION ---" + bcolors.ENDC)
print(bcolors.DEBUG + "   -- xout (model vs numdiff) --" + bcolors.ENDC)
print("MODEL.xout   : "+str(DAD.xout))
print("NUMDIFF.xout : "+str(DAD_ND.xout))
print(bcolors.DEBUG + str(np.allclose(DAD.xout, DAD_ND.xout, RTOL, ATOL)))


print("\n")

# calcDiff
print(bcolors.DEBUG + "--- TEST CALCDIFF FUNCTION ---" + bcolors.ENDC)
DAM.calcDiff(DAD, x0, tau)
DAM_ND.calcDiff(DAD_ND, x0, tau)

print(bcolors.DEBUG + "   -- Test Fu (model vs numdiff) --" + bcolors.ENDC)
# print("MODEL.Fu   :\n "+ str(DAD.Fu))
# print("NUMDIFF.Fu :\n "+ str(DAD_ND.Fu))
print(bcolors.DEBUG + str(np.allclose(DAD.Fu, DAD_ND.Fu, RTOL, ATOL)))

print(bcolors.DEBUG + "   -- Test Fx (model vs numdiff) --" + bcolors.ENDC)
# print(bcolors.DEBUG + "           Fx")
# print("MODEL.Fx   :\n "+ str(DAD.Fx))
# print("NUMDIFF.Fx :\n "+ str(DAD_ND.Fx))
print(bcolors.DEBUG + "           Fq")
print(bcolors.DEBUG + str(np.allclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL)))
# print(bcolors.DEBUG + "\n"+str(np.isclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL)))
print(bcolors.DEBUG + "           Fv")
print(bcolors.DEBUG + str(np.allclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL)))
# print(bcolors.DEBUG + "\n"+str(np.isclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL)))

# # Calc vs pinocchio analytical 
# # print("TAU = ", pin_utils.get_tau(q0, v0, xout, contactData.fext, model, np.zeros(nq)))
# pin.computeRNEADerivatives(model, data, q0, v0, xout, contactData.fext)
# Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, contactData.Jc) #Jc[:nc])
# actuation.calcDiff(actuationData, x0, tau)
# contactModel.calcDiff(contactData, x0) 

# print(bcolors.DEBUG + "   -- Test KKT (model vs python) --" + bcolors.ENDC)
# # print("PIN.KKTinv   :\n "+ str(Kinv))
# # print("MODEL.KKTinv :\n "+ str(DAD.Kinv))
# print(bcolors.DEBUG + np.allclose(Kinv, DAD.Kinv, RTOL, ATOL))
# KKT = np.zeros((nq+nc, nq+nc))
# KKT[:nq,:nq] = data.M         ; KKT[:nq,nq:] = contactData.Jc.T
# KKT[nq:,:nq] = contactData.Jc ; KKT[nq:,nq:] = np.zeros((nc,nc))
# print(bcolors.DEBUG + np.allclose(Kinv, np.linalg.inv(KKT), RTOL, ATOL))

# print(bcolors.DEBUG + "   -- Test dtau_dq (model vs python) --" + bcolors.ENDC)
# # print("dtau_dq :\n"+str(data.dtau_dq))
# # print("dtau_dq :\n"+str(DAD.multibody.pin.dtau_dq))
# print(bcolors.DEBUG + np.allclose(data.dtau_dq, DAD.multibody.pin.dtau_dq, RTOL, ATOL))

# print(bcolors.DEBUG + "   -- Test dtau_dv (model vs python) --" + bcolors.ENDC)
# # print("dtau_dv :\n"+str(data.dtau_dv))
# # print("dtau_dv :\n"+str(DAD.multibody.pin.dtau_dv))
# print(bcolors.DEBUG + np.allclose(data.dtau_dv, DAD.multibody.pin.dtau_dv, RTOL, ATOL))

# print(bcolors.DEBUG + "   -- Test actuation.dtau_dx (model vs python) --" + bcolors.ENDC)
# # print("dtau_dx :\n"+str(actuationData.dtau_dx))
# # print("dtau_dx :\n"+str(DAD.multibody.actuation.dtau_dx))
# print(bcolors.DEBUG + np.allclose(actuationData.dtau_dx, DAD.multibody.actuation.dtau_dx, RTOL, ATOL))

# print(bcolors.DEBUG + "   -- Test actuation.dtau_du (model vs python) --" + bcolors.ENDC)
# # print("dtau_du :\n"+str(actuationData.dtau_du))
# # print("dtau_du :\n"+str(DAD.multibody.actuation.dtau_du))
# print(bcolors.DEBUG + np.allclose(actuationData.dtau_du, DAD.multibody.actuation.dtau_du, RTOL, ATOL))

# print(bcolors.DEBUG + "   -- Test contact.da0_dx (model vs python) --" + bcolors.ENDC)
# # print("da0_dx :\n"+str(contactData.da0_dx))
# # print("da0_dx :\n"+str(DAD.multibody.contacts.da0_dx))
# print(bcolors.DEBUG + np.allclose(contactData.da0_dx, DAD.multibody.contacts.da0_dx, RTOL, ATOL))

# da0_dx = np.zeros((nc, nx))
# # print(contactData.da0_dx )
# da0_dx[:nc, :nx] = contactData.da0_dx 
# da0_dx[:nc, :nq] += pin.skew(oRf @ contactData.a0) @ oJf[3:]
# da0_dx[:nc, :nq] = oRf.T @ da0_dx[:nc, :nq]
# da0_dx[:nc, nq:] = oRf.T @ da0_dx[:nc, nq:]
# # print(pin.skew(oRf @ contactData.a0) @ oJf[3:])
# # Fill out stuff 
# a_partial_dtau = Kinv[:nv, :nv]
# a_partial_da   = Kinv[:nv, -self.nc:]     
# f_partial_dtau = Kinv[-self.nc:, :nv]
# f_partial_da   = Kinv[-self.nc:, -self.nc:]

# Fx = np.zeros((nv, nx))
# Fx[:,:nq] = -a_partial_dtau @ data.dtau_dq
# Fx[:,nq:] = -a_partial_dtau @ data.dtau_dv
# Fx -= a_partial_da @ da0_dx[:nc]
# Fx += a_partial_dtau @ actuationData.dtau_dx
# Fu = a_partial_dtau @ actuationData.dtau_du

# if(enable_force):
#     df_dx = np.zeros((nc, nx))
#     df_du = np.zeros((nc, nu))

#     df_dx[:nc, :nv]  = f_partial_dtau @ data.dtau_dq
#     df_dx[:nc, -nv:] = f_partial_dtau @ data.dtau_dv
#     df_dx[:nc, :]   += f_partial_da @ da0_dx[:nc]
#     df_dx[:nc, :]   -= f_partial_dtau @ actuationData.dtau_dx

#     df_du[:nc, :] = -f_partial_dtau @ actuationData.dtau_du

#     # Update acc and force derivatives
#     contactModel.updateAccelerationDiff(contactData, Fx[-nv:,:])
#     contactModel.updateForceDiff(contactData, df_dx[:nc, :], df_du[:nc, :])


#     print(bcolors.DEBUG + "   -- Test ddv_dx (model vs python) --" + bcolors.ENDC)
#     # print("PIN.ddv_dx   :\n "+ str(contactData.ddv_dx))
#     # print("MODEL.ddv_dx :\n "+ str(DAD.multibody.contacts.ddv_dx))
#     print(bcolors.DEBUG + np.allclose(contactData.ddv_dx, DAD.multibody.contacts.ddv_dx, RTOL, ATOL))


#     print(bcolors.DEBUG + "   -- Test df_dx (model vs python) --" + bcolors.ENDC)
#     # print("PIN.df_dx   :\n "+ str(contactData.contacts["contact_"+contact_frame_name].df_dx))
#     # print("MODEL.df_dx :\n "+ str(DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_dx))
#     print(bcolors.DEBUG + np.allclose(contactData.contacts["contact_"+contact_frame_name].df_dx, DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_dx, RTOL, ATOL))

#     print(bcolors.DEBUG + "   -- Test df_du (model vs python) --" + bcolors.ENDC)
#     # print("PIN.df_du   :\n "+ str(contactData.contacts["contact_"+contact_frame_name].df_du))
#     # print("MODEL.df_du :\n "+ str(DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_du))
#     print(bcolors.DEBUG + np.allclose(contactData.contacts["contact_"+contact_frame_name].df_du, DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_du, RTOL, ATOL))


# print(bcolors.DEBUG + "   -- Test Fu (python vs numdiff) --" + bcolors.ENDC)
# # print("PYTHON.Fu   :\n "+ str(Fu))
# # print("NUMDIFF.Fu :\n "+ str(DAD_ND.Fu))
# print(bcolors.DEBUG + np.allclose(Fu, DAD_ND.Fu, RTOL, ATOL))

# print(bcolors.DEBUG + "   -- Test Fx (python vs numdiff) --" + bcolors.ENDC)
# print("PYTHON.Fx   :\n "+ str(Fx))
# print("NUMDIFF.Fx :\n "+ str(DAD_ND.Fx))
# print(bcolors.DEBUG + "           Fq")
# print(bcolors.DEBUG + np.allclose(Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL))
# # print(bcolors.DEBUG + "\n"+str(np.isclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL)))
# print(bcolors.DEBUG + "           Fv")
# print(bcolors.DEBUG + np.allclose(Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL))
# # print(bcolors.DEBUG + "\n"+str(np.isclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL)))


# runningCostModel.calcDiff(costData, x0, tau)

# print(bcolors.DEBUG + "   -- Test Fu (model vs python) --" + bcolors.ENDC)
# print(bcolors.DEBUG + np.allclose(Fu, DAD.Fu, RTOL, ATOL))
# # print(bcolors.DEBUG + "   -- Test Fu (numdiff vs python) --" + bcolors.ENDC)
# # print(bcolors.DEBUG + np.allclose(Fu, DAD_ND.Fu, RTOL, ATOL))
# print(bcolors.DEBUG + "   -- Test Fx (model vs python) --" + bcolors.ENDC)
# # print("PIN.Fx   :\n "+ str(Fx))
# # print("MODEL.Fx :\n "+ str(DAD.Fx))
# print(bcolors.DEBUG + np.allclose(Fx, DAD.Fx, RTOL, ATOL))
# # print(bcolors.DEBUG + "\n"+str(np.isclose(Fx, DAD.Fx, RTOL, ATOL)))



# if(WITH_COSTS):
#     # Control regularization cost
#     uResidual = crocoddyl.ResidualModelContactControlGrav(state)
#     uRegCost = crocoddyl.CostModelResidual(state, uResidual)
#     # State regularization cost
#     xResidual = crocoddyl.ResidualModelState(state, x0)
#     xRegCost = crocoddyl.CostModelResidual(state, xResidual)
#     # End-effector frame force cost
#     desired_wrench = np.array([0., 0., -20., 0., 0., 0.])
#     frameForceResidual = crocoddyl.ResidualModelContactForce(state, contactFrameId, pin.Force(desired_wrench), nc, actuation.nu)
#     contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
#     # Populate cost models with cost terms
#     runningCostModel.addCost("stateReg", xRegCost, 1e-2)
#     runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
#     runningCostModel.addCost("force", contactForceCost, 10.)
#     terminalCostModel.addCost("stateReg", xRegCost, 1e-2)
