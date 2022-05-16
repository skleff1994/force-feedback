import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)

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
PIN_REFERENCE_FRAME         = pin.LOCAL_WORLD_ALIGNED     
ALIGN_LOCAL_WITH_WORLD      = False
TORQUE_SUCH_THAT_ZERO_FORCE = False
ZERO_JOINT_VELOCITY         = False

print(bcolors.DEBUG + "Reference frame = " + str(PIN_REFERENCE_FRAME) + bcolors.ENDC)


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
def fdyn(model, data, frameId, q, v, tau, Kp, Kv, P_ANCHOR):
    '''
    fwdyn(x,u) = forward contact dynamics(q,v,tau) 
    returns the concatenation of configuration acceleration and contact forces 
    '''
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oRf = data.oMf[frameId].rotation
    lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    pdot = lJ[:3] @ v0
    force = -Kp*(oRf.T @ data.oMf[frameId].translation - P_ANCHOR) - Kv*pdot
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(force, np.zeros(3)))
    aq = pin.aba(model, data, q0, v0, tau, fext)
    print("acc = \n")
    print(aq)
    return aq


contactFrameName = "gripper_left_fingertip_1_link"
model = robot.model
data = robot.model.createData()
frameId = model.getFrameId(contactFrameName)


Kp = 1
Kv = 2*np.sqrt(Kp)
P_ANCHOR = np.zeros(3)
# V_ANCHOR = np.zeros(3) 


# forward dynamics 
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
# Compute visco-elastic contact force 
oRf = data.oMf[frameId].rotation
lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
pdot = lJ[:3] @ v0 
assert(np.linalg.norm(pdot - pin.getFrameVelocity(model, data, frameId, pin.LOCAL).linear) <1e-4)
force = -Kp*(oRf.T @ data.oMf[frameId].translation - P_ANCHOR) - Kv*pdot
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(force, np.zeros(3)))
aq = pin.aba(model, data, q0, v0, tau, fext)
assert(np.linalg.norm(aq - fdyn(model, data, frameId, q0, v0, tau, Kp, Kv, P_ANCHOR)) <1e-4)

# Compute the derivative of ABA
aba_dq, ada_dv, aba_dtau = pin.computeABADerivatives(model, data, q0, v0, tau, fext)

# Compute the derivative of the contact force
# pin.computeJointKinematicHessians(model, data, q0) 
dJ_dq = pin.getJointKinematicHessian(model, data, model.frames[frameId].parent, pin.LOCAL_WORLD_ALIGNED)

da_dq = aba_dq + lJ
# assert(np.linalg.norm(lda0_dx - contactCalcDiff2Bis(model, data, frameId, x0, pin.LOCAL))<1e-4)
# assert(np.linalg.norm(ldk_dx[-nc:] - lda0_dx) <1e-4)
# assert(np.linalg.norm(ldk_dx[:nv] - ldrnea_dx) <1e-4)
# assert(np.linalg.norm(ldaf_dx_ND + lKinv @ np.concatenate([ldrnea_dx, lda0_dx])) <1e-3)



# # From Gabriele
# class DAMSoftContactDynamics(crocoddyl.DifferentialActionModelFreeFwdDynamics):
#     '''
#         Computes the forward dynamics under visco-elastic (spring damper) force
#     '''
#     def __init__(self, diffModel, dt=1e-3, withCostResiduals=True, f_c = np.NaN):
#             '''
#                 If f_c is undefined or NaN, it is assumed to be infinite, unfiltered case
#             '''
#             crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(diffModel.state.nx + diffModel.nu), diffModel.nu)
#             self.differential = diffModel
#             self.dt = dt
#             self.withCostResiduals = withCostResiduals
#             # Set LPF cut-off frequency
#             self.set_alpha(f_c)
#             self.nx = diffModel.state.nx
#             self.ny = self.nu + self.nx
#             # Integrate or not?
#             if self.dt == 0:
#                 self.enable_integration_ = False
#             else:
#                 self.enable_integration_ = True
#             # Default unfiltered control cost (reg + lim)
#             self.set_w_reg_lim_costs(1e-2, 
#                                      np.zeros(self.differential.nu), 
#                                      1e-1,
#                                      np.zeros(self.differential.nu))

#     def set_w_reg_lim_costs(self, w_reg_weight, w_reg_ref, w_lim_weight, w_lim_ref):
#         '''
#         Set cost on unfiltered input
#         '''
#         self.w_reg_weight = w_reg_weight
#         self.w_reg_ref = w_reg_ref
#         self.w_lim_weight = w_lim_weight
#         self.w_lim_ref = w_lim_ref
#         self.activation = crocoddyl.ActivationModelQuadraticBarrier(
#                     crocoddyl.ActivationBounds(-self.differential.state.pinocchio.effortLimit, 
#                                                 self.differential.state.pinocchio.effortLimit) )

#     def createData(self):
#         '''
#             The data is created with a custom data class that contains the filtered torque tau_plus and the activation
#         '''
#         data = IntegratedActionDataLPF(self)
#         return data

#     def set_alpha(self, f_c = None):
#         '''
#             Sets the parameter alpha according to the cut-off frequency f_c
#             alpha = 1 / (1 + 2pi dt f_c)
#         '''
#         if f_c > 0 and self.dt > 0:
#             omega = 1/(2 * np.pi * self.dt * f_c)
#             self.alpha = omega/(omega + 1)
#         else:
#             self.alpha = 0

#     def calc(self, data, y, w = None):
#         '''
#         Euler integration (or no integration depending on dt)
#         '''
#         # what if w is none?
#         x = y[:self.differential.state.nx]
#         # filtering the torque with the previous state : get tau_q+ from w 
#         data.tau_plus[:] = self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w
#         # print("Data.tau_plus = ", data.tau_plus[0])
#         # dynamics : get a_q = DAM(q, vq, tau_q+)
#         self.differential.calc(data.differential, x, data.tau_plus)
#         if self.withCostResiduals:
#             data.r = data.differential.r
#         # Euler integration step of dt : get v_q+, q+
#         if self.enable_integration_:
#             data.cost = self.dt * data.differential.cost
#             # adding the cost on the unfiltered torque
#             self.activation.calc(data.activation, w - self.w_lim_ref)
#             data.cost += self.dt * self.w_lim_weight * data.activation.a_value + self.dt * (w - self.w_reg_ref) @ ( w - self.w_reg_ref ) / 2 * self.w_reg_weight
#             data.dx = np.concatenate([x[self.differential.state.nq:] * self.dt + data.differential.xout * self.dt**2, data.differential.xout * self.dt])
#             data.xnext[:self.nx] = self.differential.state.integrate(x, data.dx)
#             data.xnext[self.nx:] = data.tau_plus
#         else:
#             data.dx = np.zeros(len(y))
#             data.xnext[:] = y
#             data.cost = data.differential.cost
#             # adding the cost on the unfiltered torque
#             self.activation.calc(data.activation, w - self.w_lim_ref)
#             data.cost += self.w_lim_weight * data.activation.a_value + (w - self.w_reg_ref) @ ( w - self.w_reg_ref ) / 2 * self.w_reg_weight

#         return data.xnext, data.cost

#     def calcDiff(self, data, y, w=None):
#         '''
#         Compute derivatives 
#         '''
#         # First call calc
#         self.calc(data, y, w)
#         x = y[:-self.differential.nu]
#         # Get derivatives of DAM under LP-Filtered input 
#         self.differential.calcDiff(data.differential, x, data.tau_plus)
#         # Get d(IAM)/dx =  [d(q+)/dx, d(v_q+)/dx] 
#         dxnext_dx, dxnext_ddx = self.differential.state.Jintegrate(x, data.dx)
#         # Get d(DAM)/dx , d(DAM)/du (why resize?)
#         da_dx, da_du = data.differential.Fx, np.resize(data.differential.Fu, (self.differential.state.nv, self.differential.nu))
#         ddx_dx = np.vstack([da_dx * self.dt, da_dx])
#         # ??? ugly way of coding identity matrix ?
#         ddx_dx[range(self.differential.state.nv), range(self.differential.state.nv, 2 * self.differential.state.nv)] += 1
#         ddx_du = np.vstack([da_du * self.dt, da_du])

#         # In this scope the data.* are in the augmented state coordinates
#         # while all the differential dd are in the canonical x coordinates
#         # we must set correctly the quantities where needed
#         Fx = dxnext_dx + self.dt * np.dot(dxnext_ddx, ddx_dx)
#         Fu = self.dt * np.dot(dxnext_ddx, ddx_du) # wrong according to NUM DIFF, no timestep

#         # TODO why is this not multiplied by timestep?
#         data.Fx[:self.nx, :self.nx] = Fx
#         data.Fx[:self.nx, self.nx:self.ny] = self.alpha * Fu
#         data.Fx[self.nx:, self.nx:] = self.alpha * np.eye(self.nu)
#         # print('Fy : ', data.Fx)
#         # TODO CHECKING WITH NUMDIFF, NO TIMESTEP HERE
#         if self.nu == 1:
#             data.Fu.flat[:self.nx] = (1 - self.alpha) * Fu
#             data.Fu.flat[self.nx:] = (1 - self.alpha) * np.eye(self.nu)
#         else:
#             data.Fu[:self.nx, :self.nu] = (1 - self.alpha) * Fu
#             data.Fu[self.nx:, :self.nu] = (1 - self.alpha) * np.eye(self.nu)

#         if self.enable_integration_:

#             data.Lx[:self.nx] = self.dt * data.differential.Lx
#             data.Lx[self.nx:] = self.dt * self.alpha * data.differential.Lu

#             data.Lu[:] = self.dt * (1 - self.alpha) * data.differential.Lu

#             data.Lxx[:self.nx,:self.nx] = self.dt * data.differential.Lxx
#             # TODO reshape is not the best, see better how to cast this
#             data.Lxx[:self.nx,self.nx:] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
#             data.Lxx[self.nx:,:self.nx] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
#             data.Lxx[self.nx:,self.nx:] = self.dt * self.alpha**2 * data.differential.Luu

#             data.Lxu[:self.nx] = self.dt * (1 - self.alpha) * data.differential.Lxu
#             data.Lxu[self.nx:] = self.dt * (1 - self.alpha) * self.alpha * data.differential.Luu

#             data.Luu[:, :] = self.dt * (1 - self.alpha)**2 * data.differential.Luu

#             # adding the unfiltered torque cost
#             self.activation.calcDiff(data.activation, w - self.w_lim_ref)
#             data.Lu[:] += self.dt * self.w_lim_weight * data.activation.Ar + (w - self.w_reg_ref) * self.dt * self.w_reg_weight
#             data.Luu[:, :] += self.dt * self.w_lim_weight * data.activation.Arr + np.diag(np.ones(self.nu)) * self.dt * self.w_reg_weight

#         else:

#             data.Lx[:self.nx] = data.differential.Lx
#             data.Lx[self.nx:] = self.alpha * data.differential.Lu

#             data.Lu[:] = (1 - self.alpha) * data.differential.Lu

#             data.Lxx[:self.nx,:self.nx] = data.differential.Lxx
#             data.Lxx[:self.nx,self.nx:] = self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
#             data.Lxx[self.nx:,:self.nx] = self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
#             data.Lxx[self.nx:,self.nx:] = self.alpha**2 * data.differential.Luu

#             data.Lxu[:self.nx] = (1 - self.alpha) * data.differential.Lxu
#             data.Lxu[self.nx:] = (1 - self.alpha) * self.alpha * data.differential.Luu

#             data.Luu[:, :] = (1 - self.alpha)**2 * data.differential.Luu

#             # adding the unfiltered torque cost
#             self.activation.calcDiff(data.activation, w - self.w_lim_ref)
#             data.Lu[:] += self.w_lim_weight * data.activation.Ar + (w - self.w_reg_ref) * self.w_reg_weight
#             data.Luu[:, :] += self.w_lim_weight * data.activation.Arr + np.diag(np.ones(self.nu)) * self.w_reg_weight


# class IntegratedActionDataLPF(crocoddyl.ActionDataAbstract):
#     '''
#     Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
#     '''
#     def __init__(self, am):
#         crocoddyl.ActionDataAbstract.__init__(self, am)
#         self.differential = am.differential.createData()
#         self.activation = am.activation.createData()
#         self.tau_plus = np.zeros(am.nu)
#         self.Fx = np.zeros((am.ny, am.ny))
#         self.Fu = np.zeros((am.ny, am.nu))
#         self.Lx = np.zeros(am.ny)
#         self.Lu = np.zeros(am.nu)
#         self.Lxx = np.zeros((am.ny, am.ny))
#         self.Lxu = np.zeros((am.ny, am.nu))
#         self.Luu = np.zeros((am.nu,am.nu))