import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin


# From Gabriele
class DAMSoftContactDynamics(crocoddyl.DifferentialActionModelFreeFwdDynamics):
    '''
    Computes the forward dynamics under visco-elastic (spring damper) force
    '''
    def __init__(self, stateMultibody, actuationModel, costModelSum, frameId, Kp=10, Kv=1, pinRefFrame=pin.LOCAL):
        '''
            If f_c is undefined or NaN, it is assumed to be infinite, unfiltered case
        '''
        crocoddyl.DifferentialActionModelFreeFwdDynamics.__init__(self, stateMultibody, actuationModel, costModelSum)
        self.state = stateMultibody
        self.actuation = actuationModel
        self.costs = costModelSum
        self.Kp = Kp 
        self.Kv = Kv
        self.pinRef = pinRefFrame
        self.frameId = frameId
        # self.model = self.state.pinocchio
        # self.data = self.state.createData()
        # hard coded costs on force ?
        # self.set_w_reg_lim_costs(1e-2, 
        #                             np.zeros(self.differential.nu), 
        #                             1e-1,
        #                             np.zeros(self.differential.nu))

    # def set_w_reg_lim_costs(self, w_reg_weight, w_reg_ref, w_lim_weight, w_lim_ref):
    #     '''
    #     Set cost on unfiltered input
    #     '''
    #     self.w_reg_weight = w_reg_weight
    #     self.w_reg_ref = w_reg_ref
    #     self.w_lim_weight = w_lim_weight
    #     self.w_lim_ref = w_lim_ref
    #     self.activation = crocoddyl.ActivationModelQuadraticBarrier(
    #                 crocoddyl.ActivationBounds(-self.differential.state.pinocchio.effortLimit, 
    #                                             self.differential.state.pinocchio.effortLimit) )

    def createData(self):
        '''
            The data is created with a custom data class that contains the filtered torque tau_plus and the activation
        '''
        data = DADSoftContactDynamics(self)
        return data

    def calc(self, data, x, u):
        '''
        Compute joint acceleration and costs 
         using visco-elastic force
        '''
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        # pin.computeAllTerms(model, data, q, v)
        # pin.forwardKinematics(model, data, q, v, np.zeros(nq))
        # pin.updateFramePlacements(model, data)
        # Compute visco-elastic contact force 

        oRf = data.oMf[self.frameId].rotation
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
        # what if w is none?
        x = y[:self.differential.state.nx]
        # filtering the torque with the previous state : get tau_q+ from w 
        data.tau_plus[:] = self.alpha * y[-self.differential.nu:] + (1 - self.alpha) * w
        # print("Data.tau_plus = ", data.tau_plus[0])
        # dynamics : get a_q = DAM(q, vq, tau_q+)
        self.differential.calc(data.differential, x, data.tau_plus)
        if self.withCostResiduals:
            data.r = data.differential.r
        # Euler integration step of dt : get v_q+, q+
        if self.enable_integration_:
            data.cost = self.dt * data.differential.cost
            # adding the cost on the unfiltered torque
            self.activation.calc(data.activation, w - self.w_lim_ref)
            data.cost += self.dt * self.w_lim_weight * data.activation.a_value + self.dt * (w - self.w_reg_ref) @ ( w - self.w_reg_ref ) / 2 * self.w_reg_weight
            data.dx = np.concatenate([x[self.differential.state.nq:] * self.dt + data.differential.xout * self.dt**2, data.differential.xout * self.dt])
            data.xnext[:self.nx] = self.differential.state.integrate(x, data.dx)
            data.xnext[self.nx:] = data.tau_plus
        else:
            data.dx = np.zeros(len(y))
            data.xnext[:] = y
            data.cost = data.differential.cost
            # adding the cost on the unfiltered torque
            self.activation.calc(data.activation, w - self.w_lim_ref)
            data.cost += self.w_lim_weight * data.activation.a_value + (w - self.w_reg_ref) @ ( w - self.w_reg_ref ) / 2 * self.w_reg_weight

        return data.xnext, data.cost

    def calcDiff(self, data, y, w=None):
        '''
        Compute derivatives 
        '''
        # First call calc
        self.calc(data, y, w)
        x = y[:-self.differential.nu]
        # Get derivatives of DAM under LP-Filtered input 
        self.differential.calcDiff(data.differential, x, data.tau_plus)
        # Get d(IAM)/dx =  [d(q+)/dx, d(v_q+)/dx] 
        dxnext_dx, dxnext_ddx = self.differential.state.Jintegrate(x, data.dx)
        # Get d(DAM)/dx , d(DAM)/du (why resize?)
        da_dx, da_du = data.differential.Fx, np.resize(data.differential.Fu, (self.differential.state.nv, self.differential.nu))
        ddx_dx = np.vstack([da_dx * self.dt, da_dx])
        # ??? ugly way of coding identity matrix ?
        ddx_dx[range(self.differential.state.nv), range(self.differential.state.nv, 2 * self.differential.state.nv)] += 1
        ddx_du = np.vstack([da_du * self.dt, da_du])

        # In this scope the data.* are in the augmented state coordinates
        # while all the differential dd are in the canonical x coordinates
        # we must set correctly the quantities where needed
        Fx = dxnext_dx + self.dt * np.dot(dxnext_ddx, ddx_dx)
        Fu = self.dt * np.dot(dxnext_ddx, ddx_du) # wrong according to NUM DIFF, no timestep

        # TODO why is this not multiplied by timestep?
        data.Fx[:self.nx, :self.nx] = Fx
        data.Fx[:self.nx, self.nx:self.ny] = self.alpha * Fu
        data.Fx[self.nx:, self.nx:] = self.alpha * np.eye(self.nu)
        # print('Fy : ', data.Fx)
        # TODO CHECKING WITH NUMDIFF, NO TIMESTEP HERE
        if self.nu == 1:
            data.Fu.flat[:self.nx] = (1 - self.alpha) * Fu
            data.Fu.flat[self.nx:] = (1 - self.alpha) * np.eye(self.nu)
        else:
            data.Fu[:self.nx, :self.nu] = (1 - self.alpha) * Fu
            data.Fu[self.nx:, :self.nu] = (1 - self.alpha) * np.eye(self.nu)

        if self.enable_integration_:

            data.Lx[:self.nx] = self.dt * data.differential.Lx
            data.Lx[self.nx:] = self.dt * self.alpha * data.differential.Lu

            data.Lu[:] = self.dt * (1 - self.alpha) * data.differential.Lu

            data.Lxx[:self.nx,:self.nx] = self.dt * data.differential.Lxx
            # TODO reshape is not the best, see better how to cast this
            data.Lxx[:self.nx,self.nx:] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.dt * self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.dt * self.alpha**2 * data.differential.Luu

            data.Lxu[:self.nx] = self.dt * (1 - self.alpha) * data.differential.Lxu
            data.Lxu[self.nx:] = self.dt * (1 - self.alpha) * self.alpha * data.differential.Luu

            data.Luu[:, :] = self.dt * (1 - self.alpha)**2 * data.differential.Luu

            # adding the unfiltered torque cost
            self.activation.calcDiff(data.activation, w - self.w_lim_ref)
            data.Lu[:] += self.dt * self.w_lim_weight * data.activation.Ar + (w - self.w_reg_ref) * self.dt * self.w_reg_weight
            data.Luu[:, :] += self.dt * self.w_lim_weight * data.activation.Arr + np.diag(np.ones(self.nu)) * self.dt * self.w_reg_weight

        else:

            data.Lx[:self.nx] = data.differential.Lx
            data.Lx[self.nx:] = self.alpha * data.differential.Lu

            data.Lu[:] = (1 - self.alpha) * data.differential.Lu

            data.Lxx[:self.nx,:self.nx] = data.differential.Lxx
            data.Lxx[:self.nx,self.nx:] = self.alpha * np.reshape(data.differential.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.alpha * np.reshape(data.differential.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.alpha**2 * data.differential.Luu

            data.Lxu[:self.nx] = (1 - self.alpha) * data.differential.Lxu
            data.Lxu[self.nx:] = (1 - self.alpha) * self.alpha * data.differential.Luu

            data.Luu[:, :] = (1 - self.alpha)**2 * data.differential.Luu

            # adding the unfiltered torque cost
            self.activation.calcDiff(data.activation, w - self.w_lim_ref)
            data.Lu[:] += self.w_lim_weight * data.activation.Ar + (w - self.w_reg_ref) * self.w_reg_weight
            data.Luu[:, :] += self.w_lim_weight * data.activation.Arr + np.diag(np.ones(self.nu)) * self.w_reg_weight


class DADSoftContactDynamics(crocoddyl.ActionDataAbstract):
    '''
    Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
    '''
    def __init__(self, am):
        crocoddyl.ActionDataAbstract.__init__(self, am)
        self.differential = am.differential.createData()
        self.activation = am.activation.createData()
        self.tau_plus = np.zeros(am.nu)
        self.Fx = np.zeros((am.ny, am.ny))
        self.Fu = np.zeros((am.ny, am.nu))
        self.Lx = np.zeros(am.ny)
        self.Lu = np.zeros(am.nu)
        self.Lxx = np.zeros((am.ny, am.ny))
        self.Lxu = np.zeros((am.ny, am.nu))
        self.Luu = np.zeros((am.nu,am.nu))