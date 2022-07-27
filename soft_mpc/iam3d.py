from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin



class StateSoftContact3D(crocoddyl.StateAbstract):
    def __init__(self, rmodel, nc):
        crocoddyl.StateAbstract.__init__(self, rmodel.nq + rmodel.nv, 2*rmodel.nv)
        self.pinocchio = self.rmodel
        self.nc = nc
        self.ny = self.nx + self.nc
        self.ndy = self.ndx + self.nc
        print("Augmented state ny = ", self.ny)
        print("Augmented state ndy = ", self.ndy)

    def diff(self, y0, y1):
        yout = np.zeros(self.ny)
        nq = self.pinocchio.nq
        # nv = self.pinocchio.nv
        yout[:nq] = pin.difference(self.pinocchio, y0[:nq], y1[:nq])
        yout[nq:] = y1[nq:] - y0[nq:]
        return yout

    def integrate(self, y, dy):
        yout = np.zeros(self.ndy)
        nq = self.pinocchio.nq
        # nv = self.pinocchio.nv
        yout[:nq] = pin.integrate(self.pinocchio, y[:nq], dy[:nq])
        yout[nq:] = y[nq:] + dy[nq:]
        return yout



class IAMSoftContactDynamics3D(crocoddyl.ActionModelAbstract):
    def __init__(self, dam, dt=1e-3, withCostResidual=True):
        crocoddyl.ActionModelAbstract(dam.state, dam.nu, dam.nr + 3)
        self.state = StateSoftContact3D(dam.pinocchio, 3)
        self.dt = dt
        self.withCostResidual = withCostResidual

    def createData(self):
        data = IADSoftContactDynamics(self)
        return data
    
    def calc(self, data, y, u):
        nx = self.differential.state.nx
        nv = self.differential.state.nv
        x = y[:nx]
        # q = x[:self.state.nq]
        v = x[self.state.nq:]
        # self.control.calc(data.control, 0., u)
        self.differential.calc(data.differential, x, u) #data.control.w)
        a = data.differential.xout
        data.dx[:nv] = v*self.dt + a*self.dt**2
        data.dx[nv:2*nv] = a*self.dt
        data.dx[-nv:] = a*self.dt
        data.cost = self.dt*data.differential.cost
        if(self.withCostResidual):
            data.r = data.differential.r

    def calcDiff(self, data, y, u):
        nx = self.differential.state.nx
        ndx = self.differential.state.ndx
        nv = self.differential.state.nv
        nu = self.differential.nu
        nc = self.state.nc
        x = y[:nx]

        # Calc forward dyn derivatives
        self.differential.calcDiff(data.differential, x, u)
        da_dx = data.differential.Fx 
        da_du = data.differential.Fu

        # Fill out blocks
        # d->Fx.topRows(nv).noalias() = da_dx * time_step2_;
        data.Fy[:nv,:ndx] = da_dx*self.dt**2
       
        # d->Fx.bottomRows(nv).noalias() = da_dx * time_step_;
        data.Fy[nv:ndx, :ndx] = da_dx*self.dt
        
        # d->Fx.topRightCorner(nv, nv).diagonal().array() += Scalar(time_step_);
        data.Fy[:nv, nv:ndx] += self.dt * np.eye(nv)
        
        # d->Fu.topRows(nv).noalias() = time_step2_ * d->da_du;
        data.Fw[:nv, :] = da_du * self.dt**2
        
        # d->Fu.bottomRows(nv).noalias() = time_step_ * d->da_du;
        data.Fw[nv:ndx, :] = da_du * self.dt
        
        # New block from augmented dynamics (top right corner)
        data.Fy[:nv, -nc:] = data.differential.dABA_df * self.dt**2
        data.Fy[nv:ndx, -nc:] = data.differential.dABA_df * self.dt
        # New block from augmented dynamics (bottom right corner)
        data.Fy[-nc:,-nc:] = np.eye(3) + data.differential.dfdt_df*self.dt
        # New block from augmented dynamics (bottom left corner)
        data.Fy[-nc:, :ndx] = data.differential.dfdt_dx * self.dt
        
        data.Fw[-nc:, :] = data.differential.dfdt_du * self.dt

        # state_->JintegrateTransport(x, d->dx, d->Fx, second);
        self.state.JintegrateTransport(y, data.dy, data.Fy, crocoddyl.Jcomponent.second)
        
        # state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
        self.state.Jintegrate(y, data.dy, data.Fy,  data.Fy, crocoddyl.addto)  # add identity to Fx = d(x+dx)/dx = d(q,v)/d(q,v)
        # data.Fy (nu, nu).diagonal().array() -=
        #     Scalar(1.);  // remove identity from Ftau (due to stateLPF.Jintegrate)
        
        # state_->JintegrateTransport(x, d->dx, d->Fu, second);
        self.state.JintegrateTransport(y, data.dy, data.Fw, crocoddyl.Jcomponent.second)


        # d->Lx.noalias() = time_step_ * d->differential->Lx;

        data.Ly[:ndx] = data.differential.Lx
        data.Ly[-nc:] = data.differential.df_du
        # control_->multiplyJacobianTransposeBy(d->control, d->differential->Lu, d->Lu);
        # d->Lu *= time_step_;
        # d->Lxx.noalias() = time_step_ * d->differential->Lxx;
        # control_->multiplyByJacobian(d->control, d->differential->Lxu, d->Lxu);
        # d->Lxu *= time_step_;
        # control_->multiplyByJacobian(d->control, d->differential->Luu, d->Lwu);
        # control_->multiplyJacobianTransposeBy(d->control, d->Lwu, d->Luu);
        # d->Luu *= time_step_;


class IADSoftContactDynamics(crocoddyl.IntegratedActionDataAbstract):
    '''
    Creates a data class for IAM
    '''
    def __init__(self, am):
        # super().__init__(am)
        crocoddyl.IntegratedActionDataAbstract.__init__(self, am)
        self.differential = am.differential.createData()
        self.xnext = np.zeros(am.state.ny)
        self.dx = np.zeros(am.ndy)
        self.Fy = np.zeros((am.state.ndy, am.state.ndy))
        self.Fw = np.zeros((am.state.ndy, am.nu))
        self.r = am.differential.nr
        self.Ly = np.zeros(am.state.ny)
        self.Lw = np.zeros(am.actuation.nu)
        self.Lyy = np.zeros((am.state.ny, am.state.ny))
        self.Lyw = np.zeros((am.state.ny, am.actuation.nu))
        self.Lww = np.zeros((am.actuation.nu, am.actuation.nu))

        # self.df_dx = np.zeros((am.nc, am.state.ny))   
        # self.df_dx_copy = np.zeros((am.nc, am.state.nx))   
        # self.f = np.zeros(am.nc)    
        # self.f_copy = np.zeros(am.nc)   
        # self.fext = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        # self.fext_copy = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]

        # self.pinocchio  = am.pinocchio.createData()
        # self.actuation_data = am.actuation.createData()
        # self.multibody = crocoddyl.DataCollectorActMultibody(self.pinocchio, self.actuation_data)
        # self.costs = am.costs.createData(crocoddyl.DataCollectorMultibody(self.pinocchio))

