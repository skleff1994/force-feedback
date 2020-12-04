"""
@package ddp_iiwa
@file ddp_iiwa/example_without_pybullet.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2019, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Simple self-contained example of DDP trajectory for KUKA - without PyBullet 
"""

import numpy as np 
import pinocchio as pin
from py_ddp_iiwa.ddp_planner import DDPPlanner
from pinocchio.robot_wrapper import RobotWrapper
import rospkg
import yaml
import os

############################################
### ROBOT MODEL & SIMULATION ENVIRONMENT ###
############################################
urdf_path = os.path.join(rospkg.RosPack().get_path("robot_properties_iiwa"), "urdf", "iiwa.urdf")
pin_robot = RobotWrapper(pin.buildModelFromUrdf(urdf_path))
id_endeff = pin_robot.model.getFrameId('contact')
nq = pin_robot.model.nq 
nv = pin_robot.model.nv
# Hard-coded initial state of StartDGM application in KUKA sunrise control panel
q0 = np.array([0., 0.349066, 0., -0.872665, 0., 0., 0.])  
dq0 = pin.utils.zero(nv)
# Update pinocchio model with forward kinematics and get frame initial placemnent + desired frame placement
pin.forwardKinematics(pin_robot.model, pin_robot.data, q0)
pin.updateFramePlacements(pin_robot.model, pin_robot.data)
M0 = pin_robot.data.oMf[id_endeff]
p_target = M0.translation
p_target = np.array([0.5, 0.2, 0.5])
M_target = pin.SE3(M0.rotation, p_target)

###################
### DDP PLANNER ###
###################
# Integration step for DDP (s) 
dt = 50e-3               
# Number of knots in the MPC horizon 
N = 100           
# Planner
ddp_planner = DDPPlanner(pin_robot, dt, N)
# Costs weights
running_costs = [10., 1e-3, 1e-5, 1.]                 # endeff, xreg, ureg, xlim
terminal_costs = [100., 100., 1.]                     # endeff, xreg, xlim
state_weights = np.array([0.]*nq + [5.]*nv)           # size nq + nv
state_weights_term = np.array([0.]*nq + [100.]*nv)  # size nq + nv
frame_weights = np.array([1.] * 3 + [.1] * 3)     # size 6 (3 pos + 3 rot)

# Initialize OCP
x0 = np.concatenate([q0, dq0])
print("x0 : ", x0 )
ddp_planner.init_ocp(x0, M_target, running_costs, terminal_costs, state_weights, state_weights_term, frame_weights, interpolation=False)

import crocoddyl

ddp=ddp_planner.ddp
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
pb=ddp.problem
m=pb.runningModels[0].differential
# costs= m.costs.costs.todict()
mT = pb.terminalModel.differential

g = pin.rnea(pin_robot.model, pin_robot.data, pb.x0[:7], np.zeros(7), np.zeros(7))
for mr in pb.runningModels:
    #mr.differential.pinocchio.rotorGearRatio[:] = 1e2
    #mr.differential.pinocchio.rotorInertia[:] = 1e-5
    #mr.differential.pinocchio.armature = np.array([.1]*7)
    mr.differential.armature = np.array([.1]*7)
    
    mr.differential.costs.costs['ctrlReg'].weight=1e-4
    # mr.differential.costs.costs['ctrlReg'].cost.reference=g
    mr.differential.costs.costs['stateReg'].weight=1e-2
    mr.differential.costs.costs['stateReg'].cost.activation.weights=np.array([ 1 ]*7+[50]*7)
    mr.differential.costs.costs['endeff'].weight=1e-2
    mr.differential.costs.costs['stateLim'].weight=100
    mr.differential.costs.costs['endeff'].cost.reference=mT.costs.costs['endeff'].cost.reference
mT.costs.costs['endeff'].weight=1e4

ddp.solve([],[],maxiter=1000)
xs=np.array(ddp.xs)
us=np.array(ddp.us)
err = np.array([ rd.differential.costs.costs['endeff'].r for rd in pb.runningDatas ])

import matplotlib.pyplot as plt #; plt.ion()

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(xs[:,:7])
plt.ylabel('pos')
plt.subplot(3,1,2)
plt.plot(xs[:,7:])
plt.ylabel('vel')
plt.subplot(3,1,3)
plt.plot(us[:,:7]-g)
plt.ylabel('torques\n nograv')
       
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(err[:,:3])
plt.xlabel('time')
plt.ylabel('tracking errors (m)')
plt.subplot(2,1,2)
plt.plot(err[:,3:])
plt.xlabel('time')
plt.ylabel('tracking errors (rad)')


# stophere
ddp_planner.update_plan(xs, us)

# Plot inital plan
ddp_planner.plot()
