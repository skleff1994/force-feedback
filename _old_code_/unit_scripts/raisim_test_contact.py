import numpy as np
import time
from core_mpc import raisim_utils, pin_utils
np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Load Kuka config from URDF
urdf_path = "/home/skleff/robot_properties_kuka_RAISIM/iiwa_test.urdf"
mesh_path = "/home/skleff/robot_properties_kuka_RAISIM"
iiwa_config = raisim_utils.IiwaMinimalConfig(urdf_path, mesh_path)

# Load Raisim environment
LICENSE_PATH = '/home/skleff/.raisim/activation.raisim'
env = raisim_utils.RaiEnv(LICENSE_PATH, dt=1e-3)
robot = env.add_robot(iiwa_config, init_config=None)
env.launch_server()

# Initialize simulation
q0 = np.asarray([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.asarray([0.,0.,0.,0.,0.,0.,0.] )
x0 = np.concatenate([q0, v0])   
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv; nu = nq
# Update robot model with initial state
robot.reset_state(q0, v0)
robot.forward_robot(q0, v0)
print(robot.get_state())
M_ee = robot.data.oMf[id_endeff]
print("Initial placement : \n")
print(M_ee)

# Add a ball close to EE
contact_placement = robot.data.oMf[id_endeff].copy()
offset = 0.035
contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
import pinocchio as pin
ball = env.display_contact_surface(contact_placement, radius=0.1)

# Compute desired external force acting on each joint 
f_ext = []
for i in range(nq+1):
    # CONTACT --> WORLD
    W_M_ct = contact_placement.copy()
    f_WORLD = W_M_ct.actionInverse.T.dot(np.asarray([0.,0.,-1,0.,0.,0.] ))
    # WORLD --> JOINT
    j_M_W = robot.data.oMi[i].copy().inverse()
    f_JOINT = j_M_W.actionInverse.T.dot(f_WORLD)
    f_ext.append(pin.Force(f_JOINT))
ug = pin_utils.get_u_grav(q0, robot)
u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model)
print("ug = ", ug)
print("u0 = ", u0)

# step 
for i in range(10000):
  robot.send_joint_command(u0)
  env.step()
  time.sleep(0.01)
  print(robot.get_contact_forces())

# Check contacts
# for c in robot.get_contact_points():
#   print("local body idx : ", c.getlocalBodyIndex())
#   print("normal         : ", c.getNormal())
#   print("frame          : \n", c.getContactFrame())
#   print("Impulse        : ", c.getImpulse())

# print(robot.get_contact_forces())

time.sleep(1000)