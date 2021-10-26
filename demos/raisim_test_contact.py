import numpy as np
import time
from utils import raisim_utils, path_utils, pin_utils
np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_ocp')
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
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
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

# env.step()
# Compute desired external force acting on each joint 
f_ext = []
for i in range(nq+1):
    # CONTACT --> WORLD
    W_X_ct = contact_placement.action
    # WORLD --> JOINT
    j_X_W  = robot.data.oMi[i].actionInverse
    # CONTACT --> JOINT
    j_X_ee = W_X_ct.dot(j_X_W)
    # ADJOINT INVERSE (wrenches)
    f_joint = j_X_ee.T.dot([0., 0., -1, 0., 0., 0.])
    f_ext.append(pin.Force(f_joint))

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