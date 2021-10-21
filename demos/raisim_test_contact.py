import numpy as np
import time
from utils import raisim_utils, path_utils
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
ball = env.world.addSphere(.01,100, 'default', 1)
offset = np.array([0.,0.,0.08])
p = robot.rai_robot.getFramePosition('EE')
R = robot.rai_robot.getFrameOrientation('EE')
pb = p + R.dot(offset)
ball.setPosition(p)

# step 
env.step()
# Check contacts
for c in robot.get_contact_points():
  print("local body idx : ", c.getlocalBodyIndex())
  print("normal         : ", c.getNormal())
  print("frame          : \n", c.getContactFrame())
  print("Impulse        : ", c.getImpulse())
