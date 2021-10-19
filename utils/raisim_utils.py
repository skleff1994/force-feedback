## raisim env
## Author : Avadesh Meduri and Paarth Shah and Sébastien Kleff
## Date : 29/03/2021

import time
import numpy as np
import raisimpy as raisim
from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import pinocchio as pin



class IiwaMinimalConfig:
    def __init__(self, urdf_path, mesh_path):
        self.end_effector_names = ["contact"]
        self.motor_inertia = 0.0000045
        self.motor_gear_ratio = 9.0
        self.robot_name = "iiwa"
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.initial_configuration = [0.]*self.model.nq
        self.initial_velocity = [0.]*self.model.nv
        self.link_names =  ['iiwa_base', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']




class PinRaiRobotWrapper:

    def __init__(self, world, robot_config, init_config = None):
        """
        Input:
            world        : rai world
            robot_config : robot minimal config file inside robot properties
            urdf_path    : incase the robot urdf is in a different place
                Note :its best to use the urdf defined in the robot minimal config file
        """

        self.world = world
        #Set up Pinocchio data
        self.model = robot_config.model 
        self.data = self.model.createData()

        #Set up robot model information
        self.ee_names = robot_config.end_effector_names
        self.end_eff_ids = [] 
        self.nb_ee = len(self.ee_names)
        self.nb_dof = self.model.nv #- 6
        
        ## TODO : Has to be general
        if robot_config.robot_name == "iiwa":
            self.body_names = robot_config.link_names
        else:
            self.body_names = self.ee_names.copy()
        self.raisim_foot_idx = np.zeros(len(self.body_names))

        #Set up Raisim Articulated Body
        self.rai_robot = self.world.addArticulatedSystem(robot_config.urdf_path)
        if isinstance(init_config, (np.ndarray)):
            rq = init_config
            self.rai_robot.setGeneralizedCoordinate(rq)
        else:
            rq = robot_config.initial_configuration
            self.rai_robot.setGeneralizedCoordinate(rq)

        if robot_config:
            self.rai_robot.setName(robot_config.robot_name)
            self.model.rotorInertia[:] = robot_config.motor_inertia
            self.model.rotorGearRatio[:] = robot_config.motor_gear_ratio
            self.name = robot_config.robot_name
        else:
            self.rai_robot.setName("Raisim_Robot")
            self.name = "raisim_robot_wrapper"

        for i in range(0, self.raisim_foot_idx.size):
            self.raisim_foot_idx[i] = self.rai_robot.getBodyIdx(self.body_names[i])

        for i in range(len(self.ee_names)):
            # print(self.model.getFrameId(self.ee_names[i]))
            self.end_eff_ids.append(self.model.getFrameId(self.ee_names[i]))

    def reset_state(self,q, v):
        rq = np.concatenate([q,v])
        self.rai_robot.setGeneralizedCoordinate(rq)

    def get_state(self):
        """
        returns states of the robot based on the desired id matching pinocchio
        convention
        Input :
            robot_id : the robot whose state is desired
        """
        rq, rv = self.rai_robot.getState()
        
        # switching the quaternion convention
        return rq, rv

    def send_joint_command(self, tau):
        """
        Applies to torque to the robot
        Input:
            tau : torque to be applied to the robot
        """
        self.rai_robot.setControlMode(raisim.ControlMode.FORCE_AND_TORQUE)
        self.rai_robot.setGeneralizedForce(tau)

    def forward_robot(self, q=None, dq=None):
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)

    def get_contact_forces(self):
        """
        returns contact forces at each end effector
        """    
        F = np.zeros(3*self.nb_ee)
        for contact in self.rai_robot.getContacts():
            for i in range(self.nb_ee):
                if contact.getlocalBodyIndex() == int(self.raisim_foot_idx[i]):
                    F[3*i:3*(i+1)] += contact.getContactFrame().dot(contact.getImpulse())/self.world.getTimeStep()
        return F

    def get_current_contacts(self):
        """
        returns boolean array of which end-effectors are currently in contact
        """
        contact_config = np.zeros(self.nb_ee)
        for contact in self.rai_robot.getContacts():
            for i in range(self.nb_ee):
                if contact.getlocalBodyIndex() == self.raisim_foot_idx[i]:
                    contact_config[i] = 1.0
        return contact_config

    def set_state(self, q, v):
        """
        set state for visualization purposes in raisim
        """
        self.rai_robot.setState(q,v)

    def get_ee_positions(self, q, v):
        ee_positions = np.zeros(self.nb_ee*3)
        pin.forwardKinematics(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeCentroidalMomentum(self.model, self.data, q, v)
        for i in range(len(self.end_eff_ids)):
            #print(self.data.oMf[self.end_eff_ids[i]].translation)
            ee_positions[i*3:i*3 + 3 ] = self.data.oMf[self.end_eff_ids[i]].translation
        return ee_positions





class RaiEnv:

    def __init__(self, LICENSE_PATH=None, dt=None):

        if(LICENSE_PATH is None):
          print("ERROR: please specify Raisim license path !")
        raisim.World.setLicenseFile(LICENSE_PATH)
        self.robots = []
        # Raisim Configuration
        self.world = raisim.World()
        if(dt is not None):
            self.dt = dt
        else:
            self.dt = 1e-3
        self.world.setTimeStep(self.dt)
        self.server = raisim.RaisimServer(self.world)
        self.ground = self.world.addGround()
        self.visualize_array = []

    def add_robot(self, robot_config=None, init_config = None):
        """
        Adds robot into the raisim world
        Input:
            robot_config : robot config file inside robot properties
            urdf_path : incase the robot urdf is in a different place
                Note :its best to use the urdf defined in the robot config file
        """
        robot = PinRaiRobotWrapper(self.world, robot_config, init_config=init_config)
        self.robots.append(robot)

        return robot

    def launch_server(self):
        self.server.launchServer(8080)
        time.sleep(2)

    def step(self, sleep = False):
        """
        Integrates the simulation
        """
        if sleep:
            s = time.time()
            self.server.integrateWorldThreadSafe()
            e = time.time()
            dt_taken = e - s
            if dt_taken < self.dt:
                time.sleep(self.dt - dt_taken)
            else:
                print("Warning : simulation is slower than real time")
        else:
            self.server.integrateWorldThreadSafe()

    def create_height_map(self, size, samples, terrain_height):
        """
        Creates height map in RAISIM.
        Creates at origin: 0.0, 0.0
        Input:
            size: (double) how big the map should be 
            samples: (int) how many times each length of the map should be divided into
            terrain_height: (matrix of doubles)
                            matrix of information relating to the desired shapes
                            matrix dimensions: (size/samples) x (size/samples)

        """

        self.world.removeObject(self.ground)

        self.height_map = self.world.addHeightMap(samples, samples, + \
            size, size, 0.0, 0.0, terrain_height.flatten().tolist())
        height_map  = self.height_map
        return height_map

    def create_height_map_perlin(self, raisimTerrain):
        """
        Creates height map in RAISIM.
        Creates at origin: 0.0, 0.0
        Input:
            raisimTerrain (class: raisim.TerrainProperties)
            see: https://raisim.com/sections/HeightMap_example_terrain_generator.html#c-way for details
            or: dir(raisimTerrain) to see options
        """
        self.world.removeObject(self.ground)

        self.height_map = self.world.addHeightMap(0.0, 0.0, raisimTerrain)
        height_map  = self.height_map
        return height_map

    def create_height_map_png(self, x_center, y_center, path_to_png, size, scale, z_offset):
        self.world.removeObject(self.ground)

        self.height_map = self.world.addHeightMap(path_to_png, x_center, y_center, size, size, scale, z_offset)
        height_map  = self.height_map
        return height_map

    def get_terrain_rot_matrix(self, x, y):
        if (self.height_map):
            contact_normal = self.height_map.getNormal(x,y)
            zUp = np.array([0,0,1.0])

            #Get axis between zUp and contact_normal
            axis = np.cross(zUp, contact_normal)

            #Check if vectors are colinear
            if (norm(axis) < 1e-3):
                return np.identity(3)

            axis = axis / norm(axis)

            #Get angle between [0, 0, 1] and contact_normal
            dot_product = np.dot(zUp, contact_normal)/norm(contact_normal)
            theta = np.arccos(np.clip(dot_product, -1.0, 1.0))

            #Create rotation vector
            rot = Rotation.from_rotvec(theta*axis)

            #Return as rotation matrix
            return rot.as_matrix()
        else:
            return np.identity(3)

    def get_terrain_height(self, x, y):
        if (self.height_map):
            return self.height_map.getHeight(x,y)
        else:
            return 0.0

    def visualize_contacts(self, vis_array, radius = 0.175):
        """
        Goes through N-Dimensional array of points to visualize
        information
        Input: N x 3 Dimensional matrix of contact points
        """

        if vis_array.shape[0] > len(self.visualize_array):
            for i in range(vis_array.shape[0]-len(self.visualize_array)):
                self.visualize_array.append(self.server.addVisualSphere("sphere" + str(len(self.visualize_array)), radius, 1, 0, 0, 1))

        elif vis_array.shape[0] < len(self.visualize_array):
            for i in range(len(self.visualize_array) - vis_array.shape[0]):
                self.visualize_array[-1*(i+1)].setPosition((0,0,100))

        for i in range(vis_array.shape[0]):
            self.visualize_array[i].setPosition((vis_array[i]))

    def get_terrain_info(self, position):
        """
        Ray Test to get orientation and position of terrain at position (x,y,z)
        input: Position (x, y, z)
        returns: Position (with modified height of ray test collision), Orientation (rotation matrix)
        """

        #Ray
        direction = [0.0, 0.0, -50.0]
        elevated_pos = position.copy()
        elevated_pos[2] += 5.0

        #Ray Test
        col = self.world.rayTest(elevated_pos, direction, 50)
        if col.size() > 0:
            collisionObject = col.at(0).getObject()
            collisionPos = col.at(0).getPosition()
            collisionOrientation = collisionObject.getOrientation(0)
            return collisionPos, collisionOrientation
        else:
            return position, np.identity(3)



# Load KUKA arm in Raisim environment
def init_kuka_RAISIM(dt=1e3, x0=None):
    '''
    Loads KUKA LBR iiwa model in Raisim using the 
    Pinocchio-Raisim wrapper to simplify interactions
    '''
    # Create Raisim sim environment + initialize sumulator
    urdf_path = "/home/skleff/robot_properties_kuka_RAISIM/iiwa.urdf"
    mesh_path = "/home/skleff/robot_properties_kuka_RAISIM"
    iiwa_config = IiwaMinimalConfig(urdf_path, mesh_path)

    # Load Raisim environment
    LICENSE_PATH = '/home/skleff/.raisim/activation.raisim'
    env = RaiEnv(LICENSE_PATH)
    robot = env.add_robot(iiwa_config, init_config=None)
    env.launch_server()
    #Raisim parameters for forward prediction
    env.world.setTimeStep(dt)
    # Initialize
    if(x0 is None):
        q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
        dq0 = np.zeros(robot.model.nv)
    else:
        q0 = x0[:robot.model.nq]
        dq0 = x0[robot.pin_robot.model.nv:]
    robot.reset_state(q0, dq0)
    robot.forward_robot(q0, dq0)
    return env, robot