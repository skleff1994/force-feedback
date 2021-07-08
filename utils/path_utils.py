import importlib_resources
import yaml
import os

# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f)
    return data 

# Returns urdf path of a kuka robot 
def get_urdf_path(robot_name, robot_family='kuka'):
    # Get config file
    with importlib_resources.path("robot_properties_"+robot_family, "config.py") as p:
        pkg_dir = p.parent.absolute()
    urdf_path = pkg_dir/"robot_properties_kuka"/(robot_name + ".urdf")
    return str(urdf_path)

# Returns urdf path of a kuka robot 
def get_mesh_dir(robot_family='kuka'):
    # Get config file
    with importlib_resources.path("robot_properties_"+robot_family, "config.py") as p:
        pkg_dir = p.parent.absolute()
    urdf_dir = pkg_dir
    return str(urdf_dir)

# Load config file
def load_config_file(config_name):
    '''
    Loads YAML config file in demos/config as a dict
    '''
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../demos', 'config/'))
    config_file = config_path+"/"+config_name+".yml"
    config = load_yaml_file(config_file)
    return config

