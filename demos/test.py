# Display a point mass in Gepetto viewer 
# MPC simulation ?
import pinocchio as pin
import numpy as np
import sys
import os
from os.path import dirname, join
from pinocchio.visualize import GepettoVisualizer

############################################
### ROBOT MODEL & SIMULATION ENVIRONMENT ###
############################################
urdf_path = os.path.join('/home/skleff/force-feedback/demos', 'point_mass.urdf')
mesh_dir = '/home/skleff/force-feedback/demos'
# pin_robot = RobotWrapper(pin.buildModelFromUrdf(urdf_path))
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, '/home/skleff/force-feedback/demos', pin.JointModelFreeFlyer())

# # Pinrobot wrapper
from pinocchio.robot_wrapper import RobotWrapper
# pin_robot = RobotWrapper(model)
# print(pin_robot.model)
# id_endeff = pin_robot.model.getFrameId('contact')
# nq = pin_robot.model.nq 
# nv = pin_robot.model.nv

# simulation loop (pinocchio + )
N = 10
for i in range(N):
    
    # Display robot + environment

    # Formulate OCP based on current state 

    # Solve it 

    # Integrate dynamics under u WITH CONTACT (pinocchio or pybullet)

    # Get new state

    # Increment


# Visualize in gepetto viewer
viz = GepettoVisualizer(model, collision_model, visual_model)
viz.initViewer()

# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)