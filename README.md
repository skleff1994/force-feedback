# What is it
Force feedback in Optimal Control : preliminary studies. Several approaches are explored, relaxing the rigid contact assumption, or not. Also the problem of "impedance MPC" is analyzed through simulations.

# Dependencies
- [PyBullet](https://pybullet.org/wordpress/)
- [bullet_utils](https://github.com/machines-in-motion/bullet_utils) 
- [robot_properties_kuka](https://github.com/machines-in-motion/robot_properties_kuka)
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) 
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)

# How to use it
All relevant scripts are in `demos` so far. In `core` there is a custom DDP solver and Kalman filter. `models` contains dynamics models and cost models that are compatible with the custom DDP solver and Crocoddyl Integrated Action Models (IAM) of simple systems (point mass, ...).  
