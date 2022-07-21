# Description
Several approaches toward Force feedback in Optimal Control are explored:
- __Assuming imperfect torque actuation (a.k.a. "LPF")__
- Relaxing the rigid contact assumption (a.k.a. "soft contact")

The first approach (submitted to IROS22) consists in modeling the actuation as a low-pass filter (LPF) on the torques in order to allow joint torque predictive feedback. It is tested on several models (KUKA iiwa, Talos) for several tasks. The core of this approach is a custom action model implemented in C++ my fork of the Crocoddyl library. 

# Dependencies
- [robot_properties_kuka](https://github.com/machines-in-motion/robot_properties_kuka)
- [Crocoddyl](https://github.com/skleff1994/crocoddyl/tree/devel) ("devel" branch)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [scipy](https://scipy.org/)
- [example_robot_data](https://github.com/Gepetto/example-robot-data) 
- [Gepetto-viewer](https://github.com/Gepetto/gepetto-viewer)

You also need either one of these in order to run MPC simulations :
<!-- - [RaiSim](https://raisim.com/index.html) (only for RaiSim simulations) -->
- [PyBullet](https://pybullet.org/wordpress/)  
- [bullet_utils](https://github.com/machines-in-motion/bullet_utils) 

If you don't have PyYaml and six installed : `pip3 install PyYaml && pip3 install six`

# How to use it
The core and utilities for each type of MPC are available in 
- classical_mpc : classical MPC based on position-velocity state feedback and torque control
- lpf_mpc : "LPF" MPC based on augmented state (position, velocity, torques) feedback 
- soft_mpc : "soft" MPC based on visco-elastic contact model to allow cartesian force feedback 

Each of these directories has an `ocp.py` and `data.py` modules that implement the OCP setup and OCP-specific data handlers. These classes derive from abstract classes implemented in `utils/ocp_utils.py` and `utils/data_utils.py`

In `demos` you'll find python scripts sorted by tasks:
- "reaching" (Static reaching task) : reach a 3D position with a speficied end-endeffector
- "circle" (Circle tracking task) : track a circular EE trajectory at a constant speed
- "rotation" (Rotation tracking task) : track an EE orientation at constant speed 
- "contact" (Normal force task) : exert a constant normal force on a flat horizontal surface
- "sanding" (Sanding task) : track a circular EE tajectory on a flat surface _while_ exerting a constant normal force (cf. IROS 22)

Python scripts and config files (in `demos`) contain the name of the task, "LPF" (or not), "OCP" or "MPC" and the name of the robot "iiwa", "talos_arm", "talos_reduced". "OCP" scripts are just setting up the OCP and solving it _offline_ with Crocoddyl, plotting the solution and animating it in Gepetto Viewer. "MPC" scripts setup & solve the OCP _online_ using a simulator (PyBullet), plot and save the results. 

## Solve an OCP
```
python demos/{task_name}/{task}_OCP.py [--robot_name=$NAME] [--PLOT] [--VISUALIZE]
```
For instance, to solve an OCP for a reaching task for the KUKA arm, plot the results and animate in Gepetto-viewer
```
python demos/static_raching_task/reaching_OCP.py --robot_name=iiwa --PLOT --VISUALIZE
```
This script reads the corresponding YAML configuration file `demos/config/iiwa_reaching_OCP.yml` and sets up the OCP defined in `utils/ocp_utils` and solves it. The results are plotted using custom plotting scripts implemented in `utils/plot_utils` (functions names starting with "plot_ddp"). Now if we want instead to solve an OCP for TALOS left arm, using the LPF approach, animate the results but not plot 
```
python demos/reaching/LPF_reaching_OCP.py --robot_name=talos --VISUALIZE
```

## Simulate MPC 
<!-- ! No active RaiSim support currently, use PyBullet instead ! --> 
<!-- We also need to specify the simulator.  -->
Similar syntax as previously, replacing "OCP" by "MPC". For instance
```
python demos/reaching/reaching_MPC.py  --robot_name=iiwa --PLOT_INIT
```
This script reads the corresponding YAML configuration file in `demos/config/iiwa_reaching_MPC.yml` and sets up an OCP defined in `utils/ocp_utils` , plots the initial solution and then simulates the MPC in PyBullet. The results are of the simulations are plotted using custom plotting scripts implemented in `utils/plot_utils` (functions names starting with "plot_mpc"). The simulation data can be optionally saved as .npz for offline analysis. 

# Comments
- The core of the code is MPC script and the OCP setup in `utils/ocp_utils.py` which acts basically as wrapper around Crocoddyl's API to conveniently setup any OCP from my templated YAML config files.
- A convenient feature that could be added as a plugin to Crocoddyl in the future is the plotting utilities that generate automatically near paper-ready & fancy plots of MPC simulations
- The rest of the repo (i.e. what is not described above) contains draft code for approaches based on soft contact models, and is mainly under construction at the moment.
