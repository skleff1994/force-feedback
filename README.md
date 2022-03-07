# What is it
Several approaches toward Force feedback in Optimal Control are explored:
- Relaxing the rigid contact assumption (a.k.a. "augmented state" and "observer")
- Assuming imperfect torque actuation (a.k.a. "LPF") 

In particular, the second approach is explored in details throughout simulations on 7-DoF manipulators (on the KUKA iiwa LBR 14 and PAL Robotics' TALOS left arm). We call it the "LPF" approach, which consists basically in modeling the actuation as a low-pass filter (LPF) on the torques. The OCP uses an augmented dynamics model treating joint torques as states, and unfiltered torque as control input. This is implemented in C++ my fork of the Crocoddyl library. Reaching tasks and contact tasks are simulated in both PyBullet and Raisim.

# Dependencies
- [robot_properties_kuka](https://github.com/machines-in-motion/robot_properties_kuka)
- [Crocoddyl](https://github.com/skleff1994/crocoddyl/tree/lpf_contact1d) ("lpf_contact1d" branch)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [scipy](https://scipy.org/)
- [example_robot_data](https://github.com/Gepetto/example-robot-data) 
- [Gepetto-viewer](https://github.com/Gepetto/gepetto-viewer)

You also need either one of these in order to run MPC simulations :
- [RaiSim](https://raisim.com/index.html) (only for RaiSim simulations)
- [PyBullet](https://pybullet.org/wordpress/)  (only for PyBullet simulations)
- [bullet_utils](https://github.com/machines-in-motion/bullet_utils) (only for PyBullet simulations)

If you don't have PyYaml and six installed : `pip3 install PyYaml && pip3 install six`

# How to use it
As of Jan. 10, 2022, all relevant scripts are in `demos` and `utils`. In `demos` you'll find python scripts sorted by tasks:
- "reaching" (Static reaching task) : reach a 3D position with a speficied end-endeffector
- "circle" (Circle tracking task) : track a circular EE trajectory at a constant speed
- "rotation" (Rotation tracking task) : track an EE orientation at constant speed 
- "contact" (Normal force task) : exert a constant normal force on a flat horizontal surface
- "sanding" (Sanding task) : track a circular EE tajectory on a flat surface _while_ exerting a constant normal force

The sanding task is meant to compare task performance between classical MPC and LPF MPC and therebt highlight the benefits of force feedback in MPC (submitted to IROS 2022). 


Python scripts (in `demos`) and config files in (`utils`) are contain the name of the task, "LPF" (or not), "OCP" or "MPC" and the name of the robot "iiwa" or "talos" (only config files). "OCP" scripts are just setting up the OCP and solving it _offline_ with Crocoddyl, plotting the solution and animating it in Gepetto Viewer. "MPC" scripts setup & solve the OCP _online_ using a simulator (PyBullet or Raisim), plot and save the results. 

## Solve an OCP
```
python demos/{task_name}/{task}_OCP.py {--robot_name=''} {--PLOT} {--VISUALIZE}
```
For instance, to solve an OCP for a reaching task for the KUKA arm, plot the results and animate in Gepetto-viewer
```
python demos/static_raching_task/reaching_OCP.py --robot_name='iiwa' --PLOT --VISUALIZE
```
This script reads the corresponding YAML configuration file `demos/config/iiwa_reaching_OCP.yml` and sets up the OCP defined in `utils/ocp_utils` and solves it. The results are plotted using custom plotting scripts implemented in `utils/plot_utils` (functions names starting with "plot_ddp"). Now if we want instead to solve an OCP for TALOS left arm, using the LPF approach, animate the results but not plot 
```
python demos/static_raching_task/LPF_reaching_OCP.py --robot_name='talos' --VISUALIZE
```

## Simulate MPC 
! No active RaiSim support currently, use PyBullet instead !

Similar syntax as previously, replacing "OCP" by "MPC". We also need to specify the simulator. For instance
```
python demos/static_reaching_task/reaching_MPC.py  --robot_name='iiwa' --simulator='bullet' 
```
This script reads the corresponding YAML configuration file in `demos/config/iiwa_reaching_MPC.yml` and sets up an OCP defined in `utils/ocp_utils` and simulates the MPC in PyBullet (or RaiSim). The results are of the simulations are plotted using custom plotting scripts implemented in `utils/plot_utils` (functions names starting with "plot_mpc"). The simulation data can be optionally saved as .npz for offline analysis. 

# Comments
- The core of the code is MPC script and the OCP setup in `utils/ocp_utils.py` which acts basically as wrapper around Crocoddyl's API to conveniently setup any OCP from my templated YAML config files.
- A convenient feature that could be added as a plugin to Crocoddyl in the future is the plotting utilities that generate automatically near paper-ready & fancy plots of MPC simulations
- The rest of the repo (i.e. what is not described above) contains draft code for approaches based on soft contact models, and is mainly under construction at the moment.
