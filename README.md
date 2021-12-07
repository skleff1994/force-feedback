# What is it
Several approaches toward Force feedback in Optimal Control are explored:
- Relaxing the rigid contact assumption ("augmented state" and "observer")
- Assuming imperfect torque actuation

In particular, the second lead is explored in details throughout simulations on the KUKA iiwa LBR 14 manipulator. We call it the "LPF" approach, which consists basically in modeling the actuation as a low-pass filter (LPF) on the torques. Reaching tasks and contact tasks are simulated in both PyBullet and Raisim.

# Dependencies
- [robot_properties_kuka](https://github.com/machines-in-motion/robot_properties_kuka)
- [Crocoddyl](https://github.com/skleff1994/crocoddyl.git) (my fork, "lpf" branch)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)

You also need either one of these in order to run MPC simulations :
- [RaiSim](https://raisim.com/index.html) (only for RaiSim simulations)
- [PyBullet](https://pybullet.org/wordpress/)  (only for PyBullet simulations)
- [bullet_utils](https://github.com/machines-in-motion/bullet_utils) (only for PyBullet simulations)

If you don't have PyYaml and six installed : `pip3 install PyYaml && pip3 install six`

# How to use it
As of Dec. 2, 2021, all relevant scripts are in `demos` and `utils`

## Solve OCP + plot results
For instance, to solve an OCP for a reaching task and plot the results, simply call
```
python demos/iiwa_reaching_OCP.py 
```
This script reads the corresponding YAML configuration file `demos/config/iiwa_reaching_OCP.yml` and sets up the OCP defined in `utils/ocp_utils` and solves it. The results are plotted using custom plotting scripts implemented in `utils/plot_utils` (functions names starting with "plot_ddp").

## Simulate MPC + plot the results
Now in order to simulate MPC, we need a physics simulator. We can use either PyBullet (open-source) or RaiSim (closed-source, free academic license) 
```
python demos/iiwa_reaching_MPC_bullet.py  # replace 'bullet' by 'raisim' to use raisim
```
This script reads the corresponding YAML configuration file in `demos/config/iiwa_reaching_MPC.yml` and sets up an OCP defined in `utils/ocp_utils` and simulates the MPC in PyBullet (or RaiSim). The results are of the simulations are plotted using custom plotting scripts implemented in `utils/plot_utils` (functions names starting with "plot_mpc"). The simulation data can be optionally saved as .npz for offline analysis. 

## How to do force feedback ?
All functions and script containing the suffix "\_LPF" can be used similarly. Those rely on an augmented dynamics model that includes a low-pass filter as an actuation abstraction, which is implemented in C++ my fork of the Crocoddyl library ('lpf' branch). The rest of the repo (i.e. what is not described above) contains draft code for approaches based on soft contact models, and is mainly under construction at the moment.
