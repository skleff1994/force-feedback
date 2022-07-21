"""
@package force_feedback
@file contact_circle_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2021-10-28
@brief Closed-loop 'LPF torque feedback' MPC for sanding task
"""

'''
The robot is tasked with exerting a constant normal force with its EE 
while drawing a circle on the contact surface
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateLPF y=(q,v,tau), control w = unfiltered torque)
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with torque feedback where
the actuation dynamics is modeled as a low pass filter (LPF) in the optimization.
  - The letter y denotes the augmented state of joint positions, velocities
    and filtered torques while the letter 'w' denotes the unfiltered torque 
    input to the actuation model. 
  - We optimize (y*,w*) using Crocoddyl but we send tau* to the simulator (NOT w*)
  - Simulator = custom actuation model (not LPF) + PyBullet RBD
'''

import sys
sys.path.append('.')

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 1


from core_mpc import ocp, path_utils, pin_utils, mpc_utils, misc_utils


from classical_mpc.data import MPCDataHandlerClassical, DDPDataHandlerClassical
from classical_mpc.ocp import OptimalControlProblemClassical


WARM_START_IK = True


def main(robot_name='iiwa', simulator='bullet', PLOT_INIT=False):


  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config, config_name = path_utils.load_config_file(__file__, robot_name)
  # Create a simulation environment & simu-pin wrapper 
  dt_simu = 1./float(config['simu_freq'])  
  q0 = np.asarray(config['q0'])
  v0 = np.asarray(config['dq0'])
  x0 = np.concatenate([q0, v0])   
  if(simulator == 'bullet'):
    from core_mpc import sim_utils as simulator_utils
    env, robot_simulator, _ = simulator_utils.init_bullet_simulation(robot_name, dt=dt_simu, x0=x0)
    robot = robot_simulator.pin_robot
  elif(simulator == 'raisim'):
    from core_mpc import raisim_utils as simulator_utils
    env, robot_simulator, _ = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
    robot = robot_simulator
  else:
    logger.error('Please choose a simulator from ["bullet", "raisim"] !')
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Initial placement
  frame_name = config['contacts'][0]['contactModelFrameName']
  id_endeff = robot.model.getFrameId(frame_name)
  ee_frame_placement = robot.data.oMf[id_endeff].copy()
  contact_placement = robot.data.oMf[id_endeff].copy()
  contact_placement.translation =  contact_placement.act( np.asarray(config['contact_plane_offset']) ) 
  # Optionally tilt the contact surface
  TILT_RPY = np.zeros(3)
  if(config['TILT_SURFACE']):
    TILT_RPY = [0., config['TILT_PITCH_LOCAL_DEG']*np.pi/180, 0.]
    contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY)
  # Create the contact surface in PyBullet simulator 
  contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement.copy(), bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
  # Set lateral friction coefficient of the contact surface
  simulator_utils.set_friction_coef(contact_surface_bulletId, 0.5)


  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  # Init shooting problem and solver
  ddp = OptimalControlProblemClassical(robot, config).initialize(x0, callbacks=False)
  # Setup tracking problem with circle ref EE trajectory
  models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
  RADIUS = config['frameCircleTrajectoryRadius'] 
  OMEGA  = config['frameCircleTrajectoryVelocity']
  for k,m in enumerate(models):
      # Ref
      t = min(k*config['dt'], config['numberOfRounds']*2*np.pi/OMEGA)
      p_ee_ref = ocp.circle_point_WORLD(t, ee_frame_placement, 
                                                radius=RADIUS,
                                                omega=OMEGA,
                                                LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
      #  Cost translation
      m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
      #  Contact model 1D update z ref (WORLD frame)
      m.differential.contacts.contacts[frame_name].contact.reference = p_ee_ref
      
  # Warm start state = IK of circle trajectory
  WARM_START_IK = True
  if(WARM_START_IK):
      logger.info("Computing warm-start using Inverse Kinematics...")
      xs_init = [] 
      us_init = []
      q_ws = q0
      for k,m in enumerate(list(ddp.problem.runningModels) + [ddp.problem.terminalModel]):
          # Get ref placement
          p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
          Mref = ee_frame_placement.copy()
          Mref.translation = p_ee_ref
          q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
          xs_init.append(np.concatenate([q_ws, v_ws]))
      us_init = [pin_utils.get_u_grav(xs_init[i][:nq], robot.model, config['armature']) for i in range(config['N_h'])]
  # Classical warm start using initial config
  else:
      ug  = pin_utils.get_u_grav(q0, robot.model, config['armature'])
      xs_init = [x0 for i in range(config['N_h']+1)]
      us_init = [ug for i in range(config['N_h'])]

  # solve
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

  # Plot initial solution
  if(PLOT_INIT):
    ddp_handler = DDPDataHandlerClassical(ddp)
    ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
    _, _ = ddp_handler.plot_ddp_results(ddp_data, markers=['.'], SHOW=True)





  # # # # # # # # # # #
  ### INIT MPC SIMU ###
  # # # # # # # # # # #
  sim_data = MPCDataHandlerClassical(config, robot)
  sim_data.init_sim_data(x0)
    # Replan & control counters
  nb_plan = 0
  nb_ctrl = 0
  # Additional simulation blocks 
  communicationModel = mpc_utils.CommunicationModel(config)
  actuationModel     = mpc_utils.ActuationModel(config, nu=nu)
  sensingModel       = mpc_utils.SensorModel(config)

  # Display target circle  trajectory (reference)
  nb_points = 20 
  for i in range(nb_points):
    t = (i/nb_points)*2*np.pi/OMEGA
    pl = pin_utils.rotate(ee_frame_placement, rpy=TILT_RPY)
    pos = ocp.circle_point_WORLD(t, pl, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
    simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])

  draw_rate = 200


  # SIMULATE
  for i in range(sim_data.N_simu): 

      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(sim_data.N_simu))
        print('')


    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
      if(i%int(sim_data.simu_freq/sim_data.plan_freq) == 0):          
          # Current simulation time
          t_simu = i*dt_simu 
          # Setup tracking problem with circle ref EE trajectory
          models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
          for k,m in enumerate(models):
              # Ref
              t = min(t_simu + k*sim_data.dt, config['numberOfRounds']*2*np.pi/OMEGA)
              p_ee_ref = ocp.circle_point_WORLD(t, ee_frame_placement.copy(), 
                                                        radius=RADIUS,
                                                        omega=OMEGA,
                                                        LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
              # Cost translation
              m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
              # Contact model
              m.differential.contacts.contacts[frame_name].contact.reference = p_ee_ref 

          # Reset x0 to measured state + warm-start solution
          ddp.problem.x0 = sim_data.state_mea_SIMU[i, :]
          xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
          xs_init[0] = sim_data.state_mea_SIMU[i, :]
          us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
          # Solve OCP 
          ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
          # Record MPC predictions, cost references and solver data 
          sim_data.record_predictions(nb_plan, ddp)
          sim_data.record_cost_references(nb_plan, ddp)
          sim_data.record_solver_data(nb_plan, ddp) 
          # Model communication delay between computer & robot (buffered OCP solution)
          communicationModel.step(sim_data.x_pred, sim_data.u_curr)
          # Record interpolated desired state, control and force at MPC frequency
          sim_data.record_plan_cycle_desired(nb_plan)
          # Increment planning counter
          nb_plan += 1

    # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
      if(i%int(sim_data.simu_freq/sim_data.ctrl_freq) == 0):   
          # Record interpolated desired state, control and force at CTRL frequency
          sim_data.record_ctrl_cycle_desired(nb_ctrl)     
          # Increment control counter
          nb_ctrl += 1

    # Simulate actuation/sensing and step simulator (physics simulation frequency)
      # Record interpolated desired state, control and force at SIM frequency
      sim_data.record_simu_cycle_desired(i)
      # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
      tau_mea_SIMU = actuationModel.step(i, sim_data.u_ref_SIMU, sim_data.ctrl_des_SIMU) 
      # RICCATI GAINS TO INTERPOLATE
      if(config['RICCATI']):
        tau_mea_SIMU += ddp.K[0].dot(ddp.problem.x0 - sim_data.state_mea_SIMU[i,:])
      #  Send output of actuation torque to the RBD simulator 
      robot_simulator.send_joint_command(tau_mea_SIMU)
      env.step()
      # Measure new state from simulation 
      q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
      # Update pinocchio model
      robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
      f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, sim_data.PIN_REF_FRAME)
      if(i%50==0): 
        logger.info("f_mea = "+str(f_mea_SIMU))
      # Record data (unnoised)
      x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
      sim_data.state_mea_no_noise_SIMU[i+1, :] = x_mea_SIMU
      # Sensor model ( simulation state ==> noised / filtered state )
      sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, x_mea_SIMU, sim_data.state_mea_SIMU)
      sim_data.force_mea_SIMU[i, :] = f_mea_SIMU


      # Display real 
      if(i%draw_rate==0):
        pos = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
        simulator_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])
    

  # # # # # # # # # # #
  # PLOT SIM RESULTS  #
  # # # # # # # # # # #
  save_dir = '/home/skleff/force-feedback/data'
  save_name = config_name+'_bullet_'+\
                          '_BIAS='+str(config['SCALE_TORQUES'])+\
                          '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                          '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                          '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)

  # Extract plot data from sim data
  plot_data = sim_data.extract_data(frame_of_interest=frame_name)
  # Plot results
  sim_data.plot_mpc_results(plot_data, which_plots=sim_data.WHICH_PLOTS,
                                      PLOT_PREDICTIONS=True, 
                                      pred_plot_sampling=int(sim_data.plan_freq/10),
                                      SAVE=True,
                                      SAVE_DIR=save_dir,
                                      SAVE_NAME=save_name,
                                      AUTOSCALE=True)
  # Save optionally
  if(config['SAVE_DATA']):
    sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)



if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)