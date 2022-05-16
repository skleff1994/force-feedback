"""
@package force_feedback
@file LPF_contact_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2021-10-28
@brief Closed-loop 'LPF torque feedback' MPC for force task
"""

'''
The robot is tasked with exerting a constant normal force with its EE 
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

from core_mpc import path_utils, pin_utils, mpc_utils, misc_utils

from lpf_mpc.data import DDPDataHandlerLPF, MPCDataHandlerLPF
from lpf_mpc.ocp import OptimalControlProblemLPF

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
    env, robot_simulator = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
    robot = robot_simulator
  else:
    logger.error('Please choose a simulator from ["bullet", "raisim"] !')
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Initial placement
  id_endeff = robot.model.getFrameId(config['frame_of_interest'])
  contact_placement = robot.data.oMf[id_endeff].copy()
  M_ct = robot.data.oMf[id_endeff].copy()
  contact_placement.translation =  contact_placement.act( np.asarray(config['contact_plane_offset']) ) 
  contact_placement.rotation    =  contact_placement.rotation
  simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)

  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  # Create DDP solver + compute warm start torque
  f_ext = pin_utils.get_external_joint_torques(contact_placement.copy(), config['frameForceRef'], robot)
  u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
  y0 = np.concatenate([x0, u0])
  ddp = OptimalControlProblemLPF(robot, config).initialize(y0, callbacks=False)
  # Warmstart and solve
  xs_init = [y0 for i in range(config['N_h']+1)]
  us_init = [u0 for i in range(config['N_h'])]
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
  # Plot initial solution
  frame_name = config['frameTranslationFrameName']
  if(PLOT_INIT):
    ddp_handler = DDPDataHandlerLPF(ddp)
    ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name)
    _, _ = ddp_handler.plot_ddp_results(ddp_data, markers=['.'], SHOW=True)


  # # # # # # # # # # #
  ### INIT MPC SIMU ###
  # # # # # # # # # # #
  sim_data = MPCDataHandlerLPF(config, robot)
  sim_data.init_sim_data(y0)
    # Replan & control counters
  nb_plan = 0
  nb_ctrl = 0
  # Additional simulation blocks 
  communicationModel = mpc_utils.CommunicationModel(config)
  actuationModel     = mpc_utils.ActuationModel(config, nu, SEED=RANDOM_SEED)
  sensingModel       = mpc_utils.SensorModel(config, nq=nq, nv=nv, ntau=nu, SEED=RANDOM_SEED)
  # # # # # # # # # # # #
  ### SIMULATION LOOP ###
  # # # # # # # # # # # #

  # SIMULATE
  for i in range(config['N_simu']): 

      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(config['N_simu']))
        print('')


    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
      if(i%int(sim_data.simu_freq/sim_data.plan_freq) == 0):       
          # Reset x0 to measured state + warm-start solution
          ddp.problem.x0 = sim_data.state_mea_SIMU[i, :]
          xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
          xs_init[0] = sim_data.state_mea_SIMU[i, :]
          us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
          # Solve OCP & record MPC predictions
          ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
          # Record MPC predictions, cost refs and solver data
          sim_data.record_predictions(nb_plan, ddp)
          # Record cost references
          sim_data.record_cost_references(nb_plan, ddp)
          # Record solver data (optional)
          sim_data.record_solver_data(nb_plan, ddp) 
          # Model communication between computer --> robot
          communicationModel.step(sim_data.y_pred, sim_data.w_curr)
          # Select reference control and state for the current PLAN cycle
          sim_data.record_plan_cycle_desired(nb_plan)
          # Increment planning counter
          nb_plan += 1


    # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
      if(i%int(sim_data.simu_freq/sim_data.ctrl_freq) == 0):        
          sim_data.record_ctrl_cycle_desired(nb_ctrl)
          nb_ctrl += 1
          
    # Simulate actuation/sensing and step simulator (physics simulation frequency)
      # Record interpolated desired state, control and force at SIM frequency
      sim_data.record_simu_cycle_desired(i)
      # Torque applied by motor on actuator : interpolate current torque and predicted torque 
      tau_ref_SIMU =  sim_data.y_ref_SIMU[-nu:] 
      # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
      tau_mea_SIMU = actuationModel.step(i, tau_ref_SIMU, sim_data.state_mea_SIMU[:,-nu:])   
      # RICCATI GAINS TO INTERPOLATE
      if(config['RICCATI']):
        K = ddp.K[0]
        alpha = np.exp(-2*np.pi*config['f_c']*config['dt'])
        Ktilde  = (1-alpha)*sim_data.OCP_TO_PLAN_RATIO*K
        Ktilde[:,2*nq:3*nq] += ( 1 - (1-alpha)*sim_data.OCP_TO_PLAN_RATIO )*np.eye(nq) # only for torques
        tau_mea_SIMU += Ktilde[:,:nq+nv].dot(ddp.problem.x0[:nq+nv] - sim_data.state_mea_SIMU[i,:nq+nv]) #position vel
        tau_mea_SIMU += Ktilde[:,:-nq].dot(ddp.problem.x0[:-nq] - sim_data.state_mea_SIMU[i,:-nq])       # torques
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
      y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU]).T 
      sim_data.state_mea_no_noise_SIMU[i+1, :] = y_mea_SIMU
      # Sensor model ( simulation state ==> noised / filtered state )
      sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, y_mea_SIMU, sim_data.state_mea_SIMU)
      sim_data.force_mea_SIMU[i, :] = f_mea_SIMU



  # # # # # # # # # # #
  # PLOT SIM RESULTS  #
  # # # # # # # # # # #
  save_dir = '/home/skleff/force-feedback/data'
  save_name = config_name+'_bullet_'+\
                          '_BIAS='+str(config['SCALE_TORQUES'])+\
                          '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                          '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                          '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)
  #  Extract plot data from sim data
  plot_data = sim_data.extract_data(frame_of_interest=frame_name)
  #  Plot results
  sim_data.plot_mpc_results(plot_data, which_plots=WHICH_PLOTS,
                                  PLOT_PREDICTIONS=True, 
                                  pred_plot_sampling=int(freq_PLAN/10),
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