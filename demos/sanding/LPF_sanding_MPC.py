"""
@package force_feedback
@file LPF_contact_circle_MPC.py
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

import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 1


from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils, mpc_utils, misc_utils

import time 



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
    from utils import sim_utils as simulator_utils
    env, robot_simulator, _ = simulator_utils.init_bullet_simulation(robot_name, dt=dt_simu, x0=x0)
    robot = robot_simulator.pin_robot
  elif(simulator == 'raisim'):
    from utils import raisim_utils as simulator_utils
    env, robot_simulator, _ = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
    robot = robot_simulator
  else:
    logger.error('Please choose a simulator from ["bullet", "raisim"] !')
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Initial placement
  id_endeff = robot.model.getFrameId(config['frame_of_interest'])
  ee_frame_placement = robot.data.oMf[id_endeff].copy()
  contact_placement = robot.data.oMf[id_endeff].copy()
  M_ct = robot.data.oMf[id_endeff].copy()
  offset = 0.03348 #0.036 #0.0335
  contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
  # Optionally tilt the contact surface
  TILT_RPY = np.zeros(3)
  if(config['TILT_SURFACE']):
    TILT_RPY = [0., config['TILT_PITCH_LOCAL_DEG']*np.pi/180, 0.]
    contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY)
  # Create the contact surface in PyBullet simulator 
  contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement.copy(), with_collision=True)
  # Set lateral friction coefficient of the contact surface
  simulator_utils.set_friction_coef(contact_surface_bulletId, 0.5)


  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  N_h = config['N_h']
  dt = config['dt']
  # Create DDP solver + compute warm start torque
  f_ext = pin_utils.get_external_joint_torques(contact_placement.copy(), config['frameForceRef'], robot)
  u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
  y0 = np.concatenate([x0, u0])
  ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=False, w_reg_ref=np.zeros(nq) ) 
  models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
  RADIUS = config['frameCircleTrajectoryRadius'] 
  OMEGA  = config['frameCircleTrajectoryVelocity']
  for k,m in enumerate(models):
      # Ref
      t = min(k*config['dt'], config['numberOfRounds']*2*np.pi/OMEGA)
      p_ee_ref = ocp_utils.circle_point_WORLD(t, ee_frame_placement, 
                                                 radius=RADIUS,
                                                 omega=OMEGA)
      # Cost translation
      m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
      # Contact model 1D update z ref (WORLD frame)
      m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref[2]
      
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
          # Get corresponding forces at each joint
          f_ext = pin_utils.get_external_joint_torques(Mref.copy(), config['frameForceRef'], robot)
          # Get joint state from IK
          q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref.copy(), DT=1e-2, IT_MAX=100)
          tau_ws = pin_utils.get_tau(q_ws, v_ws, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
          xs_init.append(np.concatenate([q_ws, v_ws, tau_ws]))
          if(k<N_h):
              us_init.append(tau_ws)
  # Classical warm start using initial config
  else:
      xs_init = [y0 for i in range(config['N_h']+1)]
      us_init = [u0 for i in range(config['N_h'])]

  
  # solve
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
  # Plot initial solution
  PLOT_INIT = PLOT_INIT
  if(PLOT_INIT):
    ddp_data = data_utils.extract_ddp_data_LPF(ddp)
    fig, ax = plot_utils.plot_ddp_results_LPF(ddp_data, markers=['.'], SHOW=True)


  # # # # # # # # # # #
  ### INIT MPC SIMU ###
  # # # # # # # # # # #
  sim_data = data_utils.init_sim_data_LPF(config, robot, y0, frame_of_interest=config['frame_of_interest'])
    # Get frequencies
  freq_PLAN = sim_data['plan_freq']
  freq_CTRL = sim_data['ctrl_freq']
  freq_SIMU = sim_data['simu_freq']
    # Replan & control counters
  nb_plan = 0
  nb_ctrl = 0
  # Sim options
  WHICH_PLOTS = config['WHICH_PLOTS']                                  # Which plots to generate ? ('y':state, 'w':control, 'p':end-eff, etc.)
  FILTER_STATE = config['FILTER_STATE']                  # Moving average smoothing of reference torques
  dt_ocp = config['dt']                                  # OCP sampling rate 
  dt_mpc = float(1./sim_data['plan_freq'])               # planning rate
  OCP_TO_PLAN_RATIO  = dt_mpc / dt_ocp                   # ratio
  PLAN_TO_SIMU_RATIO = dt_simu / dt_mpc                  # Must be an integer !!!!
  OCP_TO_SIMU_RATIO  = dt_simu / dt_ocp                  # Must be an integer !!!!
  logger.info("OCP  --> PLAN ratio = "+str(OCP_TO_PLAN_RATIO))
  logger.info("OCP  --> SIMU ratio = "+str(OCP_TO_SIMU_RATIO))
  logger.info("PLAN --> SIMU ratio = "+str(PLAN_TO_SIMU_RATIO))
  time.sleep(2)
  if(1./PLAN_TO_SIMU_RATIO%1 != 0):
    logger.warning("SIMU->MPC ratio not an integer ! (1./PLAN_TO_SIMU_RATIO = "+str(1./PLAN_TO_SIMU_RATIO)+")")
  if(1./OCP_TO_SIMU_RATIO%1 != 0):
    logger.warning("SIMU->OCP ratio not an integer ! (1./OCP_TO_SIMU_RATIO  = "+str(1./OCP_TO_SIMU_RATIO)+")")



  # Additional simulation blocks 
  communication = mpc_utils.CommunicationModel(config)
  actuation     = mpc_utils.ActuationModel(config, nu, SEED=RANDOM_SEED)
  sensing       = mpc_utils.SensorModel(config, ntau=nu, SEED=RANDOM_SEED)



  # Display target circle  trajectory (reference)
  nb_points = 20 
  for i in range(nb_points):
    t = (i/nb_points)*2*np.pi/OMEGA
    pl = pin_utils.rotate(ee_frame_placement, rpy=TILT_RPY)
    pos = ocp_utils.circle_point_WORLD(t, pl, radius=RADIUS, omega=OMEGA)
    simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
  
  draw_rate = 200

  # # # # # # # # # # # #
  ### SIMULATION LOOP ###
  # # # # # # # # # # # #

  # SIMULATE
  for i in range(sim_data['N_simu']): 
  
      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))
        print('')
  
    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
      if(i%int(freq_SIMU/freq_PLAN) == 0):       
          # Current simulation time
          t_simu = i*dt_simu 
          # Setup tracking problem with circle ref EE trajectory
          models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
          for k,m in enumerate(models):
              # Ref
              t = min(t_simu + k*dt_ocp, config['numberOfRounds']*2*np.pi/OMEGA)
              p_ee_ref = ocp_utils.circle_point_WORLD(t, ee_frame_placement.copy(), 
                                                          radius=RADIUS,
                                                          omega=OMEGA)
              # Cost translation
              m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
              # Contact model
              m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref[2] 
          # Reset x0 to measured state + warm-start solution
          ddp.problem.x0 = sim_data['state_mea_SIMU'][i, :]
          xs_init        = list(ddp.xs[1:]) + [ddp.xs[-1]]
          xs_init[0]     = sim_data['state_mea_SIMU'][i, :]
          us_init        = list(ddp.us[1:]) + [ddp.us[-1]] 
          # Solve OCP & record MPC predictions
          ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
          sim_data['state_pred'][nb_plan, :, :]  = np.array(ddp.xs)
          sim_data['ctrl_pred'][nb_plan, :, :]   = np.array(ddp.us)
          sim_data['force_pred'][nb_plan, :, :]  = np.array([ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(config['N_h'])])
          # Extract relevant predictions for interpolations
          y_curr = sim_data['state_pred'][nb_plan, 0, :]    # y0* = measured state    (q^,  v^ , tau^ )
          y_pred = sim_data['state_pred'][nb_plan, 1, :]    # y1* = predicted state   (q1*, v1*, tau1*) 
          w_curr = sim_data['ctrl_pred'][nb_plan, 0, :]     # w0* = optimal control   (w0*) !! UNFILTERED TORQUE !!
          f_curr = sim_data['force_pred'][nb_plan, 0, :]
          f_pred = sim_data['force_pred'][nb_plan, 1, :]
          # Record cost references
          data_utils.record_cost_references_LPF(ddp, sim_data, nb_plan)
          # Record solver data (optional)
          if(config['RECORD_SOLVER_DATA']):
            data_utils.record_solver_data(ddp, sim_data, nb_plan) 
          # Model OCP solving time & communication between computer --> robot
          y_pred, w_curr = communication.step(y_pred, w_curr)
          # Select reference control and state for the current PLAN cycle
          y_ref_PLAN  = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
          w_ref_PLAN  = w_curr
          f_ref_PLAN  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)
          if(nb_plan==0):
            sim_data['state_des_PLAN'][nb_plan, :] = y_curr  
          sim_data['ctrl_des_PLAN'][nb_plan, :]    = w_ref_PLAN   
          sim_data['state_des_PLAN'][nb_plan+1, :] = y_ref_PLAN    
          sim_data['force_des_PLAN'][nb_plan, :]   = f_ref_PLAN    
  
          # Increment planning counter
          nb_plan += 1
  
    # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
      if(i%int(freq_SIMU/freq_CTRL) == 0):        
          # print("  CTRL ("+str(nb_ctrl)+"/"+str(sim_data['N_ctrl'])+")")
          # Select reference control and state for the current CTRL cycle
          y_ref_CTRL = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
          w_ref_CTRL = w_curr 
          f_ref_CTRL = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)
          # First prediction = measurement = initialization of MPC
          if(nb_ctrl==0):
            sim_data['state_des_CTRL'][nb_ctrl, :] = y_curr  
          sim_data['ctrl_des_CTRL'][nb_ctrl, :]    = w_ref_CTRL  
          sim_data['state_des_CTRL'][nb_ctrl+1, :] = y_ref_CTRL  
          sim_data['force_des_CTRL'][nb_ctrl, :]   = f_ref_CTRL  
          # Increment control counter
          nb_ctrl += 1
          
    # Simulate actuation/sensing and step simulator (physics simulation frequency)
  
      # Select reference control and state for the current SIMU cycle
      y_ref_SIMU  = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
      w_ref_SIMU  = w_curr 
      f_ref_SIMU  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)
      
      # First prediction = measurement = initialization of MPC
      if(i==0):
        sim_data['state_des_SIMU'][i, :] = y_curr  
      sim_data['ctrl_des_SIMU'][i, :]    = w_ref_SIMU  
      sim_data['state_des_SIMU'][i+1, :] = y_ref_SIMU 
      sim_data['force_des_SIMU'][i, :]   = f_ref_SIMU 
  
      # Torque applied by motor on actuator : interpolate current torque and predicted torque 
      tau_ref_SIMU =  y_ref_SIMU[-nu:] 
      # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
      tau_mea_SIMU = actuation.step(i, tau_ref_SIMU, sim_data['state_mea_SIMU'][:,-nu:])   

      # RICCATI GAINS TO INTERPOLATE
      if(config['RICCATI']):
        K = ddp.K[0]
        alpha = np.exp(-2*np.pi*config['f_c']*dt)
        Ktilde  = (1-alpha)*OCP_TO_PLAN_RATIO*K
        Ktilde[:,2*nq:3*nq] += ( 1 - (1-alpha)*OCP_TO_PLAN_RATIO )*np.eye(nq) # only for torques
        tau_mea_SIMU += Ktilde[:,:nq+nv].dot(ddp.problem.x0[:nq+nv] - sim_data['state_mea_SIMU'][i,:nq+nv]) #position vel
        tau_mea_SIMU += Ktilde[:,:-nq].dot(ddp.problem.x0[:-nq] - sim_data['state_mea_SIMU'][i,:-nq])           # torques

      # Send output of actuation torque to the RBD simulator 
      robot_simulator.send_joint_command(tau_mea_SIMU)
      env.step()
      # Measure new state from simulation :
      q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
      # Update pinocchio model
      robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
      # Measure contact wrench from bullet simulator
      f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff)
      if(i%100==0): 
        print(f_mea_SIMU)
      # # Estimate measured torques from measured contact wrench
      # f_ext = pin_utils.get_external_joint_torques(robot.data.oMf[id_endeff].copy(), f_mea_SIMU, robot)
      # if(i==0):
      #   a_mea_SIMU = np.zeros(nv)
      # else:
      #   a_mea_SIMU = (v_mea_SIMU - sim_data['state_mea_SIMU'][i, nq:nq+nv])/dt_simu
      # tau_mea_SIMU = pin_utils.get_tau(q_mea_SIMU, v_mea_SIMU, a_mea_SIMU, f_ext, robot.model)
      # if(i%10==0): 
      #   # logger.info("force MEA = "+str(f_mea_SIMU))
      #   logger.info("tau   REF = "+str(tau_ref_SIMU))
      #   logger.info("tau   MEA = "+str(tau_mea_SIMU))
      #  Record data (unnoised)
      y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU]).T 
      sim_data['state_mea_no_noise_SIMU'][i+1, :] = y_mea_SIMU
      # Sensor model (optional noise + filtering)
      sim_data['state_mea_SIMU'][i+1, :] = sensing.step(i, y_mea_SIMU, sim_data['state_mea_SIMU'])
      sim_data['force_mea_SIMU'][i, :]   = f_mea_SIMU
  
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
                          '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)
  # Save optionally
  if(config['SAVE_DATA']):
    data_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)


  # Extract plot data from sim data
  plot_data = data_utils.extract_plot_data_from_sim_data_LPF(sim_data)
  # Plot results
  plot_utils.plot_mpc_results_LPF(plot_data, which_plots=WHICH_PLOTS,
                                  PLOT_PREDICTIONS=True, 
                                  pred_plot_sampling=int(freq_PLAN/10),
                                  SAVE=True,
                                  SAVE_DIR=save_dir,
                                  SAVE_NAME=save_name,
                                  AUTOSCALE=True)




if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)