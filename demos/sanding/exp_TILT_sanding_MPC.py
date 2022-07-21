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

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np  
np.random.seed(1)
np.set_printoptions(precision=4, linewidth=180)


from utils import path_utils, ocp_utils, pin_utils, data_utils, mpc_utils, misc_utils




TASK = 'contact_circle'
WARM_START_IK = True

# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [-20, -15, -10, -5, 0, 5, 10, 15, 20] 

# EXPERIMENTS = [TILT_ANGLES_DEG[n_exp] for n_s in range(len(SEEDS)) for n_exp in range(len(TILT_ANGLES_DEG)) ]
# N_EXP = len(EXPERIMENTS)

TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([0., angle*np.pi/180, 0.])
N_EXP = len(TILT_RPY)

SEEDS = [1, 2, 3, 4, 5]
N_SEEDS = len(SEEDS)
# np.random.seed(1)

def main(robot_name, simulator, PLOT_INIT):


  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config, config_name = path_utils.load_config_file('sanding_MPC', robot_name)
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



  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  # Init shooting problem and solver
  ddp = ocp_utils.init_DDP(robot, config, x0, callbacks=False) 
  # Setup tracking problem with circle ref EE trajectory
  models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
  RADIUS = config['frameCircleTrajectoryRadius'] 
  OMEGA  = config['frameCircleTrajectoryVelocity']
  for k,m in enumerate(models):
      # Ref
      t = min(k*config['dt'], config['numberOfRounds']*2*np.pi/OMEGA)
      p_ee_ref = ocp_utils.circle_point_WORLD(t, ee_frame_placement, 
                                                radius=RADIUS,
                                                omega=OMEGA,
                                                LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
      # Cost translation
      m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
      # Contact model 1D update z ref (WORLD frame)
      m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref
      
  # Warm start state = IK of circle trajectory
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



  for n_seed in range(N_SEEDS):
    
    print("Set Random Seed to "+str(SEEDS[n_seed]) + " ("+str(n_seed)+"/"+str(N_SEEDS)+")")
    np.random.seed(SEEDS[n_seed])

    for n_exp in range(N_EXP):
        
        print("   Set angle to "+str(TILT_ANGLES_DEG[n_exp]) + " ("+str(n_exp)+"/"+str(N_EXP)+")")
        # Reset robot to initial state and set table
        robot_simulator.reset_state(q0, v0)
        robot_simulator.forward_robot(q0, v0)
        contact_placement = robot.data.oMf[id_endeff].copy()
        offset = 0.03348 
        contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
        # Optionally tilt the contact surface
        contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY[n_exp])
        # Create the contact surface in PyBullet simulator 
        contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement.copy(), bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
        # Set lateral friction coefficient of the contact surface
        simulator_utils.set_friction_coef(contact_surface_bulletId, 0.5)
        # Display target circle  trajectory (reference)
        nb_points = 20 
        ballsIdTarget = np.zeros(nb_points, dtype=int)
        for i in range(nb_points):
            t = (i/nb_points)*2*np.pi/OMEGA
            pl = ee_frame_placement.copy() #pin_utils.rotate(ee_frame_placement, rpy=TILT_RPY[n_exp])
            pos = ocp_utils.circle_point_WORLD(t, pl, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
            ballsIdTarget[i] = simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
        draw_rate = 200
        ballsIdReal = []

        # # # # # # # # # # #
        ### INIT MPC SIMU ###
        # # # # # # # # # # #
        sim_data = data_utils.init_sim_data(config, robot, x0)
            # Get frequencies
        freq_PLAN = sim_data['plan_freq']
        freq_CTRL = sim_data['ctrl_freq']
        freq_SIMU = sim_data['simu_freq']
            # Replan & control counters
        nb_plan = 0
        nb_ctrl = 0
            # Sim options
        dt_ocp            = config['dt']                            # OCP sampling rate 
        dt_mpc            = float(1./sim_data['plan_freq'])         # planning rate
        OCP_TO_PLAN_RATIO  = dt_mpc / dt_ocp                         # ratio
        PLAN_TO_SIMU_RATIO = dt_simu / dt_mpc                        # Must be an integer !!!!
        OCP_TO_SIMU_RATIO  = dt_simu / dt_ocp                        # Must be an integer !!!!
        if(1./PLAN_TO_SIMU_RATIO%1 != 0):
            logger.warning("SIMU->MPC ratio not an integer ! (1./PLAN_TO_SIMU_RATIO = "+str(1./PLAN_TO_SIMU_RATIO)+")")
        if(1./OCP_TO_SIMU_RATIO%1 != 0):
            logger.warning("SIMU->OCP ratio not an integer ! (1./OCP_TO_SIMU_RATIO  = "+str(1./OCP_TO_SIMU_RATIO)+")")




        # Additional simulation blocks 
        communication = mpc_utils.CommunicationModel(config)
        actuation     = mpc_utils.ActuationModel(config, nu, SEED=SEEDS[n_seed])
        sensing       = mpc_utils.SensorModel(config, SEED=SEEDS[n_seed])


        # # # # # # # # # # # #
        ### SIMULATION LOOP ###
        # # # # # # # # # # # #

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
                # Circle defined w.r.t. NON-tilted surface (controller thinks it's all flat)
                models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
                for k,m in enumerate(models):
                    # Ref
                    t = min(t_simu + k*dt_ocp, config['numberOfRounds']*2*np.pi/OMEGA)
                    p_ee_ref = ocp_utils.circle_point_WORLD(t, ee_frame_placement.copy(), 
                                                                radius=RADIUS,
                                                                omega=OMEGA,
                                                                LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
                    # Cost translation
                    m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
                    # Contact model
                    m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref
                # Reset x0 to measured state + warm-start solution
                ddp.problem.x0 = sim_data['state_mea_SIMU'][i, :]
                xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
                xs_init[0] = sim_data['state_mea_SIMU'][i, :]
                us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
                # Solve OCP & record MPC predictions
                ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
                sim_data['state_pred'][nb_plan, :, :] = np.array(ddp.xs)
                sim_data['ctrl_pred'][nb_plan, :, :] = np.array(ddp.us)
                sim_data ['force_pred'][nb_plan, :, :] = np.array([ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(config['N_h'])])
                # Extract relevant predictions for interpolations
                x_curr = sim_data['state_pred'][nb_plan, 0, :]    # x0* = measured state    (q^,  v^ , tau^ )
                x_pred = sim_data['state_pred'][nb_plan, 1, :]    # x1* = predicted state   (q1*, v1*, tau1*) 
                u_curr = sim_data['ctrl_pred'][nb_plan, 0, :]    # u0* = optimal control   
                f_curr = sim_data['force_pred'][nb_plan, 0, :]
                f_pred = sim_data['force_pred'][nb_plan, 1, :]
                # Record cost references
                data_utils.record_cost_references(ddp, sim_data, nb_plan)
                # Record solver data (optional)
                if(config['RECORD_SOLVER_DATA']):
                    data_utils.record_solver_data(ddp, sim_data, nb_plan)   
                # Model communication between computer --> robot
                x_pred, u_curr = communication.step(x_pred, u_curr)
                # Select reference control and state for the current PLAN cycle
                x_ref_PLAN  = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)
                u_ref_PLAN  = u_curr #u_pred_prev + OCP_TO_PLAN_RATIO * (u_curr - u_pred_prev)
                f_ref_PLAN  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)
                if(nb_plan==0):
                    sim_data['state_des_PLAN'][nb_plan, :] = x_curr  
                sim_data['ctrl_des_PLAN'][nb_plan, :]   = u_ref_PLAN   
                sim_data['state_des_PLAN'][nb_plan+1, :] = x_ref_PLAN    
                sim_data['force_des_PLAN'][nb_plan, :] = f_ref_PLAN    
                
                # Increment planning counter
                nb_plan += 1

            # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
            if(i%int(freq_SIMU/freq_CTRL) == 0):        
                # Select reference control and state for the current CTRL cycle
                x_ref_CTRL = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)
                u_ref_CTRL = u_curr
                f_ref_CTRL = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)
                # First prediction = measurement = initialization of MPC
                if(nb_ctrl==0):
                    sim_data['state_des_CTRL'][nb_ctrl, :] = x_curr  
                sim_data['ctrl_des_CTRL'][nb_ctrl, :]   = u_ref_CTRL  
                sim_data['state_des_CTRL'][nb_ctrl+1, :] = x_ref_CTRL   
                sim_data['force_des_CTRL'][nb_ctrl, :] = f_ref_CTRL   
                # Increment control counter
                nb_ctrl += 1

            # Simulate actuation/sensing and step simulator (physics simulation frequency)

            # Select reference control and state for the current SIMU cycle
            x_ref_SIMU  = x_curr + OCP_TO_PLAN_RATIO * (x_pred - x_curr)
            u_ref_SIMU  = u_curr 
            f_ref_SIMU  = f_curr + OCP_TO_PLAN_RATIO * (f_pred - f_curr)

            # First prediction = measurement = initialization of MPC
            if(i==0):
                sim_data['state_des_SIMU'][i, :] = x_curr  
            sim_data['ctrl_des_SIMU'][i, :]   = u_ref_SIMU  
            sim_data['state_des_SIMU'][i+1, :] = x_ref_SIMU 
            sim_data['force_des_SIMU'][i, :] = f_ref_SIMU 

            # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU )    
            tau_mea_SIMU = actuation.step(i, u_ref_SIMU, sim_data['ctrl_des_SIMU'])  
            #  Send output of actuation torque to the RBD simulator 
            robot_simulator.send_joint_command(tau_mea_SIMU)
            env.step()
            # Measure new state from simulation 
            q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
            # Update pinocchio model
            robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
            f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff)
            if(i%100==0): 
                print(f_mea_SIMU)
            # Record data (unnoised)
            x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
            sim_data['state_mea_no_noise_SIMU'][i+1, :] = x_mea_SIMU
            # Sensor model (optional noise + filtering)
            sim_data['state_mea_SIMU'][i+1, :] = sensing.step(i, x_mea_SIMU, sim_data['state_mea_SIMU'])
            sim_data['force_mea_SIMU'][i, :] = f_mea_SIMU


            # Display real 
            if(i%draw_rate==0):
                pos = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
                ballId = simulator_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])
                ballsIdReal.append(ballId)

        # Remove table
        simulator_utils.remove_body_from_sim(contact_surface_bulletId)
        for ballId in ballsIdTarget:
            simulator_utils.remove_body_from_sim(ballId)
        for ballId in ballsIdReal:
            simulator_utils.remove_body_from_sim(ballId)

        # # # # # # # # # # #
        # PLOT SIM RESULTS  #
        # # # # # # # # # # #
        save_dir = '/tmp'
        save_name = config_name+'_bullet_'+\
                                '_BIAS='+str(config['SCALE_TORQUES'])+\
                                '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                                '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                                '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)+\
                                '_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+\
                                '_SEED='+str(SEEDS[n_seed])

        # Save optionally
        if(config['SAVE_DATA']):
            data_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)


if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)