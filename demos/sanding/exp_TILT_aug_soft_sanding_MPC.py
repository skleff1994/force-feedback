"""
@package force_feedback
@file exp_TILT_aug_soft_sanding_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2023-04-03
@brief Closed-loop 'Augmented Soft Contact Force feedback' MPC for sanding task
"""

'''
The robot is tasked with exerting a constant normal force with its EE 
while drawing a circle on the contact surface
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateSoft y=(q,v,f), control tau = joint torque
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with force feedback where
the contact force is modeled as a spring damper
  - Simulator = custom actuation model + PyBullet RBD
'''

import sys
sys.path.append('.')

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 1

from core_mpc import path_utils, pin_utils, mpc_utils, misc_utils
from core_mpc import ocp as ocp_utils


from core_mpc import ocp, path_utils, pin_utils, mpc_utils, misc_utils
from soft_mpc.aug_data import DDPDataHandlerSoftContactAugmented, MPCDataHandlerSoftContactAugmented
from soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented 


WARM_START_IK = True

import time
import pinocchio as pin

WARM_START_IK = True

# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [15, 10, 5, 0, -5, -10, -15] #[-20, -15, -10, -5, 0, 5, 10, 15, 20]

# EXPERIMENTS = [TILT_ANGLES_DEG[n_exp] for n_s in range(len(SEEDS)) for n_exp in range(len(TILT_ANGLES_DEG)) ]
# N_EXP = len(EXPERIMENTS)

TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([angle*np.pi/180, 0., 0.])
N_EXP = len(TILT_RPY)

SEEDS = [1] #, 2, 3, 4, 5]
N_SEEDS = len(SEEDS)


def main(robot_name, simulator, PLOT_INIT):


  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config, config_name = path_utils.load_config_file('aug_soft_sanding_MPC', robot_name)
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
  id_endeff = robot.model.getFrameId(config['frame_of_interest'])
  ee_frame_placement = robot.data.oMf[id_endeff].copy()
  contact_placement = robot.data.oMf[id_endeff].copy()
  offset = 0.03348 #0.036 #0.0335
  contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 


  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  N_h = config['N_h']
  dt = config['dt']
  # Create DDP solver + compute warm start torque
  f_ext = pin_utils.get_external_joint_torques(contact_placement.copy(), config['frameForceRef'], robot)
  u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
  y0 = np.concatenate([x0, u0])
  lpf_joint_names = robot.model.names[1:] #['A1', 'A2', 'A3', 'A4'] #  #
  _, lpfStateIds = getJointAndStateIds(robot.model, lpf_joint_names)
  n_lpf = len(lpf_joint_names)
  _, nonLpfStateIds = getJointAndStateIds(robot.model, list(set(robot.model.names[1:]) - set(lpf_joint_names)) )
  ddp = OptimalControlProblemSoftContactAugmented(robot, config, lpf_joint_names).initialize(y0, callbacks=False)
#   ddp = ocp.(robot, config, y0, callbacks=False)
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



  for n_seed in range(N_SEEDS):
    
    print("Set Random Seed to "+str(SEEDS[n_seed]) + " ("+str(n_seed)+"/"+str(N_SEEDS)+")")
    np.random.seed(SEEDS[n_seed])

    for n_exp in range(N_EXP):

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
        simulator_utils.set_lateral_friction(contact_surface_bulletId, 0.5)
        # Display target circle  trajectory (reference)
        nb_points = 20 
        ballsIdTarget = np.zeros(nb_points, dtype=int)
        for i in range(nb_points):
            t = (i/nb_points)*2*np.pi/OMEGA
            pl = ee_frame_placement.copy() #pin_utils.rotate(ee_frame_placement, rpy=TILT_RPY[n_exp])
            pos = ocp.circle_point_WORLD(t, pl, 
                                                  radius=RADIUS, 
                                                  omega=OMEGA, 
                                                  LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
            ballsIdTarget[i] = simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
        draw_rate = 200
        ballsIdReal = []

        # # # # # # # # # # #
        ### INIT MPC SIMU ###
        # # # # # # # # # # #
        sim_data = MPCDataHandlerLPF(config, robot, n_lpf)
        sim_data.init_sim_data(y0)
            # Replan & control counters
        nb_plan = 0
        nb_ctrl = 0

        # Additional simulation blocks 
        communicationModel = mpc_utils.CommunicationModel(config)
        actuationModel     = mpc_utils.ActuationModel(config, nu, SEED=SEEDS[n_seed])
        sensingModel       = mpc_utils.SensorModel(config, naug=n_lpf, SEED=SEEDS[n_seed])


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
                    #  Cost translation
                    m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
                    # Contact model
                    m.differential.contacts.contacts[frame_name].contact.reference = p_ee_ref  
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
            tau_ref_SIMU =  sim_data.y_ref_SIMU[-n_lpf:] 
            # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
                # Simulate imperfect actuation (for dimensions that are assumed perfectly actuated by the MPC)
                #  sim_data.w_ref_SIMU, sim_data.ctrl_des_SIMU 
            tau_mea_SIMU = sim_data.w_ref_SIMU 
            tau_mea_SIMU[nonLpfStateIds] = actuationModel.step(i, sim_data.w_ref_SIMU[nonLpfStateIds], sim_data.ctrl_des_SIMU[nonLpfStateIds]) 
                # Simulate imperfect actuation (for dimensions that are asssumed to be LPF-actuated by the MPC)
            tau_mea_SIMU[lpfStateIds] = actuationModel.step(i, tau_ref_SIMU, sim_data.state_mea_SIMU[:,-n_lpf:])   
            # RICCATI GAINS TO INTERPOLATE
            if(config['RICCATI']):
                K = ddp.K[0]
                alpha = np.exp(-2*np.pi*config['f_c']*config['dt'])
                Ktilde  = (1-alpha)*sim_data.OCP_TO_PLAN_RATIO*K
                # Ktilde[:,2*nq:3*nq] += ( 1 - (1-alpha)*sim_data.OCP_TO_PLAN_RATIO )*np.eye(nq) # only for torques
                tau_mea_SIMU += Ktilde[:,:nq+nv].dot(ddp.problem.x0[:nq+nv] - sim_data.state_mea_SIMU[i,:nq+nv]) #position vel
                # tau_mea_SIMU += Ktilde[:,:-nq].dot(ddp.problem.x0[:-nq] - sim_data.state_mea_SIMU[i,:-nq])       # torques
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
            y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU[lpfStateIds]]).T 
            sim_data.state_mea_no_noise_SIMU[i+1, :] = y_mea_SIMU
            # Sensor model ( simulation state ==> noised / filtered state )
            sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, y_mea_SIMU, sim_data.state_mea_SIMU)
            sim_data.force_mea_SIMU[i, :] = f_mea_SIMU

        
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
                                '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)+\
                                '_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+\
                                '_SEED='+str(SEEDS[n_seed])
        # Save optionally   
        if(config['SAVE_DATA']):
            sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)




if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)
