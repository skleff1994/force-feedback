"""
@package force_feedback
@file exp_TILT_LPF_sanding_MPC.py
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

from core_mpc import path_utils, pin_utils, mpc_utils, misc_utils
from core_mpc import ocp as ocp_utils
from core_mpc import sim_utils as simulator_utils


from lpf_mpc.data import DDPDataHandlerLPF, MPCDataHandlerLPF
from lpf_mpc.ocp import OptimalControlProblemLPF, getJointAndStateIds


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


def solveOCP(q, v, tau, ddp, nb_iter, node_id_reach, target_reach, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force):
    t = time.time()
    # Update initial state + warm-start
    x = np.concatenate([q, v, tau])
    ddp.problem.x0 = x
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
    xs_init[0] = x
    us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
    # Get OCP nodes
    m = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    # Update OCP for reaching phase
    if(TASK_PHASE == 1):
        # If node id is valid
        if(node_id_reach <= ddp.problem.T and node_id_reach >= 0):
            # Updates nodes between node_id and terminal node 
            for k in range( node_id_reach, ddp.problem.T+1, 1 ):
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                # print(m[k].differential.costs.costs["translation"].weight)
    # Update OCP for "increase weights" phase
    if(TASK_PHASE == 2):
        # If node id is valid
        if(node_id_track <= ddp.problem.T and node_id_track >= 0):
            # Updates nodes between node_id and terminal node 
            for k in range( node_id_track, ddp.problem.T+1, 1 ):
                # w = min(2.*(k + 1. - node_id_track) , 10.)
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                # m[k].differential.costs.costs["translation"].weight = 20.
                # print(m[k].differential.costs.costs["translation"].weight)
    # Update OCP for contact phase
    if(TASK_PHASE == 3):
        # If node id is valid
        if(node_id_contact <= ddp.problem.T and node_id_contact >= 0):
            # Updates nodes between node_id and terminal node 
            for k in range( node_id_contact, ddp.problem.T+1, 1 ):  
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].weight = 1.
                # activate contact and force cost
                m[k].differential.contacts.changeContactStatus("contact", True)
                if(k!=ddp.problem.T):
                    fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                    m[k].differential.costs.costs["force"].active = True
                    # m[k].differential.costs.costs["force"].weight = force_weight
                    m[k].differential.costs.costs["force"].cost.residual.reference = fref
    # Update OCP for circle phase
    if(TASK_PHASE == 4):
        # If node id is valid
        if(node_id_circle <= ddp.problem.T and node_id_circle >= 0):
            # Updates nodes between node_id and terminal node
            for k in range( node_id_circle, ddp.problem.T+1, 1 ):
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                m[k].differential.costs.costs["translation"].weight = 50.
                # m[k].differential.costs.costs["velocity"].active = True
                # m[k].differential.costs.costs["velocity"].cost.residual.reference = pin.Motion(np.concatenate([target_velocity[k], np.zeros(3)]))
                # m[k].differential.costs.costs["velocity"].cost.activation.weights = np.array([1., 1., 0., 1., 1., 1.])
                # m[k].differential.costs.costs["velocity"].weight = 1.
                # activate contact and force cost
                m[k].differential.contacts.changeContactStatus("contact", True)
                if(k!=ddp.problem.T):
                    fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                    m[k].differential.costs.costs["force"].active = True
                    # m[k].differential.costs.costs["force"].weight = force_weight
                    m[k].differential.costs.costs["force"].cost.residual.reference = fref
    # Solve OCP 
    ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
    # Send solution to parent process + riccati gains
    t_child = time.time() - t
    return ddp.us, ddp.xs, ddp.K, t_child   


def main(robot_name):

  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config, config_name = path_utils.load_config_file('LPF_sanding_MPC', robot_name)
  # Create a simulation environment & simu-pin wrapper 
  dt_simu = 1./float(config['simu_freq'])  
  q0 = np.asarray(config['q0'])
  v0 = np.asarray(config['dq0'])
  x0 = np.concatenate([q0, v0])   
  env, robot_simulator, _ = simulator_utils.init_bullet_simulation(robot_name+'_reduced', dt=dt_simu, x0=x0)
  robot = robot_simulator.pin_robot
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Placement of LOCAL end-effector frame w.r.t. WORLD frame
  frame_of_interest = config['frame_of_interest']
  id_endeff = robot.model.getFrameId(frame_of_interest)
  oMf = robot.data.oMf[id_endeff].copy()
  import pinocchio as pin
  contact_placement   = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
  contact_placement_0 = contact_placement.copy()
  # EE translation target : contact point + vertical offset (radius of the ee ball)
  contactTranslationTarget = np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
  targetId = simulator_utils.display_ball(contactTranslationTarget, RADIUS=0.02, COLOR=[1.,0.,0.,0.2])

  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  N_h = config['N_h']
  dt = config['dt']
  # Create DDP solver + compute warm start torque
  f_ext = pin_utils.get_external_joint_torques(contact_placement.copy(), config['frameForceRef'], robot)
  u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
  lpf_joint_names = robot.model.names[1:] #['A1', 'A2', 'A3', 'A4'] #  #
  _, lpfStateIds = getJointAndStateIds(robot.model, lpf_joint_names)
  n_lpf = len(lpf_joint_names)
  _, nonLpfStateIds = getJointAndStateIds(robot.model, list(set(robot.model.names[1:]) - set(lpf_joint_names)) )
  logger.debug("LPF state ids ")
  logger.debug(lpfStateIds)
  logger.debug("Non LPF state ids ")
  logger.debug(nonLpfStateIds)
  y0 = np.concatenate([x0, u0[lpfStateIds]])
  ddp = OptimalControlProblemLPF(robot, config, lpf_joint_names).initialize(y0, callbacks=False)
  
  models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
  # !!! Deactivate all costs & contact models initially !!!
  for k,m in enumerate(models):
      m.differential.costs.costs["translation"].active = False
      if(k < config['N_h']):
           m.differential.costs.costs["force"].active = False
           m.differential.costs.costs["force"].cost.residual.reference = pin.Force.Zero()
      m.differential.contacts.changeContactStatus("contact", False)
      # logger.debug(str(m.differential.costs.active.tolist()))


  # Setup tracking problem with circle ref EE trajectory + Warm start state = IK of circle trajectory
  RADIUS = config['frameCircleTrajectoryRadius'] 
  OMEGA  = config['frameCircleTrajectoryVelocity']
  xs_init = [] 
  us_init = []

  # Force trajectory
  F_MIN = 5.
  F_MAX = config['frameForceRef'][2]
  N_total = 10000 # int((config['T_tot'] - config['T_CONTACT'])/config['dt'] + config['N_h'])
  N_min  = 5
  N_ramp = N_min + 10
  target_force_traj = np.zeros( (N_total, 3) )
  target_force_traj[0:N_min*config['N_h'], 2] = F_MIN
  target_force_traj[N_min*config['N_h']:N_ramp*config['N_h'], 2] = [F_MIN + (F_MAX - F_MIN)*i/((N_ramp-N_min)*config['N_h']) for i in range((N_ramp-N_min)*config['N_h'])]
  target_force_traj[N_ramp*config['N_h']:, 2] = F_MAX
  target_force = np.zeros(config['N_h']+1)
  force_weight = np.asarray(config['frameForceWeight'])
  # Circle trajectory 
  N_total_pos = 10000 # int((config['T_tot'] - config['T_REACH'])/config['dt'] + config['N_h'])
  N_circle = 10000 # int((config['T_tot'] - config['T_CIRCLE'])/config['dt']) + config['N_h']
  target_position_traj = np.zeros( (N_total_pos, 3) )
  target_velocity_traj = np.zeros( (N_total_pos, 3) )
  # absolute desired position
  oPc_offset = np.asarray(config['oPc_offset'])
  pdes = np.asarray(config['contactPosition']) + oPc_offset
  target_position_traj[0:N_circle, :] = [np.array([pdes[0] + RADIUS * np.sin(i*config['dt']*OMEGA), 
                                                        pdes[1] + RADIUS * (1-np.cos(i*config['dt']*OMEGA)),
                                                        pdes[2]]) for i in range(N_circle)]
  target_velocity_traj[0:N_circle, :] = [np.array([RADIUS * OMEGA * np.cos(i*config['dt']*OMEGA), 
                                                        RADIUS * OMEGA * np.sin(i*config['dt']*OMEGA),
                                                        0.]) for i in range(N_circle)]
  target_position_traj[N_circle:, :] = target_position_traj[N_circle-1,:]
  target_velocity_traj[N_circle:, :] = np.zeros(3)
  target_position = np.zeros((config['N_h']+1, 3)) 
  target_position[:,:] = pdes.copy()
  target_velocity = np.zeros((config['N_h']+1, 3)) 
  q_ws = q0
  logger.info("Computing warm-start using Inverse Kinematics...")
  for k,m in enumerate(models):
      # Ref
      t = min(k*config['dt'], 2*np.pi/OMEGA)
      p_ee_ref = ocp_utils.circle_point_WORLD(t, contact_placement_0.copy(), 
                                                 radius=RADIUS,
                                                 omega=OMEGA,
                                                 LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
      # Cost translation
      m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
      # Contact model 1D update z ref (WORLD frame)
      m.differential.oPc = p_ee_ref
      # Get ref placement
      p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
      Mref = contact_placement_0.copy()
      Mref.translation = p_ee_ref
      # Get corresponding forces at each joint + joint state from IK
      f_ext = pin_utils.get_external_joint_torques(Mref.copy(), config['frameForceRef'], robot)
      q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
      tau_ws = pin_utils.get_tau(q_ws, v_ws, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
      xs_init.append(np.concatenate([q_ws, v_ws, tau_ws[lpfStateIds]]))
      if(k<N_h):
          us_init.append(tau_ws)
  # Warmstart and solve
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)




  for n_seed in range(N_SEEDS):
    
    print("Set Random Seed to "+str(SEEDS[n_seed]) + " ("+str(n_seed)+"/"+str(N_SEEDS)+")")
    np.random.seed(SEEDS[n_seed])

    for n_exp in range(N_EXP):

        # Reset robot to initial state and set table
        robot_simulator.reset_state(q0, v0)
        robot_simulator.forward_robot(q0, v0)
        # simulator_utils.print_dynamics_info(1, 9)
        # Display contact surface + optional tilt
        import pinocchio as pin
        contact_placement   = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
        contact_placement_0 = contact_placement.copy()
        contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY[n_exp])
        contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
        # Make the contact soft (e.g. tennis ball or sponge on the robot)
        simulator_utils.set_lateral_friction(contact_surface_bulletId, 0.5)
        simulator_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 1000000, 2000)
        # Display target circle  trajectory (reference)
        nb_points = 20 
        ballsIdTarget = np.zeros(nb_points, dtype=int)
        ballsIdReal = []
        for i in range(nb_points):
            t = (i/nb_points)*2*np.pi/OMEGA
            pos = ocp_utils.circle_point_WORLD(t, contact_placement_0, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
            ballsIdTarget[i] = simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
        draw_rate = 200

        logger.warning("ROBOT ID = "+str(robot_simulator.robot_id))
        logger.warning("TARGET ID = "+str(targetId))
        logger.warning("CONTACT ID = "+str(contact_surface_bulletId))
        logger.warning("CONTACTs ID = "+str(ballsIdTarget))
        
        # # # # # # # # # # #
        ### INIT MPC SIMU ###
        # # # # # # # # # # #
        sim_data = MPCDataHandlerLPF(config, robot_simulator.pin_robot, n_lpf)
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
        from core_mpc.analysis_utils import MPCBenchmark
        bench = MPCBenchmark()

        # Horizon in simu cycles
        node_id_reach   = -1
        node_id_contact = -1
        node_id_track   = -1
        node_id_circle  = -1
        TASK_PHASE      = 0
        NH_SIMU   = int(config['N_h']*sim_data.dt/sim_data.dt_simu)
        T_REACH   = int(config['T_REACH']/sim_data.dt_simu)
        T_TRACK   = int(config['T_TRACK']/sim_data.dt_simu)
        T_CONTACT = int(config['T_CONTACT']/sim_data.dt_simu)
        T_CIRCLE   = int(config['T_CIRCLE']/sim_data.dt_simu)
        OCP_TO_MPC_CYCLES = 1./(sim_data.dt_plan / config['dt'])
        OCP_TO_SIMU_CYCLES = 1./(sim_data.dt_simu / config['dt'])
        logger.debug("Size of MPC horizon in simu cycles     = "+str(NH_SIMU))
        logger.debug("Start of reaching phase in simu cycles = "+str(T_REACH))
        logger.debug("Start of tracking phase in simu cycles = "+str(T_TRACK))
        logger.debug("Start of contact phase in simu cycles  = "+str(T_CONTACT))
        logger.debug("Start of circle phase in simu cycles   = "+str(T_CIRCLE))
        logger.debug("OCP to PLAN time ratio = "+str(OCP_TO_MPC_CYCLES))


        # # # # # # # # # # # #
        ### SIMULATION LOOP ###
        # # # # # # # # # # # #

        # SIMULATE
        for i in range(config['N_simu']): 

            if(i%config['log_rate']==0 and config['LOG']): 
                print('')
                logger.info("SIMU step "+str(i)+"/"+str(config['N_simu']))
                print('')


            # # # # # # # # # 
            # # Update OCP  #
            # # # # # # # # # 
            time_to_reach   = int(i - T_REACH)
            time_to_track   = int(i - T_TRACK)
            time_to_contact = int(i - T_CONTACT)
            time_to_circle  = int(i - T_CIRCLE)

            if(time_to_reach == 0): 
                logger.warning("Entering reaching phase")
            # If tracking phase enters the MPC horizon, start updating models from the end with tracking models      
            if(0 <= time_to_reach and time_to_reach <= NH_SIMU):
                TASK_PHASE = 1
                # If current time matches an OCP node 
                if(time_to_reach%OCP_TO_SIMU_CYCLES == 0):
                    # Select IAM
                    node_id_reach = config['N_h'] - int(time_to_reach/OCP_TO_SIMU_CYCLES)

            if(time_to_track == 0): 
                logger.warning("Entering tracking phase")
            # If "increase weights" phase enters the MPC horizon, start updating models from the end with tracking models      
            if(0 <= time_to_track and time_to_track <= NH_SIMU):
                TASK_PHASE = 2
                # If current time matches an OCP node 
                if(time_to_track%OCP_TO_SIMU_CYCLES == 0):
                    # Select IAM
                    node_id_track = config['N_h'] - int(time_to_track/OCP_TO_SIMU_CYCLES)

            if(time_to_contact == 0): 
                logger.warning("Entering contact phase")
                # Record end-effector position at the time of the contact switch
                position_at_contact_switch = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
                target_position[:,:] = position_at_contact_switch.copy()
            # If contact phase enters horizon start updating models from the the end with contact models
            if(0 <= time_to_contact and time_to_contact <= NH_SIMU):
                TASK_PHASE = 3
                # If current time matches an OCP node 
                if(time_to_contact%OCP_TO_SIMU_CYCLES == 0):
                    # Select IAM
                    node_id_contact = config['N_h'] - int(time_to_contact/OCP_TO_SIMU_CYCLES)

            if(0 <= time_to_contact and time_to_contact%OCP_TO_SIMU_CYCLES == 0):
                ti  = int(time_to_contact/OCP_TO_SIMU_CYCLES)
                tf  = ti + config['N_h']+1
                target_force = target_force_traj[ti:tf,2]

            if(time_to_circle == 0): 
                logger.warning("Entering circle phase")
            # If circle tracking phase enters the MPC horizon, start updating models from the end with tracking models      
            if(0 <= time_to_circle and time_to_circle <= NH_SIMU):
                TASK_PHASE = 4
                # If current time matches an OCP node 
                if(time_to_circle%OCP_TO_SIMU_CYCLES == 0):
                    # Select IAM
                    node_id_circle = config['N_h'] - int(time_to_circle/OCP_TO_SIMU_CYCLES)

            if(0 <= time_to_circle and time_to_circle%OCP_TO_SIMU_CYCLES == 0):
                # set position refs over current horizon
                ti  = int(time_to_circle/OCP_TO_SIMU_CYCLES)
                tf  = ti + config['N_h']+1
                # Target in (x,y)  = circle trajectory + offset to start from current position instead of absolute target
                offset_xy = position_at_contact_switch[:2] - pdes[:2]
                target_position[:,:2] = target_position_traj[ti:tf,:2] + offset_xy
                # Target in z is fixed to the anchor at switch (equals absolute target if RESET_ANCHOR = False)
                # No position tracking in z : redundant with zero activation weight on z
                target_position[:,2]  = robot_simulator.pin_robot.data.oMf[id_endeff].translation[2].copy()
                # Record target signals                
                target_velocity[:,:2] = target_velocity_traj[ti:tf,:2] 
                target_velocity[:,2]  = 0.



            # Solve OCP if we are in a planning cycle (MPC/planning frequency)
            if(i%int(sim_data.simu_freq/sim_data.plan_freq) == 0):       
                # Reset x0 to measured state + warm-start solution
                q = sim_data.state_mea_SIMU[i, :nq]
                v = sim_data.state_mea_SIMU[i, nq:nq+nv]
                f = sim_data.state_mea_SIMU[i, nq+nv:]
                # Solve OCP 
                bench.start_timer()
                bench.start_croco_profiler()
                solveOCP(q, v, f, ddp, config['maxiter'], node_id_reach, target_position, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force)
                bench.record_profiles()
                bench.stop_timer(nb_iter=ddp.iter)
                bench.stop_croco_profiler()
                # Record MPC predictions, cost references and solver data 
                sim_data.record_predictions(nb_plan, ddp)
                sim_data.record_cost_references(nb_plan, ddp)
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
            # f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, sim_data.PIN_REF_FRAME)
            f_mea_SIMU = robot_simulator.end_effector_forces(sim_data.PIN_REF_FRAME)[1][0]
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
            
        bench.plot_timer()
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
    main(args.robot_name)
