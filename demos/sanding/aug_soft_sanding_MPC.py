"""
@package force_feedback
@file demos/sanding/aug_soft_sanding_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2023-04-04
@brief Closed-loop MPC for force task 
"""

'''
The robot is tasked with exerting a constant normal force  while drawing a circle
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from state x=(q,v,f), control u = tau) 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with state feedback, optionally
imperfect actuation (bias, noise, delays) at higher frequency
'''


import sys
sys.path.append('.')

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc import path_utils, pin_utils, mpc_utils, misc_utils
from core_mpc import ocp as ocp_utils

from soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from soft_mpc.aug_data import DDPDataHandlerSoftContactAugmented, MPCDataHandlerSoftContactAugmented
from soft_mpc.utils import SoftContactModel3D, SoftContactModel1D


WARM_START_IK      = True
RESET_ANCHOR_POINT = True

import time

def solveOCP(q, v, f, ddp, nb_iter, node_id_reach, target_reach, anchor_point, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force):
        t = time.time()
        x = np.concatenate([q, v, f])
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
        # Update OCP for "increase weights" phase
        if(TASK_PHASE == 2):
            # If node id is valid
            if(node_id_track <= ddp.problem.T and node_id_track >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_track, ddp.problem.T+1, 1 ):
                    w = min(1.*(k + 1. - node_id_track) , 5.)
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].weight = w
        # Update OCP for contact phase
        if(TASK_PHASE == 3):
            # If node id is valid
            if(node_id_contact <= ddp.problem.T and node_id_contact >= 0):
                # Updates nodes between node_id and terminal node
                for k in range( node_id_contact, ddp.problem.T+1, 1 ):
                    # fref = np.array([0., 0., target_force[k]]) 
                    fref = np.array([target_force[k]])
                    m[k].differential.active_contact = True
                    m[k].differential.f_des = fref.copy()
                    m[k].differential.f_weight = force_weight
                    m[k].differential.oPc = anchor_point
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].weight = 0.
                    # m[k].differential.costs.costs["velocity"].cost.activation.weights = np.array([10., 10., 1., 1., 1., 1.])
        # Update OCP for circle phase
        if(TASK_PHASE == 4):
            # If node id is valid
            if(node_id_circle <= ddp.problem.T and node_id_circle >= 0):
                # Updates nodes between node_id and terminal node
                for k in range( node_id_circle, ddp.problem.T+1, 1 ):
                    # fref = np.array([0., 0.,  target_force[k]]) 
                    fref = np.array([target_force[k]])
                    m[k].differential.active_contact = True
                    m[k].differential.f_des = fref.copy()
                    m[k].differential.f_weight = force_weight
                    m[k].differential.oPc = anchor_point
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                    m[k].differential.costs.costs["translation"].weight = 50.
                    # m[k].differential.costs.costs["velocity"].active = True
                    # m[k].differential.costs.costs["velocity"].cost.residual.reference = pin.Motion(np.concatenate([target_velocity[k], np.zeros(3)]))
                    # m[k].differential.costs.costs["velocity"].cost.activation.weights = np.array([1., 1., 0., 1., 1., 1.])
                    # m[k].differential.costs.costs["velocity"].weight = 1.
                    # print(m[k].differential.costs.costs["velocity"].cost.residual.reference)
                    # print(m[k].differential.costs.costs["velocity"].cost.residual.type)
        problem_formulation_time = time.time()
        t_child_1 =  problem_formulation_time - t
        # Solve OCP 
        ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        # ddp.problem.calcDiff(ddp.xs, ddp.us)
        # Send solution to parent process + riccati gains
        solve_time = time.time()
        ddp_iter = ddp.iter
        t_child =  solve_time - problem_formulation_time
        return ddp.us, ddp.xs, ddp.K, t_child, ddp_iter, t_child_1


def main(robot_name='iiwa', simulator='bullet', PLOT_INIT=False):

  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config, config_name = path_utils.load_config_file(__file__, robot_name)
  # config_name = 'soft_mpc_contact'
  # file = '/home/skleff/ws/workspace/src/force_feedback_dgh/config/soft_mpc_contact.yml' # 
  # config = path_utils.load_yaml_file(file)
  # Create a simulation environment & simu-pin wrapper 
  dt_simu = 1./float(config['simu_freq'])  
  q0 = np.asarray(config['q0'])
  v0 = np.asarray(config['dq0'])
  x0 = np.concatenate([q0, v0])   
  if(simulator == 'bullet'):
    from core_mpc import sim_utils as simulator_utils
    env, robot_simulator, base_placement = simulator_utils.init_bullet_simulation(robot_name+'_reduced', dt=dt_simu, x0=x0)
    robot = robot_simulator.pin_robot
  elif(simulator == 'raisim'):
    from core_mpc import raisim_utils as simulator_utils
    env, robot_simulator = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
    robot = robot_simulator
  else:
    logger.error('Please choose a simulator from ["bullet", "raisim"] !')
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Placement of LOCAL end-effector frame w.r.t. WORLD frame
  frame_of_interest = config['frame_of_interest']
  id_endeff = robot.model.getFrameId(frame_of_interest)
  oMf = robot.data.oMf[id_endeff].copy()

  # simulator_utils.print_dynamics_info(1, 9)
  # EE translation target : contact point + vertical offset (radius of the ee ball)
  contactTranslationTarget = np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
  simulator_utils.display_ball(contactTranslationTarget, RADIUS=0.02, COLOR=[1.,0.,0.,0.2])
  # Display contact surface + optional tilt
  import pinocchio as pin
  contact_placement   = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
  contact_placement_0 = contact_placement.copy()
  TILT_RPY = np.zeros(3)
  if(config['TILT_SURFACE']):
    # TILT_RPY = [0., config['TILT_PITCH_LOCAL_DEG']*np.pi/180, 0.]
    TILT_RPY = [config['TILT_PITCH_LOCAL_DEG']*np.pi/180, 0., 0.]
    contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY)
  contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
  # Make the contact soft (e.g. tennis ball or sponge on the robot)
  simulator_utils.set_lateral_friction(contact_surface_bulletId, 0.1)
  simulator_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 1000000, 2000)


  # Contact model
  oPc = contact_placement.translation + np.asarray(config['oPc_offset'])
  simulator_utils.display_ball(oPc, base_placement, RADIUS=0.01, COLOR=[1.,0.,0.,1.])
  if('1D' in config['contactType']):
      softContactModel = SoftContactModel1D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['contactType'], config['pinRefFrame'])
  else:
      softContactModel = SoftContactModel3D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['pinRefFrame'])

  softContactModel.print()
  
  # Measure initial force in pybullet
  f0 = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, softContactModel.pinRefFrame)
  y0 = np.concatenate([x0, f0[-softContactModel.nc:]])  
  # Get corresponding external torques
  f_ext0 = pin_utils.get_external_joint_torques(contact_placement, f0, robot)   

  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # #
  # Compute initial gravity compensation torque torque   
  u0 = pin_utils.get_tau(q0, v0, np.zeros(nq), f_ext0, robot.model, np.zeros(nq))
  # Setup Croco OCP and create solver
  ddp = OptimalControlProblemSoftContactAugmented(robot, config).initialize(y0, softContactModel, callbacks=False) #True)
    # Warmstart and solve
  xs_init = [y0 for i in range(config['N_h']+1)]
  us_init = [u0 for i in range(config['N_h'])] #ddp.problem.quasiStatic(xs_init[:-1])
    
  # !!! Deactivate all costs & contact models initially !!!
  # Set the force cost reference frame to LWA 
  models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
  for k,m in enumerate(models):
    m.differential.costs.costs["translation"].active = False
    m.differential.active_contact = False
    m.differential.f_des = np.zeros(softContactModel.nc)
    m.differential.cost_ref = pin.LOCAL_WORLD_ALIGNED

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
  N_total_pos = int((config['T_tot'] - config['T_REACH'])/config['dt'] + config['N_h'])
  N_circle = int((config['T_tot'] - config['T_CIRCLE'])/config['dt']) + config['N_h']
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
  anchor_point = pdes.copy()
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
      f_ws = np.array([softContactModel.computeForce_(robot.model, q_ws, v_ws)])
      xs_init.append(np.concatenate([q_ws, v_ws, f_ws]))
      if(k<config['N_h']):
        us_init.append(pin_utils.get_tau(y0[:nq], y0[nq:nq+nv], np.zeros(nv), f_ext, robot.model, np.zeros(nv)))
  # Warmstart and solve
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

  
  if(PLOT_INIT):
    #  Plot
    ddp_handler = DDPDataHandlerSoftContactAugmented(ddp, softContactModel)
    ddp_data = ddp_handler.extract_data(frame_of_interest, frame_of_interest, robot.model)
    _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], 
                                                        colors=['r'], 
                                                        markers=['.'], 
                                                        SHOW=True)  
  
  # # # # # # # # # # #
  ### INIT MPC SIMU ###
  # # # # # # # # # # #
  sim_data = MPCDataHandlerSoftContactAugmented(config, robot, softContactModel.nc)
  sim_data.init_sim_data(y0)
    # Replan & control counters
  nb_plan = 0
  nb_ctrl = 0
  # Additional simulation blocks 
  communicationModel = mpc_utils.CommunicationModel(config)
  actuationModel     = mpc_utils.ActuationModel(config, nu=nu)
  sensingModel       = mpc_utils.SensorModel(config, naug=softContactModel.nc)
  # Display target circle  trajectory (reference)
  nb_points = 20 
  for i in range(nb_points):
    t = (i/nb_points)*2*np.pi/OMEGA
    # pl = pin_utils.rotate(contact_placement_0, rpy=TILT_RPY)
    pos = ocp_utils.circle_point_WORLD(t, contact_placement_0, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
    simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])

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

  # SIMULATE
  for i in range(sim_data.N_simu): 
      
      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(sim_data.N_simu))
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
          logger.warning("Entering contact phase ( RESET_ANCHOR = "+str(RESET_ANCHOR_POINT)+" )")
          # Record end-effector position at the time of the contact switch
          position_at_contact_switch = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
          target_position[:,:] = position_at_contact_switch.copy()
          # Optionally reset the anchor point to the current position
          if(RESET_ANCHOR_POINT): 
              anchor_point = position_at_contact_switch + oPc_offset
          logger.warning("   Anchor point = "+str(anchor_point)+" )")
      # If contact phase enters horizon start updating models from the the end with contact models
      if(0 <= time_to_contact and time_to_contact <= NH_SIMU):
          TASK_PHASE = 3
          # If current time matches an OCP node 
          if(time_to_contact%OCP_TO_SIMU_CYCLES == 0):
              # Select IAM
              node_id_contact = config['N_h'] - int(time_to_contact/OCP_TO_SIMU_CYCLES)

      if(0 <= time_to_contact and time_to_contact%OCP_TO_SIMU_CYCLES == 0):
          # set force refs over current horizon
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
          # Reset anchor point (x,y) to the current location of the end-effector to allow lateral motion
          # Redundant with 0 gains on the lateral directions
          anchor_point[:2] = robot_simulator.pin_robot.data.oMf[id_endeff].translation[:2].copy()
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
          solveOCP(q, v, f, ddp, config['maxiter'], node_id_reach, target_position, anchor_point, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force)
          bench.record_profiles()
          bench.stop_timer(nb_iter=ddp.iter)
          bench.stop_croco_profiler()
          # Record MPC predictions, cost references and solver data 
          sim_data.record_predictions(nb_plan, ddp)
          sim_data.record_cost_references(nb_plan, ddp)
          sim_data.record_solver_data(nb_plan, ddp) 
          # Model communication delay between computer & robot (buffered OCP solution)
          communicationModel.step(sim_data.y_pred, sim_data.u_curr)
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
      # RICCATI GAINS TO INTERPOLATE
      tau_mea_SIMU = sim_data.u_ref_SIMU
      if(config['RICCATI']):
        tau_mea_SIMU += ddp.K[0][:,:nq+nv].dot(sim_data.y_ref_SIMU[:nq+nv] - sim_data.state_mea_SIMU[i,:][:nq+nv])
        # tau_mea_SIMU += ddp.K[0].dot(sim_data.y_ref_SIMU - sim_data.state_mea_SIMU[i,:])
      # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
      tau_mea_SIMU = actuationModel.step(i, tau_mea_SIMU, sim_data.ctrl_des_SIMU)
      # tau_mea_SIMU = actuationModel.step(i, sim_data.u_ref_SIMU, sim_data.ctrl_des_SIMU) if actuation BEFORE riccati?
      #  Send output of actuation torque to the RBD simulator 0
      robot_simulator.send_joint_command(tau_mea_SIMU)
      env.step()
      # Measure new state from simulation 
      q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
      # Update pinocchio model
      robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
      # f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, softContactModel.pinRefFrame)
      f_mea_SIMU = robot_simulator.end_effector_forces(softContactModel.pinRefFrame)[1][0]
      fz_mea_SIMU = np.array([f_mea_SIMU[2]])
      if(i%100==0): 
        logger.info("f_mea  = "+str(f_mea_SIMU))
      # Record data (unnoised)
      y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, fz_mea_SIMU]).T 
      # y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, sim_data.y_pred[-softContactModel.nc:]]).T 
      sim_data.state_mea_no_noise_SIMU[i+1, :] = y_mea_SIMU
      # Sensor model ( simulation state ==> noised / filtered state )
      sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, y_mea_SIMU, sim_data.state_mea_SIMU)

  bench.plot_timer()
  # bench.plot_profiles()
  # bench.plot_avg_profiles()
  # # # # # # # # # # #
  # PLOT SIM RESULTS  #
  # # # # # # # # # # #
  save_dir = '/tmp'
  save_name = config_name+'_'+simulator+'_'+\
                          '_BIAS='+str(config['SCALE_TORQUES'])+\
                          '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                          '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                          '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)
  # Extract plot data from sim data
  plot_data = sim_data.extract_data(frame_of_interest=frame_of_interest)
  # Plot results
  sim_data.plot_mpc_results(plot_data, which_plots=sim_data.WHICH_PLOTS,
                                      PLOT_PREDICTIONS=True, 
                                      pred_plot_sampling=int(sim_data.plan_freq/10),
                                      SAVE=False,
                                      SAVE_DIR=save_dir,
                                      SAVE_NAME=save_name,
                                      AUTOSCALE=False)
  # Save optionally
  if(config['SAVE_DATA']):
    sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)





if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)