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

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, misc_utils, mpc_utils
from core_mpc_utils import sim_utils as simulator_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.ocp_core_data import save_data
from croco_mpc_utils.math_utils import circle_point_WORLD
from lpf_mpc.data import MPCDataHandlerLPF
from lpf_mpc.ocp import OptimalControlProblemLPF, getJointAndStateIds

import mim_solvers
from mim_robots.robot_loader import load_bullet_wrapper
from mim_robots.pybullet.env import BulletEnvWithGround
import pybullet as p

# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [6, 4, 2, 0, -2, -4, -6] 
TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([angle*np.pi/180, 0., 0.])
N_EXP = len(TILT_RPY)
SEEDS = [19, 71, 89, 83, 41, 73, 17, 47, 29, 7]
N_SEEDS = len(SEEDS)


from LPF_sanding_MPC import solveOCP
from croco_mpc_utils.utils import load_yaml_file


# SAVE_DIR = '/home/skleff/Desktop/soft_contact_sim_exp/with_torque_control'

def main(SAVE_DIR, TORQUE_TRACKING):

    # # # # # # # # # # # # # # # # # # #
    ### LOAD ROBOT MODEL and SIMU ENV ### 
    # # # # # # # # # # # # # # # # # # # 
    # Read config file
    # config, config_name = path_utils.load_config_file('LPF_sanding_MPC', robot_name)
    config_name = 'iiwa_LPF_sanding_MPC'
    config = load_yaml_file('/home/skleff/ws_croco2/workspace/src/force-feedback/demos/sanding/config/iiwa_LPF_sanding_MPC.yml')
   
    logger.warning("save dir = "+SAVE_DIR)
    logger.warning("tracking = "+str(TORQUE_TRACKING))
    
    # Create a simulation environment & simu-pin wrapper 
    dt_simu = 1./float(config['simu_freq'])  
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])  
    env             = BulletEnvWithGround(dt=dt_simu, server=p.DIRECT)
    robot_simulator = load_bullet_wrapper('iiwa', locked_joints=['A7'])
    env.add_robot(robot_simulator) 
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
    # Initialize lpf ids, states and controls
    f_ext = pin_utils.get_external_joint_torques(contact_placement.copy(), config['frameForceRef'], robot)
    u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
    lpf_joint_names = robot.model.names[1:] #['A1', 'A2', 'A3', 'A4'] #  #
    _, lpfStateIds = getJointAndStateIds(robot.model, lpf_joint_names)
    n_lpf = len(lpf_joint_names)
    _, nonLpfStateIds = getJointAndStateIds(robot.model, list(set(robot.model.names[1:]) - set(lpf_joint_names)) )
    logger.debug("LPF state ids ")
    logger.debug(lpfStateIds)
    y0 = np.concatenate([x0, u0[lpfStateIds]])
    

    #### START EXPERIMENTS ####

    for n_seed in range(N_SEEDS):

        print("Set Random Seed to "+str(SEEDS[n_seed]) + " ("+str(n_seed)+"/"+str(N_SEEDS)+")")
        np.random.seed(SEEDS[n_seed])

        for n_exp in range(N_EXP):

            # Construct target force and position trajectories + OCP warm-start
            RADIUS = config['frameCircleTrajectoryRadius'] 
            OMEGA  = config['frameCircleTrajectoryVelocity']
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

            # Initialize Optimal Control Problem
            ocp = OptimalControlProblemLPF(robot, config, lpf_joint_names).initialize(y0)
            # !!! Deactivate all costs & contact models initially !!!
            models = list(ocp.runningModels) + [ocp.terminalModel]
            for k,m in enumerate(models):
                m.differential.costs.costs["translation"].active = False
                if(k < config['N_h']):
                    m.differential.costs.costs["force"].active = False
                    m.differential.costs.costs["force"].cost.residual.reference = pin.Force.Zero()
                m.differential.contacts.changeContactStatus("contact", False)

            # Warmstart and solve
            xs_init = [y0 for _ in range(config['N_h']+1)] 
            us_init = [u0 for _ in range(config['N_h'])]
            solver = mim_solvers.SolverSQP(ocp)
            solver.regMax                 = 1e6
            solver.reg_max                = 1e6
            solver.termination_tolerance  = 0.0001 
            solver.use_filter_line_search = True
            solver.filter_size            = config['maxiter']
            solver.solve(xs_init, us_init, maxiter=100, isFeasible=False)

            # Reset robot to initial state and set table
            robot_simulator.reset_state(q0, v0)
            robot_simulator.forward_robot(q0, v0)
            # Display contact surface + optional tilt
            contact_placement        = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
            contact_placement_0      = contact_placement.copy()
            contact_placement        = pin_utils.rotate(contact_placement, rpy=TILT_RPY[n_exp])
            contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
            # Make the contact soft (e.g. tennis ball or sponge on the robot)
            simulator_utils.set_lateral_friction(contact_surface_bulletId, 0.5)
            simulator_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 1e6, 1e3)
            # Display target circle  trajectory (reference)
            nb_points = 20 
            ballsIdTarget = np.zeros(nb_points, dtype=int)
            ballsIdReal = []
            for i in range(nb_points):
                t = (i/nb_points)*2*np.pi/OMEGA
                pos = circle_point_WORLD(t, contact_placement_0, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
                ballsIdTarget[i] = simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
            draw_rate = 1000
            
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
            if(int(TORQUE_TRACKING) == 0):
                use = False
            else:
                use = True
            torqueController   = mpc_utils.LowLevelTorqueController(config, nu=nu, use=use)
            antiAliasingFilter = mpc_utils.AntiAliasingFilter()
            

            # # # # # # # # # # # #
            ### SIMULATION LOOP ###
            # # # # # # # # # # # #
            # from core_mpc.analysis_utils import MPCBenchmark
            # bench = MPCBenchmark()

            # Horizon in simu cycles
            node_id_reach   = -1
            node_id_contact = -1
            node_id_track   = -1
            node_id_circle  = -1
            TASK_PHASE      = 0
            NH_SIMU   = int(config['N_h']*sim_data.dt/sim_data.dt_simu)
            T_REACH   = int(config['T_REACH']/sim_data.dt_simu)
            # T_TRACK   = int(config['T_TRACK']/sim_data.dt_simu)
            T_CONTACT = int(config['T_CONTACT']/sim_data.dt_simu)
            T_CIRCLE   = int(config['T_CIRCLE']/sim_data.dt_simu)
            OCP_TO_MPC_CYCLES = 1./(sim_data.dt_plan / config['dt'])
            OCP_TO_SIMU_CYCLES = 1./(sim_data.dt_simu / config['dt'])
            logger.debug("Size of MPC horizon in simu cycles     = "+str(NH_SIMU))
            logger.debug("Start of reaching phase in simu cycles = "+str(T_REACH))
            # logger.debug("Start of tracking phase in simu cycles = "+str(T_TRACK))
            logger.debug("Start of contact phase in simu cycles  = "+str(T_CONTACT))
            logger.debug("Start of circle phase in simu cycles   = "+str(T_CIRCLE))
            logger.debug("OCP to PLAN time ratio = "+str(OCP_TO_MPC_CYCLES))


            # # # # # # # # # # # #
            ### SIMULATION LOOP ###
            # # # # # # # # # # # #
            err_fz = 0
            err_p = 0
            count = 0
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
                # time_to_track   = int(i - T_TRACK)
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

                # if(time_to_track == 0): 
                #     logger.warning("Entering tracking phase")
                # # If "increase weights" phase enters the MPC horizon, start updating models from the end with tracking models      
                # if(0 <= time_to_track and time_to_track <= NH_SIMU):
                #     TASK_PHASE = 2
                #     # If current time matches an OCP node 
                #     if(time_to_track%OCP_TO_SIMU_CYCLES == 0):
                #         # Select IAM
                #         node_id_track = config['N_h'] - int(time_to_track/OCP_TO_SIMU_CYCLES)

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
                    x_filtered = antiAliasingFilter.step(nb_plan, i, sim_data.plan_freq, sim_data.simu_freq, sim_data.state_mea_SIMU)
                    q   = x_filtered[:nq]
                    v   = x_filtered[nq:nq+nv]
                    tau = x_filtered[nq+nv:]
                    # Solve OCP 
                    # bench.start_timer()
                    # bench.start_croco_profiler()
                    solveOCP(q, v, tau, solver, config['maxiter'], node_id_reach, target_position, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force)
                    # if(n_exp==1):
                    #     print("u* = ", solver.us[0])
                    # bench.record_profiles()
                    # bench.stop_timer(nb_iter=solver.iter)
                    # bench.stop_croco_profiler()
                    # Record MPC predictions, cost references and solver data 
                    sim_data.record_predictions(nb_plan, solver)
                    sim_data.record_cost_references(nb_plan, solver)
                    sim_data.record_solver_data(nb_plan, solver) 
                    # Model communication between computer --> robot
                    communicationModel.step(sim_data.y_pred, sim_data.w_curr)
                    # Select reference control and state for the current PLAN cycle
                    sim_data.record_plan_cycle_desired(nb_plan)
                    # Increment planning counter
                    nb_plan += 1


                # # # # # # # # # #
                # # Send policy # #
                # # # # # # # # # #
                # If we are in a control cycle send reference torque to motor driver and compute the motor torque
                if(i%int(sim_data.simu_freq/sim_data.ctrl_freq) == 0):   
                    # Anti-aliasing filter on measured torques (sim-->ctrl)
                    tau_mea_CTRL            = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.state_mea_SIMU[:,-n_lpf:])
                    tau_mea_derivative_CTRL = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.tau_mea_derivative_SIMU[:,-n_lpf:])
                    # Select the desired torque as interpolation between current and prediction
                    tau_des_CTRL = sim_data.y_curr[-n_lpf:] + sim_data.OCP_TO_PLAN_RATIO * (sim_data.y_pred[-n_lpf:]  - sim_data.y_curr[-n_lpf:] )
                    # Optionally interpolate to the control frequency using Riccati gains
                    if(config['RICCATI']):
                        y_filtered = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.state_mea_SIMU)
                        alpha = np.exp(-2*np.pi*config['f_c']*config['dt'])
                        Ktilde  = (1-alpha)*sim_data.OCP_TO_PLAN_RATIO*solver.K[0]
                        Ktilde[:,-n_lpf:] += ( 1 - (1-alpha)*sim_data.OCP_TO_PLAN_RATIO )*np.eye(n_lpf) # only for torques
                        # tau_des_CTRL += Ktilde[:,:nq+nv].dot(solver.problem.x0[:nq+nv] - y_filtered[:nq+nv]) #position vel
                        tau_des_CTRL += Ktilde.dot(solver.problem.x0 - y_filtered)     # position, vel, torques
                    # Compute the motor torque 
                    tau_mot_CTRL = torqueController.step(tau_des_CTRL, tau_mea_CTRL, tau_mea_derivative_CTRL)
                    # Increment control counter
                    nb_ctrl += 1


                # Simulate actuation
                tau_mea_SIMU = actuationModel.step(i, tau_mot_CTRL, joint_vel=sim_data.state_mea_SIMU[i,nq:nq+nv])
                # Step PyBullet simulator
                robot_simulator.send_joint_command(tau_mea_SIMU)
                env.step()
                # Measure new state + forces from simulation 
                q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
                robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
                f_mea_SIMU = robot_simulator.end_effector_forces()[1][0]
                fz_mea_SIMU = np.array([f_mea_SIMU[2]])
                if(i%1000==0): 
                    logger.info("f_mea  = "+str(f_mea_SIMU))
                    
                # Compute force and position errors
                if(i >= T_CIRCLE):
                    count+=1
                    f0 = target_force[0]
                    err_fz += np.linalg.norm(fz_mea_SIMU - f0)
                    p0 = target_position[0][:2] #solver.problem.runningModels[0].differential.costs.costs['translation'].cost.residual.reference[:2]
                    err_p += np.linalg.norm(robot_simulator.pin_robot.data.oMf[id_endeff].translation[:2] - p0)
                
                # Record data (unnoised)
                y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU[lpfStateIds]]).T 
                # Simulate sensing 
                y_mea_no_noise_SIMU = sensingModel.step(y_mea_SIMU)
                # Record measurements of state, torque and forces 
                sim_data.record_simu_cycle_measured(i, y_mea_SIMU, y_mea_no_noise_SIMU, f_mea_SIMU)

            
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
            logger.warning("Force error = "+str(err_fz/count))
            logger.warning("Position error = "+str(err_p/count))
            logger.warning("count = "+str(count))
            
            save_dir = SAVE_DIR # '/home/skleff/force-feedback/data/soft_contact_article/dataset5_no_tracking' # '/tmp'
            save_name = config_name+'_bullet_'+\
                                    '_BIAS='+str(config['SCALE_TORQUES'])+\
                                    '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                                    '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                                    '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)+\
                                    '_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+\
                                    '_SEED='+str(SEEDS[n_seed])

            # Save optionally
            if(config['SAVE_DATA']):
                plot_data = sim_data.extract_data(frame_of_interest=frame_of_interest)
                save_data(plot_data, save_dir=save_dir, save_name=save_name)
                # sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)

if __name__=='__main__':
    # args = misc_utils.parse_MPC_script(sys.argv[1:])
    # main(args.SAVE_DIR)
    main(sys.argv[1], sys.argv[2])