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
The robot_simulator.pin_robot is tasked with exerting a constant normal force with its EE 
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
np.random.seed(1)
np.set_printoptions(precision=4, linewidth=180)


from core_mpc import path_utils, ocp, pin_utils, mpc_utils, misc_utils
import core_mpc.sim_utils as simulator_utils


from classical_mpc.ocp import OptimalControlProblemClassical
from classical_mpc.data import MPCDataHandlerClassical

import time
import pinocchio as pin
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

jRc = np.eye(3)
jpc = np.array([0, 0., 0.12])
jMc = pin.SE3(jRc, jpc)

def solveOCP(q, v, ddp, nb_iter, node_id_reach, target_reach, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force):
        t = time.time()
        x = np.concatenate([q, v])
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
            pass
            # If node id is valid
            if(node_id_track <= ddp.problem.T and node_id_track >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_track, ddp.problem.T+1, 1 ):
                    w = min(1.*(k + 1. - node_id_track) , 3.)
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].weight = w
        # Update OCP for contact phase
        if(TASK_PHASE == 3):
            # If node id is valid
            if(node_id_contact <= ddp.problem.T and node_id_contact >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_contact, ddp.problem.T+1, 1 ):  
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].weight = 2.
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k < ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        # print(m[k].differential.costs.costs["force"])
                        # print(m[k].differential.costs.costs["force"].cost)
                        # print(m[k].differential.costs.costs["force"].weight)
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
                    m[k].differential.costs.costs["translation"].weight = 10.
                    # m[k].differential.costs.costs["velocity"].active = True
                    # m[k].differential.costs.costs["velocity"].cost.residual.reference = pin.Motion(np.concatenate([target_velocity[k], np.zeros(3)]))
                    # m[k].differential.costs.costs["velocity"].cost.activation.weights = np.array([1., 1., 0., 1., 1., 1.])
                    # m[k].differential.costs.costs["velocity"].weight = 1.
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k < ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        # m[k].differential.costs.costs["force"].weight = force_weight
                        m[k].differential.costs.costs["force"].cost.residual.reference = fref
        # get predicted force from rigid model (careful : expressed in LOCAL !!!)
        j_wrenchpred = ddp.problem.runningDatas[0].differential.multibody.contacts.contacts['contact'].f
        fpred = jMc.actInv(j_wrenchpred).linear
        # print(fpred)
        problem_formulation_time = time.time()
        t_child_1 =  problem_formulation_time - t
        # Solve OCP 
        ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        # ddp.problem.calcDiff(ddp.xs, ddp.us)
        # Send solution to parent process + riccati gains
        solve_time = time.time()
        ddp_iter = ddp.iter
        t_child =  solve_time - problem_formulation_time
        return ddp.us, ddp.xs, ddp.K, t_child, ddp_iter, t_child_1, fpred


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
  env, robot_simulator, _ = simulator_utils.init_iiwa_reduced_bullet(dt=dt_simu, x0=x0)
  # Get dimensions 
  nq, nv = robot_simulator.pin_robot.model.nq, robot_simulator.pin_robot.model.nv; nu = nq
  # Initial placement
  id_endeff = robot_simulator.pin_robot.model.getFrameId(config['frame_of_interest'])
  ee_frame_placement = robot_simulator.pin_robot.data.oMf[id_endeff].copy()



  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  # Init shooting problem and solver
  ddp = OptimalControlProblemClassical(robot_simulator.pin_robot, config).initialize(x0, callbacks=False) 
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
          q_ws, v_ws, eps = pin_utils.IK_placement(robot_simulator.pin_robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
          xs_init.append(np.concatenate([q_ws, v_ws]))
      us_init = [pin_utils.get_u_grav(xs_init[i][:nq], robot_simulator.pin_robot.model, config['armature']) for i in range(config['N_h'])]
  # Classical warm start using initial config
  else:
      ug  = pin_utils.get_u_grav(q0, robot_simulator.pin_robot.model, config['armature'])
      xs_init = [x0 for i in range(config['N_h']+1)]
      us_init = [ug for i in range(config['N_h'])]

  # solve
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)



  for n_seed in range(N_SEEDS):
    
    print("Set Random Seed to "+str(SEEDS[n_seed]) + " ("+str(n_seed)+"/"+str(N_SEEDS)+")")
    np.random.seed(SEEDS[n_seed])

    for n_exp in range(N_EXP):
        
        print("   Set angle to "+str(TILT_ANGLES_DEG[n_exp]) + " ("+str(n_exp)+"/"+str(N_EXP)+")")
        # Reset robot_simulator.pin_robot to initial state and set table
        robot_simulator.reset_state(q0, v0)
        robot_simulator.forward_robot(q0, v0)
        contact_placement = robot_simulator.pin_robot.data.oMf[id_endeff].copy()
        offset = 0.03348 
        contact_placement.translation = contact_placement.act(np.array([0., 0., offset])) 
        # Optionally tilt the contact surface
        contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY[n_exp])
        # Create the contact surface in PyBullet simulator 
        contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement.copy(), bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
        # Set lateral friction coefficient of the contact surface
        # simulator_utils.set_friction_coef(contact_surface_bulletId, 0.5)
        # Display target circle  trajectory (reference)
        nb_points = 20 
        ballsIdTarget = np.zeros(nb_points, dtype=int)
        # for i in range(nb_points):
        #     t = (i/nb_points)*2*np.pi/OMEGA
        #     pl = ee_frame_placement.copy() #pin_utils.rotate(ee_frame_placement, rpy=TILT_RPY[n_exp])
        #     pos = ocp.circle_point_WORLD(t, pl, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
        #     ballsIdTarget[i] = simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
        draw_rate = 200
        ballsIdReal = []

        # # # # # # # # # # #
        ### INIT MPC SIMU ###
        # # # # # # # # # # #
        sim_data = MPCDataHandlerClassical(config, robot_simulator.pin_robot)
        sim_data.init_sim_data(x0)
            # Get frequencies
        nb_plan = 0
        nb_ctrl = 0
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



        # Additional simulation blocks 
        communicationModel = mpc_utils.CommunicationModel(config)
        actuationModel     = mpc_utils.ActuationModel(config, nu, SEED=SEEDS[n_seed])
        sensingModel       = mpc_utils.SensorModel(config, SEED=SEEDS[n_seed])


        # # # # # # # # # # # #
        ### SIMULATION LOOP ###
        # # # # # # # # # # # #

        for i in range(sim_data.N_simu): 

            if(i%config['log_rate']==0 and config['LOG']): 
                print('')
                logger.info("SIMU step "+str(i)+"/"+str(sim_data.N_simu))
                print('')
            

            # Solve OCP if we are in a planning cycle (MPC/planning frequency)
            if(i%int(sim_data.dt_plan/sim_data.dt_simu) == 0):
                # Current simulation time
                t_simu = i*dt_simu 
                # Setup tracking problem with circle ref EE trajectory
                # Circle defined w.r.t. NON-tilted surface (controller thinks it's all flat)
                models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
                for k,m in enumerate(models):
                    # Ref
                    t = min(t_simu + k*config['dt'], config['numberOfRounds']*2*np.pi/OMEGA)
                    p_ee_ref = ocp.circle_point_WORLD(t, ee_frame_placement.copy(), 
                                                                radius=RADIUS,
                                                                omega=OMEGA,
                                                                LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
                    # Cost translation
                    m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
                    # Contact model
                    m.differential.contacts.contacts["contact"].contact.reference = p_ee_ref
                # Reset x0 to measured state + warm-start solution
                ddp.problem.x0 = sim_data.state_mea_SIMU[i, :]
                xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
                xs_init[0] = sim_data.state_mea_SIMU[i, :]
                us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
                # Solve OCP & record MPC predictions
                ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
                # bench.record_profiles()
                # bench.stop_timer(nb_iter=ddp.iter)
                # bench.stop_croco_profiler()
                # Record MPC predictions, cost references and solver data 
                # pin.framesForwardKinematics(sim_data.rmodel, sim_data.rdata, q)
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
                # f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, softContactModel.pinRefFrame)
                f_mea_SIMU = robot_simulator.end_effector_forces(sim_data.PIN_REF_FRAME)[1][0]
                fz_mea_SIMU = np.array([f_mea_SIMU[2]])
                if(i%100==0): 
                    logger.info("f_mea  = "+str(f_mea_SIMU))
                # Record data (unnoised)
                x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
                sim_data.state_mea_no_noise_SIMU[i+1, :] = x_mea_SIMU
                # Sensor model ( simulation state ==> noised / filtered state )
                sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, x_mea_SIMU, sim_data.state_mea_SIMU)
                sim_data.force_mea_SIMU[i, :] = f_mea_SIMU
                # # Display real 
                # if(i%draw_rate==0):
                #     pos = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
                #     ballId = simulator_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])
                #     ballsIdReal.append(ballId)

        import time
        time.sleep(1)
        logger.warning("ROBOT = "+str(robot_simulator.robotId))

        logger.warning("CONTACT = "+str(contact_surface_bulletId))
        # # Remove table
        simulator_utils.remove_body_from_sim(contact_surface_bulletId)
        # for ballId in ballsIdTarget:
        #     simulator_utils.remove_body_from_sim(ballId)
        # for ballId in ballsIdReal:
        #     simulator_utils.remove_body_from_sim(ballId)

        # # # # # # # # # # #
        # PLOT SIM RESULTS  #
        # # # # # # # # # # #
        # save_dir = '/tmp'
        # save_name = config_name+'_bullet_'+\
        #                         '_BIAS='+str(config['SCALE_TORQUES'])+\
        #                         '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
        #                         '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
        #                         '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)+\
        #                         '_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+\
        #                         '_SEED='+str(SEEDS[n_seed])

        # # Save optionally
        # if(config['SAVE_DATA']):
        #     sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)


if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)