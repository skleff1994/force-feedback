"""
@package force_feedback
@file LPF_circle_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop MPC for tracking a circle trajectory (LPF)
"""

'''
The robot is tasked with tracking a circle trajectory 
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateLPF y=(q,v,tau), control u = w 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate closed-loop MPC on a simple reaching task 
'''


import sys
sys.path.append('.')

import numpy as np  
from utils import path_utils, ocp_utils, pin_utils, plot_utils, data_utils, mpc_utils
np.random.seed(1)
np.set_printoptions(precision=4, linewidth=180)


import logging
FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
logging.basicConfig(format=FORMAT_SHORT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TASK = 'circle'
WARM_START_IK = True


def main(robot_name='iiwa', simulator='bullet', PLOT_INIT=False):


    # # # # # # # # # # # # # # # # # # #
    ### LOAD ROBOT MODEL and SIMU ENV ### 
    # # # # # # # # # # # # # # # # # # # 
    # Read config file
    config_name = robot_name+'_LPF_'+TASK+'_MPC'
    config      = path_utils.load_config_file(config_name)
    # Create a simulation environment & simu-pin wrapper 
    dt_simu = 1./float(config['simu_freq'])  
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    if(simulator == 'bullet'):
        from utils import sim_utils as simulator_utils
        env, robot_simulator = simulator_utils.init_bullet_simulation(robot_name, dt=dt_simu, x0=x0)
        robot = robot_simulator.pin_robot
    elif(simulator == 'raisim'):
        from utils import raisim_utils as simulator_utils
        env, robot_simulator = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
        robot = robot_simulator
    else:
        logger.error('Please choose a simulator from ["bullet", "raisim"] !')
    # Initial placement
    id_endeff = robot.model.getFrameId(config['frame_of_interest'])
    nq, nv = robot.model.nq, robot.model.nv; ny = nq+nv+nq; nu = nq
    M_ee = robot.data.oMf[id_endeff].copy()



    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    N_h = config['N_h']
    dt = config['dt']
    # Setup Croco OCP and create solver
    ug = pin_utils.get_u_grav(q0, robot.model) 
    y0 = np.concatenate([x0, ug])
    ddp = ocp_utils.init_DDP_LPF(robot, config, y0, callbacks=False, 
                                                    w_reg_ref=np.zeros(nq), #'gravity',
                                                    TAU_PLUS=False, 
                                                    LPF_TYPE=config['LPF_TYPE'],
                                                    WHICH_COSTS=config['WHICH_COSTS'] ) 
    # Setup tracking problem with circle ref EE trajectory
    models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    RADIUS = config['frameCircleTrajectoryRadius'] 
    OMEGA  = config['frameCircleTrajectoryVelocity']
    for k,m in enumerate(models):
        # Ref
        t = min(k*config['dt'], 2*np.pi/OMEGA)
        p_ee_ref = ocp_utils.circle_point_WORLD(t, M_ee.copy(), 
                                                    radius=RADIUS,
                                                    omega=OMEGA,
                                                    LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
        # Cost translation
        m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
    
    # Warm start state = IK of circle trajectory
    if(WARM_START_IK):
        logger.info("Computing warm-start using Inverse Kinematics...")
        xs_init = [] 
        us_init = []
        q_ws = q0
        for k,m in enumerate(list(ddp.problem.runningModels) + [ddp.problem.terminalModel]):
            # Get ref placement
            p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
            Mref = M_ee.copy()
            Mref.translation = p_ee_ref        
            q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
            tau_ws = pin_utils.get_u_grav(q_ws, robot.model)
            xs_init.append(np.concatenate([q_ws, v_ws, tau_ws]))
            if(k<N_h):
                us_init.append(tau_ws)
    
    # Classical warm start using initial config
    else:
        xs_init = [y0 for i in range(config['N_h']+1)]
        us_init = [ug for i in range(config['N_h'])]
    
    # Solve 
    ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
    

    # Plot initial solution
    if(PLOT_INIT):
        ddp_data = data_utils.extract_ddp_data_LPF(ddp, frame_of_interest=config['frame_of_interest'])
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
    WHICH_PLOTS       = config['WHICH_PLOTS']                   # Which plots to generate ? ('y':state, 'w':control, 'p':end-eff, etc.)
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
    actuation     = mpc_utils.ActuationModel(config)
    sensing       = mpc_utils.SensorModel(config)


    # Display target circle
    nb_points = 20 
    for i in range(nb_points):
        t = (i/nb_points)*2*np.pi/OMEGA
        # if(i%20==0):
        pos = ocp_utils.circle_point_WORLD(t, M_ee, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
        simulator_utils.display_ball(pos, RADIUS=0.02)

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

    # If the current simulation cycle matches an OCP node, update tracking problem
        if(i%int(1./OCP_TO_SIMU_RATIO)==0):
            # Current simulation time
            t_simu = i*dt_simu 
            # Setup tracking problem with circle ref EE trajectory
            models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
            for k,m in enumerate(models):
                # Ref
                t = min(t_simu + k*dt_ocp, 2*np.pi/OMEGA)
                p_ee_ref = ocp_utils.circle_point_WORLD(t, M_ee.copy(), 
                                                        radius=RADIUS,
                                                        omega=OMEGA,
                                                        LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
                # Cost translation
                m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref.copy()

    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
        if(i%int(freq_SIMU/freq_PLAN) == 0):       
            # Reset x0 to measured state + warm-start solution
            ddp.problem.x0 = sim_data['state_mea_SIMU'][i, :]
            xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
            xs_init[0] = sim_data['state_mea_SIMU'][i, :]
            us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
            # Solve OCP & record MPC predictions
            ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
            sim_data['state_pred'][nb_plan, :, :] = np.array(ddp.xs)
            sim_data['ctrl_pred'][nb_plan, :, :] = np.array(ddp.us)
            # Extract relevant predictions for interpolations
            y_curr = sim_data['state_pred'][nb_plan, 0, :]    # y0* = measured state    (q^,  v^ , tau^ )
            y_pred = sim_data['state_pred'][nb_plan, 1, :]    # y1* = predicted state   (q1*, v1*, tau1*) 
            w_curr = sim_data['ctrl_pred'][nb_plan, 0, :]    # w0* = optimal control   (w0*) !! UNFILTERED TORQUE !!
            # w_pred = sim_data['ctrl_pred'][nb_plan, 1, :]  # w1* = predicted optimal control   (w1*) !! UNFILTERED TORQUE !!
            # Record cost references
            data_utils.record_cost_references_LPF(ddp, sim_data, nb_plan)
            # Record solver data (optional)
            if(config['RECORD_SOLVER_DATA']):
                data_utils.record_solver_data(ddp, sim_data, nb_plan) 
            # Model communication between computer --> robot
            y_pred, w_curr = communication.step(y_pred, w_curr)
            # Select reference control and state for the current PLAN cycle
            y_ref_PLAN  = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
            w_ref_PLAN  = w_curr
            if(nb_plan==0):
                sim_data['state_des_PLAN'][nb_plan, :] = y_curr  
                sim_data['ctrl_des_PLAN'][nb_plan, :]   = w_ref_PLAN   
                sim_data['state_des_PLAN'][nb_plan+1, :] = y_ref_PLAN    

            # Increment planning counter
            nb_plan += 1

    # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
        if(i%int(freq_SIMU/freq_CTRL) == 0):        
            # print("  CTRL ("+str(nb_ctrl)+"/"+str(sim_data['N_ctrl'])+")")
            # Select reference control and state for the current CTRL cycle
            COEF       = float(i%int(freq_CTRL/freq_PLAN)) / float(freq_CTRL/freq_PLAN)
            y_ref_CTRL = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
            w_ref_CTRL = w_curr 
            # First prediction = measurement = initialization of MPC
            if(nb_ctrl==0):
                sim_data['state_des_CTRL'][nb_ctrl, :] = y_curr  
                sim_data['ctrl_des_CTRL'][nb_ctrl, :]   = w_ref_CTRL  
                sim_data['state_des_CTRL'][nb_ctrl+1, :] = y_ref_CTRL   
            # Increment control counter
            nb_ctrl += 1
            
    # Simulate actuation/sensing and step simulator (physics simulation frequency)

        # Select reference control and state for the current SIMU cycle
        COEF        = float(i%int(freq_SIMU/freq_PLAN)) / float(freq_SIMU/freq_PLAN)
        y_ref_SIMU  = y_curr + OCP_TO_PLAN_RATIO * (y_pred - y_curr)
        w_ref_SIMU  = w_curr 

        # First prediction = measurement = initialization of MPC
        if(i==0):
            sim_data['state_des_SIMU'][i, :] = y_curr  
            sim_data['ctrl_des_SIMU'][i, :]   = w_ref_SIMU  
            sim_data['state_des_SIMU'][i+1, :] = y_ref_SIMU 

        # Torque applied by motor on actuator : interpolate current torque and predicted torque 
        tau_ref_SIMU =  y_ref_SIMU[-nu:] 
        # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
        tau_mea_SIMU = actuation.step(i, tau_ref_SIMU, sim_data['state_mea_SIMU'][:,-nu:])   
        # Send output of actuation torque to the RBD simulator 
        robot_simulator.send_joint_command(tau_mea_SIMU)
        env.step()
        # Measure new state from simulation :
        q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
        # Update pinocchio model
        robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
        # Record data (unnoised)
        y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU]).T 
        sim_data['state_mea_no_noise_SIMU'][i+1, :] = y_mea_SIMU
        # Sensor model (optional noise + filtering)
        sim_data['state_mea_SIMU'][i+1, :] = sensing.step(i, y_mea_SIMU, sim_data['state_mea_SIMU'])

        # Display real 
        if(i%draw_rate==0):
            pos = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
            simulator_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])


    print('--------------------------------')
    print('Simulation exited successfully !')
    print('--------------------------------')




    # # # # # # # # # # #
    # PLOT SIM RESULTS  #
    # # # # # # # # # # #
    save_dir = '/home/skleff/force-feedback/data'
    save_name = config_name+'_'+simulator+'_'+\
                            '_BIAS='+str(config['SCALE_TORQUES'])+\
                            '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                            '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                            '_Fp='+str(freq_PLAN/1000)+'_Fc='+str(freq_CTRL/1000)+'_Fs'+str(freq_SIMU/1000)

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
    # Save optionally
    if(config['SAVE_DATA']):
        data_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)



if __name__=='__main__':
    if(len(sys.argv) < 2 or len(sys.argv) > 3):
        print("Usage: python LPF_circle_MPC.py [arg1: robot_name (str)] [arg2: simulator (str)] [arg3: PLOT_INIT (bool)]")
        sys.exit(0)
    elif(len(sys.argv)==2):
        sys.exit(main(str(sys.argv[1])))
    elif(len(sys.argv)==3):
        sys.exit(main(str(sys.argv[1]), str(sys.argv[2])))
    elif(len(sys.argv)==4):
        sys.exit(main(str(sys.argv[1]), str(sys.argv[2]), bool(sys.argv[3])))