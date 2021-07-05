# Number of runs
"""
@package force_feedback
@file mpc_iiwa_sim.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Closed-loop MPC for force task with the KUKA iiwa 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl in closed-loop MPC (feedback from state x=(q,v))
Using PyBullet simulator for rigid-body dynamics 
Using PyBullet GUI for visualization

The goal of this script is to simulate the low-level torque control
as well at higher frequency (5 to 20kHz) . In face of noise we should 
still recover the performance of closed-loop MPC (ICRA 2021) because 
the KUKA had a low-level torque control

Automate the simulations and data saving: 
- runs N_EXP sims for different freqs
- saves plots of x,u,p and acc error in specified subdirs of /data
- saves data dict as compressed npz , can be used later for analysis (separate script)
'''

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np  
import pinocchio as pin
import crocoddyl
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from utils import utils 
import pybullet as p
import time 


# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
    # Read config file
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config/'))
config_name = 'static_reaching_task3'
config_file = config_path+"/"+config_name+".yml"
config = utils.load_yaml_file(config_file)
    # Create a Pybullet simulation environment + set simu freq
simu_freq = config['simu_freq']         
dt_simu = 1./simu_freq
env = BulletEnvWithGround(p.GUI, dt=dt_simu)
pybullet_simulator = IiwaRobot()
env.add_robot(pybullet_simulator)
    # Create a robot instance. This initializes the simulator as well.
robot = pybullet_simulator.pin_robot
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
    # Reset robot to initial state in PyBullet + update pinocchio data
q0 = np.asarray(config['q0'])
dq0 = np.asarray(config['dq0']) 
pybullet_simulator.reset_state(q0, dq0)
pybullet_simulator.forward_robot(q0, dq0)
    # Get initial frame placement
M_ee = robot.data.oMf[id_endeff]
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")

# # # # # # # # #
### SETUP OCP ### 
# # # # # # # # #
  # OCP parameters 
dt = config['dt']                   # OCP integration step (s)               
N_h = config['N_h']                 # Number of knots in the horizon 
x0 = np.concatenate([q0, dq0])      # Initial state 
  # Construct cost function terms
   # State and actuation models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
   # State regularization
stateRegWeights = np.asarray(config['stateRegWeights'])
x_reg_ref = x0 #np.zeros(nq+nv)     
xRegCost = crocoddyl.CostModelState(state, 
                                    crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                    x_reg_ref, 
                                    actuation.nu)
print("[OCP] Created state reg cost.")
   # Control regularization
ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
u_grav = pin.rnea(robot.model, robot.data, x0[:nq], np.zeros((nv,1)), np.zeros((nq,1))) #
uRegCost = crocoddyl.CostModelControl(state, 
                                      crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                      u_grav)
print("[OCP] Created ctrl reg cost.")
   # State limits penalization
x_lim_ref  = np.zeros(nq+nv)
xLimitCost = crocoddyl.CostModelState(state, 
                                      crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(state.lb, state.ub)), 
                                      x_lim_ref, 
                                      actuation.nu)
print("[OCP] Created state lim cost.")
   # Control limits penalization
u_min = -np.asarray(config['u_lim']) 
u_max = +np.asarray(config['u_lim']) 
u_lim_ref = np.zeros(nq)
uLimitCost = crocoddyl.CostModelControl(state, 
                                        crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                        u_lim_ref)
print("[OCP] Created ctrl lim cost.")
   # End-effector placement 
# p_target = np.asarray(config['p_des']) 
# M_target = pin.SE3(M_ee.rotation.T, p_target)
desiredFramePlacement = M_ee.copy() # M_target
p_ref = desiredFramePlacement.translation.copy()
framePlacementWeights = np.asarray(config['framePlacementWeights'])
framePlacementCost = crocoddyl.CostModelFramePlacement(state, 
                                                       crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                       crocoddyl.FramePlacement(id_endeff, desiredFramePlacement), 
                                                       actuation.nu) 
print("[OCP] Created frame placement cost.")
   # End-effector velocity 
desiredFrameMotion = pin.Motion(np.array([0.,0.,0.,0.,0.,0.]))
frameVelocityWeights = np.ones(6)
frameVelocityCost = crocoddyl.CostModelFrameVelocity(state, 
                                                     crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                     crocoddyl.FrameMotion(id_endeff, desiredFrameMotion), 
                                                     actuation.nu) 
print("[OCP] Created frame velocity cost.")
# Create IAMs
runningModels = []
for i in range(N_h):
  # Create IAM 
  runningModels.append(crocoddyl.IntegratedActionModelEuler( 
      crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                       actuation, 
                                                       crocoddyl.CostModelSum(state, nu=actuation.nu)), dt ) )
  # Add cost models
  runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['frameWeight'])
  runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['xRegWeight'])
  runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['uRegWeight'])
  # runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['xLimWeight'])
  # runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['uLimWeight'])
  # Add armature
  runningModels[i].differential.armature = np.asarray(config['armature'])
  # Terminal IAM + set armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                        actuation, 
                                                        crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
   # Add cost models
terminalModel.differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeightTerminal'])
terminalModel.differential.costs.addCost("stateReg", xRegCost, config['xRegWeightTerminal'])
terminalModel.differential.costs.addCost("velocity", frameVelocityCost, 1e4)
# terminalModel.differential.costs.addCost("stateLim", xLimitCost, config['xLimWeightTerminal'])
  # Add armature
terminalModel.differential.armature = np.asarray(config['armature']) 
print("[OCP] Created IAMs.")
# Create the shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
# Creating the DDP solver 
ddp = crocoddyl.SolverFDDP(problem)
print("[OCP] OCP is ready.")
print("-------------------------------------------------------------------")

# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
# freqs = ['BASELINE', 250 , 500, 1000, 2000, 5000, 10000] #, 20000]
freqs = [250, 10000] #[ 'BASELINE', 1000, 2000, 5000, 10000]
N_EXP = 1
DATASET_NAME = 'DATASET6'

# For data analysis
data = {}
PERFORMANCE_ANALYSIS = True
FIX_RANDOM_SEED = True

if(FIX_RANDOM_SEED):
  np.random.seed(1)

# Generate one bias on torque per experiment (to make comparison fair btw freqs)
alphas = []
betas = []
for n_exp in range(N_EXP):
  alphas.append(np.random.uniform(low=config['alpha_min'], high=config['alpha_max'], size=(nq,)))
  betas.append(np.random.uniform(low=config['beta_min'], high=config['beta_max'], size=(nq,)))

# For each MPC frequency 
for MPC_frequency in freqs:

  data[str(MPC_frequency)] = {}

  # For each experiment
  for n_exp in range(N_EXP):

    # LOG
    print('######################')
    print('# ' + str(MPC_frequency) + ' Hz (exp. '+str(n_exp+1)+'/'+str(N_EXP)+') #')
    print('######################')

    # MPC & simulation parameters
    maxit = config['maxiter']
    T_tot = config['T_tot']
    if(MPC_frequency == 'BASELINE'):
      plan_freq = 1000                    
    else:
      plan_freq = MPC_frequency           # MPC re-planning frequency (Hz)
    ctrl_freq = config['ctrl_freq']       # Control - simulation - frequency (Hz)
    N_plan = int(T_tot*plan_freq)         # Total number of planning steps in the simulation
    N_ctrl = int(T_tot*ctrl_freq)         # Total number of control steps in the simulation 
    N_simu = int(T_tot*simu_freq)         # Total number of simulation steps 
    T_h = N_h*dt                          # Duration of the MPC horizon (s)
    dt_ctrl = float(1./ctrl_freq)         # Time step duration of the control loop
    dt_plan = float(1./plan_freq)         # Time step duration of planning loop
    # Initialize data
    sim_data = {}
      # MPC sim parameters
    sim_data['T_tot'] = T_tot
    sim_data['N_simu'] = N_simu
    sim_data['N_ctrl'] = N_ctrl
    sim_data['N_plan'] = N_plan
    sim_data['dt_plan'] = dt_plan
    sim_data['dt_ctrl'] = dt_ctrl
    sim_data['dt_simu'] = dt_simu
    sim_data['nq'] = nq
    sim_data['nv'] = nv
    sim_data['T_h'] = T_h
    sim_data['N_h'] = N_h
    sim_data['p_ref'] = p_ref
      # To be recorded
    sim_data['X_pred'] = np.zeros((N_plan, N_h+1, nx))     # Predicted states (output of DDP, i.e. ddp.xs)
    sim_data['U_pred'] = np.zeros((N_plan, N_h, nu))       # Predicted torques (output of DDP, i.e. ddp.us)
    sim_data['U_ref'] = np.zeros((N_ctrl, nu))             # Reference torque for motor drivers (i.e. ddp.us[0] interpolated to control frequency)
    sim_data['U_mea'] = np.zeros((N_simu, nu))             # Actuation torques (i.e. disturbed reference sent to PyBullet at simu/HF)
    sim_data['X_mea'] = np.zeros((N_simu+1, nx))           # Measured states (i.e. measured from PyBullet at simu/HF)
    sim_data['X_mea_no_noise'] = np.zeros((N_simu+1, nx))  # Measured states (i.e measured from PyBullet at simu/HF) without noise
    sim_data['K'] = np.zeros((N_plan, nq, nx))             # Ricatti gains (K_0)
    vel_U_ref = np.zeros((N_ctrl, nu))         # Desired torques (current ff output by DDP)
    vel_U_mea = np.zeros((N_simu, nu))         # Actuation torques (sent to PyBullet)
    vel_U_ref_HF = np.zeros((N_simu, nu))      # Actuation torques (sent to PyBullet)
    vel_U_mea[0,:] = np.zeros(nq)
      # Initialize PID errors
    err_u = np.zeros(nq)
    vel_err_u = np.zeros(nq)
    int_err_u = np.zeros(nq)
      # Initialize average acceleration tracking error (avg over 1ms)
    sim_data['A_err'] = np.zeros((N_ctrl, nx))
      # Initialize measured state and simulator to initial state x0 
    pybullet_simulator.reset_state(q0, dq0)
    pybullet_simulator.forward_robot(q0, dq0)
    # q_mea, v_mea = pybullet_simulator.get_state()
    # pybullet_simulator.forward_robot(q_mea, v_mea)
    x0 = np.concatenate([q0, dq0]).T
    sim_data['X_mea'][0, :] = x0
    sim_data['X_mea_no_noise'][0, :] = x0
    p0 = robot.data.oMf[id_endeff].translation.T.copy()
      # Replan & control counters
    nb_plan = 0
    nb_ctrl = 0
    # Low-level simulation parameters (actuation model)
      # Scaling of desired torque
    alpha = alphas[n_exp]
    beta = betas[n_exp]
    sim_data['alpha'] = alpha
    sim_data['beta'] = beta
      # White noise on desired torque and measured state
    var_u = 0.001*(u_max - u_min) #u_np.asarray(config['var_u']) 0.5% of range on the joint
    var_q = np.asarray(config['var_q'])
    var_v = np.asarray(config['var_v'])
      # Buffers for delays
    delay_OCP_ms = config['delay_OCP_ms']                   # in ms
    delay_OCP_cycle = int(delay_OCP_ms * 1e-3 * plan_freq)  # in planning cycles
    delay_sim_cycle = int(config['delay_sim_cycle'])        # in simu cycles
    buffer_OCP = []                                         # buffer for desired torques
    buffer_sim = []                                         # buffer for measured torque
      # Proportional-integral torque control gains
    Kp = config['Kp']*np.eye(nq)
    Ki = config['Ki']*np.eye(nq)
    Kd = config['Kd']*np.eye(nq)
      # Moving avg filter
    u_avg_filter_length = config['u_avg_filter_length']    # in HF cycles
    x_avg_filter_length = config['x_avg_filter_length']    # in HF cycles
    # Sim options
    if(MPC_frequency == 'BASELINE'):
      TORQUE_TRACKING = True                          # Activate low-level reference torque tracking (PID) 
    else:
      TORQUE_TRACKING = False                
    DELAY_SIM = config['DELAY_SIM']                   # Add delay in reference torques (low-level)
    DELAY_OCP = config['DELAY_OCP']                   # Add delay in OCP solution (i.e. ~1ms resolution time)
    SCALE_TORQUES = config['SCALE_TORQUES']           # Affine scaling of reference torque
    NOISE_TORQUES = config['NOISE_TORQUES']           # Add Gaussian noise on reference torques
    FILTER_TORQUES = config['FILTER_TORQUES']         # Moving average smoothing of reference torques
    NOISE_STATE = config['NOISE_STATE']               # Add Gaussian noise on the measured state 
    FILTER_STATE = config['FILTER_STATE']             # Moving average smoothing of reference torques
    INTERPOLATE_PLAN = config['INTERPOLATE_PLAN']     # Interpolate DDP desired feedforward torque to control frequency
    INTERPOLATE_CTRL = config['INTERPOLATE_CTRL']     # Interpolate motor driver reference torque and time-derivatives to low-level frequency 

    # # # # # # # # # # # #
    ### SIMULATION LOOP ###
    # # # # # # # # # # # #

    if(config['INIT_LOGS']):
      print('                  ***********************')
      print('                  * Simulation is ready *') 
      print('                  ***********************')        
      print('---------------------------------------------------------')
      print('- Total simulation duration            : T_tot  = '+str(T_tot)+' s')
      print('- Simulation frequency                 : f_simu = '+str(float(simu_freq/1000.))+' kHz')
      print('- Control frequency                    : f_ctrl = '+str(float(ctrl_freq/1000.))+' kHz')
      print('- Replanning frequency                 : f_plan = '+str(float(plan_freq/1000.))+' kHz')
      print('- Total # of simulation steps          : N_ctrl = '+str(N_simu))
      print('- Total # of control steps             : N_ctrl = '+str(N_ctrl))
      print('- Total # of planning steps            : N_plan = '+str(N_plan))
      print('- Duration of MPC horizon              : T_ocp  = '+str(T_h)+' s')
      print('- OCP integration step                 : dt     = '+str(dt)+' s')
      print('---------------------------------------------------------')
      print('- Simulate low-level torque control?   : TORQUE_TRACKING  = '+str(TORQUE_TRACKING))
      if(TORQUE_TRACKING):
        print('    - PID gains = \n'
            +'      Kp ='+str(Kp)+'\n'
            +'      Ki ='+str(Ki)+'\n'
            +'      Kd ='+str(Kd)+'\n')
      print('- Simulate delay in low-level torque?  : DELAY_SIM        = '+str(DELAY_SIM)+' ('+str(delay_sim_cycle)+' cycles)')
      print('- Simulate delay in OCP solution?      : DELAY_OCP        = '+str(DELAY_OCP)+' ('+str(delay_OCP_ms)+' ms)')
      print('- Affine scaling of ref. ctrl torque?  : SCALE_TORQUES    = '+str(SCALE_TORQUES))
      if(SCALE_TORQUES):
        print('    a='+str(alpha)+'\n')
        print('    b='+str(beta)+')')
      print('- Noise on torques?                    : NOISE_TORQUES    = '+str(NOISE_TORQUES))
      print('- Filter torques?                      : FILTER_TORQUES   = '+str(FILTER_TORQUES))
      print('- Noise on state?                      : NOISE_STATE      = '+str(NOISE_STATE))
      print('- Filter state?                        : FILTER_STATE     = '+str(FILTER_STATE))
      print('- Interpolate planned torque?          : INTERPOLATE_PLAN = '+str(INTERPOLATE_PLAN))
      print('- Interpolate control torque?          : INTERPOLATE_CTRL = '+str(INTERPOLATE_CTRL))
      print('---------------------------------------------------------')
      print("Simulation will start...")
      time.sleep(config['log_display_time'])

    # SIMULATE
    log_rate = 10000
    for i in range(N_simu): 

        if(i%log_rate==0): 
          print("  ")
          print("SIMU step "+str(i)+"/"+str(N_simu))

      # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
        if(i%int(simu_freq/plan_freq) == 0):
            # print("  PLAN ("+str(nb_plan)+"/"+str(N_plan)+")")
            # Reset x0 to measured state + warm-start solution
            ddp.problem.x0 = sim_data['X_mea'][i, :]
            xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
            xs_init[0] = sim_data['X_mea'][i, :]
            us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
            # Solve OCP & record MPC predictions
            ddp.solve(xs_init, us_init, maxiter=maxit, isFeasible=False)
            sim_data['X_pred'][nb_plan, :, :] = np.array(ddp.xs)
            sim_data['U_pred'][nb_plan, :, :] = np.array(ddp.us)
            # Extract desired control torque + prepare interpolation to control frequency
            x_pred_1 = sim_data['X_pred'][nb_plan, 1, :]
            u_pred_0 = sim_data['U_pred'][nb_plan, 0, :]
            # Record Ricatti gain
            sim_data['K'][nb_plan, :, :] = ddp.K[0]
            # Delay due to OCP resolution time 
            if(DELAY_OCP):
              buffer_OCP.append(u_pred_0)
              if(len(buffer_OCP)<delay_OCP_cycle): 
                pass
              else:                            
                u_pred_0 = buffer_OCP.pop(-delay_OCP_cycle)
            # Optionally interpolate to control frequency
            if(nb_plan >= 1 and INTERPOLATE_PLAN==True):
              u_pred_0_next = sim_data['U_pred'][nb_plan, 1, :]
            else:
              u_pred_0_next = u_pred_0 
            # Increment planning counter
            nb_plan += 1
            
      # If we are in a control cycle select reference torque to send to motors
        if(i%int(simu_freq/ctrl_freq) == 0):
            # print("  CTRL ("+str(nb_ctrl)+"/"+str(N_ctrl)+")")
            # Optionally interpolate desired torque to control frequency
            if(INTERPOLATE_PLAN):
              coef = float(i % int(ctrl_freq/plan_freq)) / (float(ctrl_freq/plan_freq))
              u_ref = (1-coef)*u_pred_0 + coef*u_pred_0_next   
            else:
              u_ref = u_pred_0
            # Record reference torque
            sim_data['U_ref'][nb_ctrl, :] = u_ref 
            # Optionally interpolate to HF
            if(nb_ctrl >= 1 and INTERPOLATE_CTRL):
              u_ref_prev = sim_data['U_ref'][nb_ctrl-1, :]
              vel_u_ref_prev = vel_U_ref[nb_ctrl-1, :]
            else:
              u_ref_prev = u_ref
              vel_u_ref_prev = np.zeros(nq)
            # Estimate reference torque time-derivative by finite-differences for low-level PID
            vel_u_ref = ( u_ref - u_ref_prev ) / dt_ctrl
            vel_U_ref[nb_ctrl, :] = vel_u_ref
            # vel_u_des = (U_des[nb_ctrl-4, :] - 8*U_des[nb_ctrl-3, :] + U_des[nb_ctrl-1, :] - U_des[nb_ctrl, :]) / (12*dt_ctrl)
            # Increment control counter
            nb_ctrl += 1
            
      # Simulate actuation with PI torque tracking controller (low-level control frequency)
        # Optionally interpolate reference torque to HF / let constant
        if(INTERPOLATE_CTRL):
          coef = float(i%int(simu_freq/ctrl_freq)) / float(simu_freq/ctrl_freq)
          u_ref_HF = (1-coef)*u_ref_prev + coef*u_ref  
          vel_u_ref_HF = (1-coef)*vel_u_ref_prev + coef*vel_u_ref  
        else:
          u_ref_HF = u_ref  
          vel_u_ref_HF = vel_u_ref
        vel_U_ref_HF[i,:] = vel_u_ref_HF
        # Initialize measured torque to reference torque
        if(TORQUE_TRACKING):
          u_mea = u_ref_HF - Kp.dot(err_u) - Ki.dot(int_err_u) - Kd.dot(vel_err_u)
        else:
          u_mea = u_ref_HF 
        # Actuation = scaling + noise + filtering + delay
        if(SCALE_TORQUES):
          u_mea = alpha*u_mea + beta
        if(NOISE_TORQUES):
          u_mea += np.random.normal(0., var_u)
        if(FILTER_TORQUES):
          n_sum = min(i, u_avg_filter_length)
          for k in range(n_sum):
            u_mea += sim_data['U_mea'][i-k-1, :]
          u_mea = u_mea / (n_sum + 1)
        if(DELAY_SIM):
          buffer_sim.append(u_mea)            
          if(len(buffer_sim)<delay_sim_cycle):    
            pass
          else:                          
            u_mea = buffer_sim.pop(-delay_sim_cycle)
        # Record measured torque & step simulator
        sim_data['U_mea'][i, :] = u_mea
        pybullet_simulator.send_joint_command(sim_data['U_mea'][i, :])
        p.stepSimulation()
        # Measure new state from simulation 
        q_mea, v_mea = pybullet_simulator.get_state()
        # Update pinocchio model
        pybullet_simulator.forward_robot(q_mea, v_mea)
        # Record data (unnoised)
        x_mea = np.concatenate([q_mea, v_mea]).T 
        sim_data['X_mea_no_noise'][i+1, :] = x_mea
        # Accumulate acceleration error over the control cycle
        sim_data['A_err'][nb_ctrl-1,:] += (np.abs(x_mea - x_pred_1))/float(simu_freq/ctrl_freq)
        # Optional noise + filtering
        if(NOISE_STATE):
          wq = np.random.normal(0., var_q, nq)
          wv = np.random.normal(0., var_v, nv)
          x_mea += np.concatenate([wq, wv]).T
        if(FILTER_STATE):
          n_sum = min(i, x_avg_filter_length)
          for k in range(n_sum):
            x_mea += sim_data['X_mea'][i-k-1, :]
          x_mea = x_mea / (n_sum + 1)
        # Record noised data
        sim_data['X_mea'][i+1, :] = x_mea 
        # Estimate torque time-derivative
        if(i>=1):
          vel_U_mea[i, :] = (u_mea - sim_data['U_mea'][i-1, :]) / (dt_simu)
        else:
          vel_U_mea[i, :] = np.zeros(nq)
        # Update PID errors
        if(TORQUE_TRACKING):
          err_u = sim_data['U_mea'][i, :] - u_ref_HF              
          int_err_u += err_u                             
          vel_err_u = vel_U_mea[i, :] #- vel_u_ref_HF #vel_u_ref_HF # vs vel_u_ref  

    print('--------------------------------')
    print('Simulation exited successfully !')
    print('--------------------------------')

    # # # # # # # # # # # #
    # PROCESS SIM RESULTS #
    # # # # # # # # # # # #

    # Post-process EE trajectories and record in sim data
    print('Post-processing end-effector trajectories...')
    sim_data['P_pred'] = np.zeros((N_plan, N_h+1, 3))
    for node_id in range(N_h+1):
      sim_data['P_pred'][:, node_id, :] = utils.get_p(sim_data['X_pred'][:, node_id, :nq], robot, id_endeff) - np.array([p_ref]*N_plan)
    sim_data['P_mea'] = utils.get_p(sim_data['X_mea'][:,:nq], robot, id_endeff)
    q_des = np.vstack([sim_data['X_mea'][0,:nq], sim_data['X_pred'][:,1,:nq]])
    sim_data['P_des'] = utils.get_p(q_des, robot, id_endeff)
    sim_data['P_mea_no_noise'] = utils.get_p(sim_data['X_mea_no_noise'][:,:nq], robot, id_endeff)
    
    # Process Ricatti gains
    print("Post-processing Ricatti gains...")
     # SVD
    
     # VP / SV
    # Saving params
    save_name = 'tracking='+str(TORQUE_TRACKING)+'_'+str(plan_freq)+'Hz__exp_'+str(n_exp)
    save_dir = '/home/skleff/force-feedback/data/'+DATASET_NAME+'/'+str(MPC_frequency)

    # Plots
    plot_data = utils.extract_plot_data(sim_data)
    figs = utils.plot_results(plot_data, which_plots=['x','u','a','p','K'],
                                         PLOT_PREDICTIONS=True, 
                                         pred_plot_sampling=int(plan_freq/20),
                                         SAVE=True,
                                         SAVE_DIR=save_dir,
                                         SAVE_NAME=save_name,
                                         SHOW=False,
                                         AUTOSCALE=True)

    #Save data for performance analysis as compressed .npz
    if(config['SAVE_DATA']):
      data[str(MPC_frequency)][str(n_exp)] = plot_data
      utils.save_data(plot_data, save_name=save_name, save_dir=save_dir)


# # # # # # # # # # # # # # # # # 
### PROCESS FOR PERF ANALYSIS ###
# # # # # # # # # # # # # # # # #
if(PERFORMANCE_ANALYSIS):

  # Sort and add BASELINE (1000Hz) if necessary
  if('BASELINE' in freqs):
    freqs.remove('BASELINE')
    freqs.sort(key=int)
    freqs.insert(0, 'BASELINE')

  print(freqs)
  # Process data for performance analysis along relevant axis
  pz_err_max = np.zeros((len(freqs), N_EXP))
  pz_err_max_avg = np.zeros(len(freqs))
  pz_err_res = np.zeros((len(freqs), N_EXP))
  pz_err_res_avg = np.zeros(len(freqs))
  for k, MPC_frequency in enumerate(freqs):
    for n_exp in range(N_EXP):
      # Get data
      d = data[str(MPC_frequency)][str(n_exp)]
      # Record error peak (max deviation from ref) along z axis
      pz_abs_err = np.abs(d['p_mea_no_noise'][:,2] - d['p_ref'][2])
      pz_err_max[k, n_exp] = np.max(pz_abs_err)
      pz_err_max_avg[k] += pz_err_max[k, n_exp]
      # Calculate steady-state error (avg error over last points) along z 
      length = int(N_simu/2)
      pz_err_res[k, n_exp] = np.sum(pz_abs_err[-length:])/length
      pz_err_res_avg[k] += pz_err_res[k, n_exp]
    pz_err_max_avg[k] = pz_err_max_avg[k]/N_EXP
    pz_err_res_avg[k] = pz_err_res_avg[k]/N_EXP

  # # # # # # # # # # # # 
  ### PLOT PERFORMANCE ##
  # # # # # # # # # # # # 
  import matplotlib.pyplot as plt
  # Plots
  fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Max err in z (averaged over N_EXP) , vs MPC frequency
  fig2, ax2 = plt.subplots(1, 1, figsize=(19.2,10.8)) # plot avg SS ERROR in z vs frequencies DOTS connected 
  # For each experiment plot errors 
  for k in range(len(freqs)): 
    if(freqs[k] != 'BASELINE'):
      # Color for the current freq
      coef = np.tanh(float(k) / float(len(data)) )
      col = [coef, coef/3., 1-coef, 1.]
      # For each exp plot max err , steady-state err
      for n_exp in range(N_EXP):
        # max err
        ax1.plot(freqs[k], pz_err_max[k, n_exp], marker='o', color=[coef, coef/3., 1-coef, .3]) 
        # SS err
        ax2.plot(freqs[k], pz_err_res[k, n_exp], marker='o', color=[coef, coef/3., 1-coef, .3])
      # AVG max err
      ax1.plot(freqs[k], pz_err_max_avg[k], marker='o', markersize=12, color=col, label=str(freqs[k])+' Hz')
      ax1.set(xlabel='Frequency (kHz)', ylabel='$AVG max|p_{z} - pref_{z}|$ (m)')
      # Err norm
      ax2.plot(freqs[k], pz_err_res_avg[k], marker='o', markersize=12, color=col, label=str(freqs[k])+' Hz')
      ax2.set(xlabel='Frequency (kHz)', ylabel='$AVG Steady-State Error |p_{z} - pref_{z}|$')

  # BASELINE tracking
  # For each exp plot max err , steady-state err
  if('BASELINE' in freqs):
    for n_exp in range(N_EXP):
      # max err
      ax1.plot(1000., pz_err_max[0, n_exp], marker='o', color=[0., 1., 0., .5],) 
      # SS err
      ax2.plot(1000, pz_err_res[0, n_exp], marker='o', color=[0., 1., 0., .5],) 
    # AVG max err
    ax1.plot(1000, pz_err_max_avg[0], marker='o', markersize=12, color=[0., 1., 0., 1.], label='BASELINE (1000) Hz')
    ax1.set(xlabel='Frequency (kHz)', ylabel='$AVG max|p_{z} - pref_{z}|$ (m)')
    # Err norm
    ax2.plot(1000, pz_err_res_avg[0], marker='o', markersize=12, color=[0., 1., 0., 1.], label='BASELINE (1000) Hz')
    ax2.set(xlabel='Frequency (kHz)', ylabel='$AVG Steady-State Error |p_{z} - pref_{z}|$')
  
  # Grids
  ax2.grid() 
  ax1.grid() 
  # Legend error
  handles1, labels1 = ax1.get_legend_handles_labels()
  fig1.legend(handles1, labels1, loc='upper right', prop={'size': 16})
  # Legend error norm 
  handles2, labels2 = ax2.get_legend_handles_labels()
  fig2.legend(handles2, labels2, loc='upper right', prop={'size': 16})
  # titles
  fig1.suptitle('Average peak error for EE task')
  fig2.suptitle('Average steady-state error for EE task')
  # Save, show , clean
  fig1.savefig('/home/skleff/force-feedback/data/'+DATASET_NAME+'/peak_err.png')
  fig2.savefig('/home/skleff/force-feedback/data/'+DATASET_NAME+'/resi_err.png')
  plt.show()
  plt.close('all')