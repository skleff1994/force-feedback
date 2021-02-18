# Title : test_IAMPM.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Test of the "augmented state" approach for force feedback on the point mass system 

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

import numpy as np
import crocoddyl

from models.dyn_models import PointMassLPF
from models.cost_models import *
from models.croco_IAMs import ActionModelPointMass

from utils import animatePointMass, plotPointMass


# Action model for point mass
dt = 1e-2
N_h = 100
# running_models = []
# for i in range(N_h):
#     md = ActionModelPointMass(dt=dt)
#     running_models.append(md)

running_model = ActionModelPointMass(dt=dt)

terminal_model = ActionModelPointMass(dt=0.)
terminal_model.w_x = 1.

# Problem + solver
x0 = np.array([1., 0.])
problem = crocoddyl.ShootingProblem(x0, [running_model]*N_h, terminal_model)
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([ crocoddyl.CallbackVerbose() ])

# Solve and retrieve X,U
done = ddp.solve([], [], 100)
X = np.array(ddp.xs)
U = np.array(ddp.us)

# PLOT
import matplotlib.pyplot as plt
p = X[:,0]
v = X[:,1]
u = U
# Create time spans for X and U
tspan_x = np.linspace(0, N_h*dt, N_h+1)
tspan_u = np.linspace(0, N_h*dt, N_h)
# Create figs and subplots
fig_x, ax_x = plt.subplots(3, 1)
# fig_u, ax_u = plt.subplots(1, 1)
# Plot joints
ax_x[0].plot(tspan_x, p, 'b-', label='pos')
ax_x[0].set(xlabel='t (s)', ylabel='p (m)')
ax_x[0].grid()
ax_x[1].plot(tspan_x, v, 'g-', label='vel')
ax_x[1].set(xlabel='t (s)', ylabel='v (m/s)')
ax_x[1].grid()
ax_x[2].plot(tspan_u, u, 'r-', label='torque')
ax_x[2].set(xlabel='t (s)', ylabel='tau (Nm)')
ax_x[2].grid()
# # If ref specified
# if(ref is not None):
#     ax_x[0].plot(tspan_x, [ref[0]]*(N+1), 'k-.', label='ref')
#     ax_x[1].plot(tspan_x, [ref[1]]*(N+1), 'k-.')
#     ax_x[2].plot(tspan_x, [ref[2]]*(N+1), 'k-.')
# ax_u.plot(tspan_u, u, 'k-', label='control') 
# ax_u.set(xlabel='t (s)', ylabel='w')
# ax_u.grid()
# Legend
handles_x, labels_x = ax_x[0].get_legend_handles_labels()
fig_x.legend(loc='upper right', prop={'size': 16})
# handles_u, labels_u = ax_u.get_legend_handles_labels()
# fig_u.legend(loc='upper right', prop={'size': 16})
# Titles
fig_x.suptitle('State - Control trajectories', size=16)
# fig_u.suptitle('Control trajectory', size=16)
plt.show()