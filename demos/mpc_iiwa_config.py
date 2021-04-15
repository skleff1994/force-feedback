'''
Config file for iiwa MPC sim
'''

import numpy as np

np.set_printoptions(precision=2, linewidth=200, suppress=True,
                    formatter={'all':lambda x: "%.3f, "%x})

# Friction coefficient, stiffness damping
mu = 0.7                              
K = 10e6
B = np.sqrt(K)
# update robot configuration in viwewer every DISPLAY_T (only for CONSIM)
DISPLAY_T = 0.001                
# Initial configuration
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
# Visualize or not (only for CONSIM)
use_viewer = 0
