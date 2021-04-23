'''
Config file for iiwa MPC sim
'''

import numpy as np

np.set_printoptions(precision=2, linewidth=200, suppress=True,
                    formatter={'all':lambda x: "%.3f, "%x})

# Contact normal
contact_normal = np.array([0., 0., 1.])
# Friction coefficient, stiffness damping
mu = 0.7 
stiffness_coef = 10e6
K = stiffness_coef*np.asarray(np.diagflat([1., 1., 1.]))
B = np.sqrt(stiffness_coef)*np.asarray(np.diagflat([1., 1., 1.]))                            
# update robot configuration in viwewer every DISPLAY_T (only for CONSIM)
DISPLAY_T = 0.001                
# Initial configuration
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
# Visualize or not (only for CONSIM)
use_viewer = 0
