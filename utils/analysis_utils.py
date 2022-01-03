import time
import numpy as np
import os
from utils import data_utils, plot_utils

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


    
# Save data (dict) into compressed npz
def moving_average_filter(input_data, filter_size=1):
    '''
    moving average on 1st dimension of some array ((N,n))
    '''
    output_data = input_data.copy() 
    # Filter them with moving average
    for i in range( output_data.shape[0] ):
        n_sum = min(i, filter_size)
        # Sum up over window
        for k in range(n_sum):
            output_data[i,:] += input_data[i-k-1, :]
        #  Divide by number of samples
        output_data[i,:] = input_data[i,:] / (n_sum + 1)
    return output_data 
    
# import sys# def main(DATASET_NAME=None, N_=1):
#     pass

# if __name__=='__main__':
#     if len(sys.argv) <= 1:
#         print("Usage: python plot_end_effector_errors < arg1: DATASET_NAME, arg2: N_EXP >")
#         sys.exit(0)
#     sys.exit(main(sys.argv[1]))


