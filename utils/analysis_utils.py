import numpy as np

from utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

# Linear interpolator
def linear_interpolation(data, N):
    '''
    linear interpolation of trajectory with N interpolation knots
     INPUT: 
       data   : input trajectory of type np.array((N_samples, sample_dim))
       N      : number of sub-intervals bewteen 2 consecutive samples
                ( N = 1 ==> no interpolation )
     OUTPUT:
       interp : interpolated trajectory of size N_samples
    '''
    n = data.shape[0] # Number of input samples 
    d = data.shape[1] # Dimension of each input sample
    m = N*(n-1)+1     # Number of output samples (interpolated)
    interp = np.zeros((m, d))
    sample = 0        # Index of input sample 
    for i in range(m):
      coef = float(i % N) / N
      if(i > 0 and coef==0):
        sample+=1
      interp[i] = data[sample]*(1-coef) + data[min(sample+1, n-1)]*coef
    return interp 

def linear_interpolation_demo():
    '''
     Demo of linear interpolation of order N on example data
    '''
    # Generate data if None provided
    data = np.ones((10,2))
    for i in range(data.shape[0]):
        data[i] = i**2
    N = 5
    logger.info("Input data = \n")
    logger.info(data)
    # Interpolate
    logger.info("Interpolate with "+str(N)+" intermediate knots")
    interp = linear_interpolation(data, N)
    # Plot
    import matplotlib.pyplot as plt
    input, = plt.plot(np.linspace(0, 1, data.shape[0]), data[:,1], 'ro', label='input data')
    output, = plt.plot(np.linspace(0, 1, interp.shape[0]), interp[:,1], 'g*', label='interpolated')
    plt.legend(handles=[input, output])
    plt.grid()
    plt.show()


    
# Moving Average Filter
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
        output_data[i,:] = output_data[i,:] / (n_sum + 1)
    return output_data 
    
# add demo of M.A. filter

