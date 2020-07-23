# Author: Sebastien
# NYU 2020
# Kalman filter

import numpy as np
import matplotlib.pyplot as plt
import time 

class KalmanFilter:
    '''
    Kalman filter environment class
    '''
    
    def __init__(self, A, B, Q, H, R):
        '''
        Initialize model & filter parameters
         Input:
          - A : state transition matrix      
          - B : control matrix
          - Q : process noise covariance matrix 
          - H : state-output matrix      
          - R : measurement noise covariance matrix
        '''

        # State transition model
        self.A = A
        self.B = B
        self.Q = Q
        # Measurement model
        self.H = H
        self.R = R
        # Dimensions
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.nz = H.shape[0]
    
    def run(self, x0, P0):
        '''
        Run fitler
        '''
        pass

    def predict(self, x_post_prev, P_post_prev, u_prev):
        '''
        Prediction step 
         Input:
          - x_post_prev : previous a posteriori state estimate      : x^+(k-1)
          - P_post_prev : previous a posteriori covariance estimate : P+(k-1)
          - u_prev      : previous control input                    : u(k-1)
          Output:
          - P_prio_curr : current a priori covariance estimate      : P-(k)
          - x_prio_curr : current a priori state estimate           : x^-(k)
        '''
        # predict state estimate
        x_prio_curr = self.A.dot(x_post_prev) + self.B.dot(u_prev)
        # predict covariance estimate
        P_prio_curr = self.A.dot(P_post_prev).dot(self.A.T) + self.Q

        return x_prio_curr, P_prio_curr

    def update(self, x_prio_curr, P_prio_curr, y_curr):
        '''
        Update step 
         Input:
          - x_prio_curr : current a priori state estimate           : x^-(k)
          - P_prio_curr : current a priori covariance estimate      : P-(k)
          - y_curr      : current measurement                       : y(k)
         Output:
          - y_res_curr  : current measurement residual (innovation) : y~(k)
          - K           : current Kalman gain                       : K(k)
          - x_post_curr : current a posteriori state estimate       : x^+(k)
          - P_post_curr : current a posteriori covariance estimate  : P+(k)
        '''
        # update innovation
        y_res_curr = y_curr - self.H.dot(x_prio_curr)
        # update Kalman gain
        K = P_prio_curr.dot(H.T).dot(np.linalg.inverse(self.R + self.H.dot(P_prio_curr).dot(self.H.T)))
        # update state estimate
        x_post_curr = x_prio_curr + K.dot(y_res_curr)
        # update covariance estimate
        P_post_curr = (np.eye(self.nx) - K.dot(self.H)).dot(P_prio_curr)

        return y_res_curr, K, x_post_curr, P_post_curr