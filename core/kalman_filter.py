# Title : kalman_filter.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Custom Kalman filter

import numpy as np
import matplotlib.pyplot as plt
import time 

class KalmanFilter:
    '''
    Kalman filter environment class
    '''
    
    def __init__(self, model, Q, R):
        '''
        Initialize model & filter parameters
         Input:
          - model : discrete linear system (from dyn_models or croco_IAMs) 
          - Q     : process noise covariance matrix 
          - H     : state-output matrix      
          - R     : measurement noise covariance matrix
        '''

        # State transition model 
        # x(n+1) = Ax(n) + Bu(n) + w(n) where w(n)~N(0,Q)
        self.A = model.Ad
        self.B = model.Bd
        self.Q = Q
        # Measurement model
        # y(n) = Hx(n) + v(n) where v(n)~N(0,R)
        self.H = model.Hd
        self.R = R
        # Dimensions
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.H.shape[0]
    
    def step(self, x, P, u, y):
        '''
        Take a filtering step : predict + update
         Input:
          - x : previous state estimate      : x^+(k-1)
          - P : previous covariance estimate : P+(k-1)
          - u : previous control input       : u(k-1)
          - y : new measurement              : y(k)
         Output:
          - x_post_curr : new state estimate       : x^+(k)
          - P_post_curr : new covariance estimate  : P+(k)
          - K           : new Kalman gain          : K(k)
          - y_res_curr  : new measurement residual : y~(k)
        '''
        # Predict (state and covariance based on previous estimates and model)
        x_pred, P_pred = self.predict(x, P, u)
        # Update predictions based on latest measurement
        x_next, P_next, K, y_err = self.update(x_pred, P_pred, y)
        return x_next, P_next, K, y_err

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
        # Return predicted state estimate and covariance
        return x_prio_curr, P_prio_curr

    def update(self, x_prio_curr, P_prio_curr, y_curr):
        '''
        Update step 
         Input:
          - x_prio_curr : current a priori state estimate           : x^-(k)
          - P_prio_curr : current a priori covariance estimate      : P-(k)
          - y_curr      : current measurement                       : y(k)
         Output:
          - x_post_curr : current a posteriori state estimate       : x^+(k)
          - P_post_curr : current a posteriori covariance estimate  : P+(k)
          - K           : current Kalman gain                       : K(k)
          - y_res_curr  : current measurement residual (innovation) : y~(k)
        '''

        # update innovation
        y_res_curr = y_curr - self.H.dot(x_prio_curr)
        # update Kalman gain
        K = P_prio_curr.dot(self.H.T).dot(np.linalg.inv(self.R + self.H.dot(P_prio_curr).dot(self.H.T)))
        # update state estimate
        x_post_curr = x_prio_curr + K.dot(y_res_curr)
        # update covariance estimate
        P_post_curr = (np.eye(self.nx) - K.dot(self.H)).dot(P_prio_curr)
        # Return measurement residual, Kalman gain, updated state estimate and covariance
        return x_post_curr, P_post_curr, K, y_res_curr