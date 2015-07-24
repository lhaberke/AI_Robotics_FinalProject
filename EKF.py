# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:30:19 2015

@author: Luke
"""

from numpy import matrix, zeros, eye, diag, sin, cos, linalg, pi

def EKF_Motion(X = None, P = None):
    # Extended Kalman Filter Motion Estimate for nonlinear X state
    #       I am modeling with a constant velocity and yaw rate

    dt = 1.0 # time step
    max_speed = 1.5 # taken from problem in this case
    max_turn_rate = pi/4 # max of 45deg/sec
    if not X: # Initialize X statespace
        X = matrix([[0.],  # x
                    [0.],  # y
                    [0.],  # theta (x_dir is 0 deg, y_dir is 90 deg)
                    [0.],  # velocity
                    [0.]]) # d_theta (positive is counter clockwise)
    if not P: # Initialize Uncertainty Matrix - no correlation uncertainty
        P = diag([1000., 1000., 2*pi, 100., 2*pi])
        
    # Break out statespace for readability
    x, y, theta, v, d_theta = X[0][0], X[1][0], X[2][0], X[3][0], X[4][0]
    
    if abs(d_theta) < 0.0001: # Avoid divide by zero, use as if no turning
        # Using a linear FX for this case of no turning
        FX = matrix([[ x + v * dt * cos(theta)],   # basic no turn geometry
                    [ y + v * dt * sin(theta)],   # basic no turn geometry
                    [         theta          ],   # no turning so theta = theta
                    [           v            ],   # velocity is constant
                    [      0.0000001         ]])  # Avoid divide by zero in JF
    else: # Take d_theta into account with nonlinear FX
        # FX is the nonlinear F(X) that predicts motion update
            # x = x + integral(v*cos(theta + d_theta*dt) - v*cos(theta))
            # y = y + integral(v*sin(theta + d_theta*dt) - v*sin(theta))
            # theta = theta + d_theta*dt
        FX = matrix([[x + v/d_theta * ( sin(theta + d_theta*dt) - sin(theta))],
                    [ y + v/d_theta * (-cos(theta + d_theta*dt) + cos(theta))],
                    [                   theta + d_theta*dt                   ],
                    [                           v                            ],
                    [                       d_theta                         ]])
                
    # Since X = F(X), we can just set X = FX
    X = FX
    
    # Break out new estimated statespace for readability
    x, y, theta, v, d_theta = X[0][0], X[1][0], X[2][0], X[3][0], X[4][0]
    
    # JF is the linearized F(X) matrix, to get JF(X), we do partial derivatives
    #        |  dF(X[0])/dX[0]  ...  dF(X[n])/dX[0]  |
    #   JF = |       ...                   ...       |
    #        |  dF(X[0])/dX[n]  ...  dF(X[n])/dX[n]  |
    # Notice diagonals will all be 1 and lower triangle has no correlation (=0)
    JF = eye(5)
    JF[0][2] =   v/d_theta * (cos(theta + d_theta*dt) - cos(theta))
    JF[0][3] =  1./d_theta * (sin(theta + d_theta*dt) - sin(theta))
    JF[0][4] = - v/(d_theta**2) * (sin(theta + d_theta*dt) - sin(theta)) \
               + v/d_theta * dt * cos(theta + d_theta*dt)
    JF[1][2] =   v/d_theta * (sin(theta + d_theta*dt) - sin(theta))
    JF[1][3] =  1./d_theta * (-cos(theta + d_theta*dt) + cos(theta))            
    JF[1][4] = - v/(d_theta**2) * (-cos(theta + d_theta*dt) + cos(theta)) \
               + v/d_theta * dt * sin(theta + d_theta*dt)
    JF[2][4] = dt
        
    # Various motion noise for Q
    x_var = y_var = max_speed*dt    # set for max speed
    theta_var = max_turn_rate*dt    # Assuming max turn in a step
    v_var = max_speed               # set for max speed
    d_theta_var = max_turn_rate     # assuming low acceleration
    
    # Q is the Motion Uncertainty Matrix, I'll use max step changes for now.
    #       Assuming no correlation to motion noise for now
    Q = diag([x_var**2, y_var**2, theta_var**2, v_var**2, d_theta_var**2])
    
    # Update Probability Matrix
    P = JF * P * JF.T + Q
    
    estimate_xy = [X[0][0], X[1][0]]
    
    return estimate_xy, X, P
    
def EKF_Measurement(measurement = [0., 0.], X = None, P = None):
    # Extended Kalman Filter Measurement Estimate for nonlinear X state
    #       I am modeling with a constant velocity and yaw rate

    xy_noise_var = 0.2 # taken from problem description
    dt = 1.0 # time step
    
    if not X: # Initialize X statespace
        X = matrix([[0.],  # x
                    [0.],  # y
                    [0.],  # theta (x_dir is 0 deg, y_dir is 90 deg)
                    [0.],  # velocity
                    [0.]]) # d_theta (positive is counter clockwise)
    if not P: # Initialize Uncertainty Matrix - no correlation uncertainty
        P = diag([1000., 1000., 2*pi, 100., 2*pi])
        
    # Z is the measurement itself, currently only measure x,y
    Z = matrix([[float(measurement[0])],  # Only measures x, y.  Add sensors
                [float(measurement[1])]]) # for theta, etc
                 
    # Break out statespace for readability
    x, y, theta, v, d_theta = X[0][0], X[1][0], X[2][0], X[3][0], X[4][0]    
    
    # HF is the nonlinear (linear if only gps sensor) measurement matrix
    HF = matrix([[x],  # Only measures x, y.  Add sensors
                 [y]])  # for theta, etc
    
    # JH is linearized jacobian of H
    JH = matrix([[1., 0., 0., 0., 0.],  # x row
                 [0., 1., 0., 0., 0.],
                 [0., 1., 0., 0., 0.],
                 [0., 1., 0., 0., 0.],
                 [0., 1., 0., 0., 0.]]) # y row. add rows for more sensors

    # R is the measurement noise matrix.  Using problem's x,y noise.
    R = matrix([[xy_noise_var**2, 0., 0., 0., 0.],
                [0., xy_noise_var**2, 0., 0., 0.],
                [0.,    1.,    0.,    0.,     0.],
                [0.,    1.,    0.,    0.,     0.],
                [0.,    1.,    0.,    0.,     0.]])    
    
    I = eye(5)
         
    S = JH * P * JH.T + R
    
    # Kalman factor - correction matrix
    K = (P * JH.T) * linalg.inv(S)
    
    # Y is the error matrix (measurement - estimate)
    Y = Z - HF
    X = X + (K * Y)
    
    # Probability matrix will get more precise with measurement
    P = (I - (K * JH)) * P
    
    estimate_xy = [X[0][0], X[1][0]]
    
    return estimate_xy, X, P
    
    