# ----------
# Part Five
#
# This time, the sensor measurements from the runaway Traxbot will be VERY 
# noisy (about twice the target's stepsize). You will use this noisy stream
# of measurements to localize and catch the target.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time. 
#
# ----------
# GRADING
# 
# Same as part 3 and 4. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
#from matrix import *
from copy import deepcopy
import turtle
import random
import time
#import EKF

def next_move(hunter_position, hunter_heading, target_measurement, 
              max_distance, OTHER = None):
    # This function will be called after each time the target moves. 

    # ************************* My Code Start *******************
    
    # Measurement filter needed since it is noisy.   Using None as:
    # [target_measurements, hunter_positions, hunter_headings, P]
    # where P is our uncertainty matrix
    
    noise_est = 60. # should be greater than noise variance
    if not OTHER: # first time calling this function, set up my OTHER variables.
        last_est_xy = target_measurement[:]
        X = None
        P = None
        OTHER = [last_est_xy, X, P]
    else: # not the first time, update my history
        last_est_xy, X, P = OTHER[:]

    est_target_xy, X, P = \
            EKF_Measurement(target_measurement, X, P, 1., noise_est)
            #EKF.EKF_Measurement(target_measurement, X, P, 1., noise_est)
    # Best guess as to true target coordinates now
    #print 'est: ', est_target_xy, ', meas: ', target_measurement
    #next_est_target_xy, X, P = EKF.EKF_Motion(X, P, dt=1.)
    next_est_target_xy, X, P = EKF_Motion(X, P, dt=1.)
    # Uses new estimate to predict the next estimated target location
    
    hunter_to_xy = next_est_target_xy # works if target will be within range
    dist_to_target = distance_between(next_est_target_xy, hunter_position)
    X_next, P_next = X.copy(), P.copy()
    
    for D in range(int(dist_to_target / (max_distance))):
        # to catch target, look ahead D moves and go that way
        # Don't update P since we have no real information to update with 
        #hunter_to_xy, X_next, _ = EKF.EKF_Motion(X_next, P_next, 1.)
        hunter_to_xy, X_next, _ = EKF_Motion(X_next, P_next, 1.)
    #print hunter_to_xy    
    turning = angle_trunc(get_heading(hunter_position, hunter_to_xy) - hunter_heading)
    distance = min(dist_to_target, max_distance)
    OTHER = [next_est_target_xy, X, P]
    # ************************** My Code End ********************

    
    # The OTHER variable is a place for you to store any historical information about
    # the progress of the hunt (or maybe some localization information). Your return format
    # must be as follows in order to be graded properly.
    return turning, distance, OTHER
    
    
    
from numpy import zeros, eye, diag, sin, cos, linalg, pi, matrix
import pylab

def EKF_Motion(X = None, P = None, dt = 0.):
    # Extended Kalman Filter Motion Estimate for nonlinear X state
    #       I am modeling with a constant velocity and yaw rate

    if not dt: dt = 1.0 # time step
    
    max_speed = 1.5 # taken from problem in this case
    max_turn_rate = pi/8 # max of 22.5deg/sec
    
    # Various motion noise for Q
    x_var = y_var = max_speed*dt    # set for max speed
    theta_var = max_turn_rate*dt    # Assuming max turn in a step
    v_var = max_speed               # set for max speed
    d_theta_var = .05               # assuming low acceleration
    
    if type(X) == type(None): # Initialize X statespace
        X = matrix([[0.],  # x
                    [0.],  # y
                    [0.],  # theta (x_dir is 0 deg, y_dir is 90 deg)
                    [0.],  # velocity
                    [0.]]) # d_theta (positive is counter clockwise)
    if type(P) == type(None): # Initialize Uncertainty Matrix - no correlation uncertainty
        P = diag([1000., 1000., 2*pi, 100., 2*pi])
        
    # Break out statespace for readability
    x, y, theta, v, d_theta = X[0,0], X[1,0], X[2,0], X[3,0], X[4,0]
    
    if abs(d_theta) < 0.0001: # Avoid divide by zero, use as if no turning
        # Using a linear FX for this case of no turning
        FX = matrix([[x + v * dt * cos(theta)],   # basic no turn geometry
                     [y + v * dt * sin(theta)],   # basic no turn geometry
                     [        theta          ],   # no turning so theta = theta
                     [          v            ],   # velocity is constant
                     [     0.0000001         ]])  # Avoid divide by zero in JF
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
    x, y, theta, v, d_theta = X[0,0], X[1,0], X[2,0], X[3,0], X[4,0]
    
    # JF is the linearized F(X) matrix, to get JF(X), we do partial derivatives
    #        |  dF(X[0])/dX[0]  ...  dF(X[n])/dX[0]  |
    #   JF = |       ...                   ...       |
    #        |  dF(X[0])/dX[n]  ...  dF(X[n])/dX[n]  |
    # Notice diagonals will all be 1 and lower triangle has no correlation (=0)
    JF = eye(5)
    JF[0,2] =   v/d_theta * (cos(theta + d_theta*dt) - cos(theta))
    JF[0,3] =  1./d_theta * (sin(theta + d_theta*dt) - sin(theta))
    JF[0,4] = - v/(d_theta**2) * (sin(theta + d_theta*dt) - sin(theta)) \
               + v/d_theta * dt * cos(theta + d_theta*dt)
    JF[1,2] =   v/d_theta * (sin(theta + d_theta*dt) - sin(theta))
    JF[1,3] =  1./d_theta * (-cos(theta + d_theta*dt) + cos(theta))            
    JF[1,4] = - v/(d_theta**2) * (-cos(theta + d_theta*dt) + cos(theta)) \
               + v/d_theta * dt * sin(theta + d_theta*dt)
    JF[2,4] = dt
    
    # Q is the Motion Uncertainty Matrix, I'll use max step changes for now.
    #       Assuming no correlation to motion noise for now
    Q = diag([x_var**2, y_var**2, theta_var**2, v_var**2, d_theta_var**2])
    
    # Update Probability Matrix
    P = JF * P * JF.T + Q
    
    estimate_xy = [X[0,0], X[1,0]]
    
    return estimate_xy, X, P
    
def EKF_Measurement(measurement=[0.,0.], X=None, P=None, dt=0, noise_est=0):
    # Extended Kalman Filter Measurement Estimate for nonlinear X state
    #       I am modeling with a constant velocity and yaw rate

    # How much measurement noise to account for? 
    #   Higher means rely more on average of data and motion prediction
    #   Try to set low but high enough for estimates not to diverge.
    #   ie try 2x-5x the gauss variation
    if noise_est: xy_noise_var = noise_est
    else: xy_noise_var = 20.  
    
    if not dt: dt = 1.0 # time step
    
    if type(X) == type(None): # Initialize X statespace
        X = matrix([[0.],  # x
                    [0.],  # y
                    [0.],  # theta (x_dir is 0 deg, y_dir is 90 deg)
                    [0.],  # velocity
                    [0.]]) # d_theta (positive is counter clockwise)
    if type(P) == type(None): # Initialize Uncertainty Matrix - no correlation uncertainty
        P = diag([1000., 1000., 2.*pi, 100., 2.*pi])
        
    # Z is the measurement itself, currently only measure x,y
    Z = matrix([[float(measurement[0])],  # Only measures x, y.  Add sensors
                [float(measurement[1])],
                [         0.          ],
                [         0.          ],
                [         0.          ]]) # for theta, etc
                 
    # Break out statespace for readability
    x, y, theta, v, d_theta = X[0,0], X[1,0], X[2,0], X[3,0], X[4,0]    
    
    # HF is the nonlinear (linear if only gps sensor) measurement matrix
    HF = matrix([[x],  # Only measures x, y.  Add sensors
                 [y],
                 [0.],
                 [0.],
                 [0.]])  # for theta, etc
    
    # JH is linearized jacobian of H
    JH = matrix([[1., 0., 0., 0., 0.],  # x row
                 [0., 1., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]) # y row. add rows for more sensors

    # R is the measurement noise matrix.  Using problem's x,y noise.
    R = matrix([[xy_noise_var**2, 0., 0., 0., 0.],
                [0., xy_noise_var**2, 0., 0., 0.],
                [0.,    0., 0.001,    0.,     0.],
                [0.,    0.,    0., 0.001,     0.],
                [0.,    0.,    0.,    0.,  0.001]])    
    
    I = eye(5)
         
    S = JH * P * JH.T + R
    
    # Kalman factor - correction matrix
    K = (P * JH.T) * linalg.inv(S)
    
    # Y is the error matrix (measurement - estimate)
    Y = Z - HF
    X = X + (K * Y)
    
    # Probability matrix will get more precise with measurement
    P = (I - (K * JH)) * P
    
    estimate_xy = [X[0,0], X[1,0]]
    
    return estimate_xy, X, P
    
def EKF_Example():
    # Just an example of EKF in work
    import random  
    from math import sqrt  
    
    print 'Running an EKF on a rover running along y=2x.'
    meas_sigma = float(input('What measurement variance do you want?: '))
    noise_est = float(input('What measurement noise factor do you want?: '))

    X = None
    P = None
    xmot = []
    ymot = []
    xmeas = []
    ymeas = []
    sum_error = 0
    for i in range(100):
        measurement = [random.gauss(float(i), meas_sigma), 
                       random.gauss(float(2*i), meas_sigma)]
        estimate, X, P = EKF_Measurement(measurement, X, P, 1., noise_est)
        xmot.append(estimate[0])
        ymot.append(estimate[1])
        estimate, X, P = EKF_Motion(X, P, 1.)
        xmeas.append(estimate[0])
        ymeas.append(estimate[1])
        sum_error += sqrt((float(i)-estimate[0])**2+(float(2*i)-estimate[1])**2)
    pylab.figure()
    pylab.plot(xmot, ymot, 'bo')
    pylab.plot(xmeas,ymeas,'r+')
    print 'Sum_error = ', sum_error
    print 'Blue circles are state estimates post measurement update'
    print 'Red plus signs are state estimates post motion update'
          
def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(from_position, to_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    from_x, from_y = from_position
    to_x, to_y = to_position
    heading = atan2(to_y - from_y, to_x - from_x)
    heading = angle_trunc(heading)
    return heading        
        
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def turtle_demo(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we 
    will grade your submission."""
    max_distance = 0.98 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0
    #For Visualization

    window = turtle.Screen()
    window.reset()
    window.bgcolor('white')
    chaser_robot = turtle.Turtle()
    chaser_robot.shape('arrow')
    chaser_robot.color('blue')
    chaser_robot.resizemode('user')
    chaser_robot.shapesize(0.3, 0.3, 0.3)
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.3, 0.3, 0.3)
    size_multiplier = 15.0 #change size of animation
    chaser_robot.hideturtle()
    chaser_robot.penup()
    chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
    chaser_robot.showturtle()
    broken_robot.hideturtle()
    broken_robot.penup()
    broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
    broken_robot.showturtle()
    measuredbroken_robot = turtle.Turtle()
    measuredbroken_robot.shape('circle')
    measuredbroken_robot.color('red')
    measuredbroken_robot.penup()
    measuredbroken_robot.resizemode('user')
    measuredbroken_robot.shapesize(0.1, 0.1, 0.1)
    EKF_broken_robot = turtle.Turtle()
    EKF_broken_robot.shape('turtle')
    EKF_broken_robot.color('red')
    EKF_broken_robot.penup()
    EKF_broken_robot.resizemode('user')
    EKF_broken_robot.shapesize(0.1, 0.1, 0.1)
    broken_robot.pendown()
    chaser_robot.pendown()
    EKF_broken_robot.pendown()
    #End of Visualization
    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        print 'step: %5d, separation: %5f' % (ctr, separation)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()
        #Visualize it
        measured_estimate = OTHER[0]
        EKF_broken_robot.setheading(target_bot.heading*180/pi)
        EKF_broken_robot.goto(measured_estimate[0]*size_multiplier, measured_estimate[1]*size_multiplier-100)
        EKF_broken_robot.stamp()
        measuredbroken_robot.setheading(target_bot.heading*180/pi)
        measuredbroken_robot.goto(target_measurement[0]*size_multiplier, target_measurement[1]*size_multiplier-100)
        measuredbroken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
        chaser_robot.setheading(hunter_bot.heading*180/pi)
        chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
        #End of visualization
        ctr += 1            
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught


target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
measurement_noise = 2.0*target.distance # VERY NOISY!!
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

#print demo_grading(hunter, target, next_move)
turtle_demo(hunter, target, next_move)#, None)





