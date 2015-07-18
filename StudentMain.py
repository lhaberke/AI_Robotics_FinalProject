# This file was taken from Udacity's Artificial Intelligence for Robotics course
#    and was modified as instructed per assignment.
#    This is not a final work.


# ----------
# Part Four
#
# Again, you'll track down and recover the runaway Traxbot. 
# But this time, your speed will be about the same as the runaway bot. 
# This may require more careful planning than you used last time.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time. 
#
# ----------
# GRADING
# 
# Same as part 3. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
from matrix import *
from copy import deepcopy
import turtle
import random

def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # This function will be called after each time the target moves. 

    # ************************* My Code Start *******************
    
    # Measurement filter needed since it is noisy.   Using None as:
    # [target_measurements, hunter_positions, hunter_headings, P]
    # where P is our uncertainty matrix
    
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement[:]]
        hunter_positions = [hunter_position[:]]
        hunter_headings = [hunter_heading]
        P = None
        last_est_xy = target_measurement[:]
        OTHER = [measurements, hunter_positions, hunter_headings, P, last_est_xy] 
    else: # not the first time, update my history
        OTHER[0].append(target_measurement) # I can change measurements w/o affecting 
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        # will update OTHER[3] = P later
        # will update OTHER[4] = last_est_xy later
        measurements, hunter_positions, hunter_headings, P, last_est_xy = OTHER[:] 
        #print 'measurement: ', measurements[-1]
        #print 'hunter_position: ', hunter_positions[-1]
        #print 'hunter_heading: ', hunter_headings[-1]
        # now I can change these without affecting OTHER
    
    est_target_xy, P_new = filter_sensor(measurements, 1, P)
    # uses the measurement to update uncertainty matrix P and give est_target_xy
    
    est_measurements = deepcopy(measurements)
    est_measurements.append(est_target_xy)
    next_est_target_xy, _ = filter_sensor(est_measurements,1,P_new)
    # uses new estimate to predict the next estimated target location
    
    hunter_to_xy = next_est_target_xy # works if target will be within range
    dist_to_target = distance_between(next_est_target_xy, hunter_position)
    for D in range(int(dist_to_target / (max_distance - .1))):
        # to catch target, look ahead D moves and go that way
        est_measurements.append(hunter_to_xy) 
        #print 'measurements: ',measurements
        hunter_to_xy, _ = filter_sensor(est_measurements,1,P_new)
    P_new_arrayed = P_new.matrix2array()
    OTHER = OTHER[:3]
    OTHER.append(P_new)
    OTHER.append(est_target_xy[:])
    turning = angle_trunc(get_heading(hunter_position, hunter_to_xy) - hunter_heading)
    distance = min(dist_to_target, max_distance)
    
    # ************************** My Code End ********************

    
    # The OTHER variable is a place for you to store any historical information about
    # the progress of the hunt (or maybe some localization information). Your return format
    # must be as follows in order to be graded properly.
    return turning, distance, OTHER

def filter_sensor(measurements, N=1, P = None):
    # Uses the last N measurements to filter with recurring P uncertainty matrix
    # Note, this needs [heading, speed] not [x,y]
    dt = 1. # just assuming a timestep of 1 for now
    
    if not P:
        # P - initial uncertainty: 0 for heading and speed, 100 for turning, acceleration
        # P can be stored for later use so we don't need to repeat old measurements
        P = matrix([[0., 0., 0.,   0.], 
                    [0., 0., 0.,   0.], 
                    [0., 0., 100., 0.], 
                    [0., 0., 0., 100.]]) 
    
    # u - external motion - none modeled
    u = matrix([[0.], [0.], [0.], [0.]])
    # F - next state function: heading2 = heading1 + turning, speed2 = speed1+accel
    F = matrix([[1., 0., dt, 0.], 
                [0., 1., 0., dt], 
                [0., 0., 1., 0.], 
                [0., 0., 0., 1.]])
    # H - measurement function: reflect the fact that we observe heading and speed only
    H = matrix([[1., 0., 0., 0.], 
                [0., 1., 0., 0.]])
    # R - measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
    R = matrix([[.1, 0.], 
                [0., .1]])
    # I - 4d identity matrix
    I = matrix([[1., 0., 0., 0.], 
                [0., 1., 0., 0.], 
                [0., 0., 1., 0.], 
                [0., 0., 0., 1.]])
    
    start = -(min(len(measurements), N))

    # x - initial state (heading, speed, turning, acceleration)   
    x = matrix([[measurements[start][0]], [measurements[start][1]], [0.], [0.]])
    #print 'x: '
    #x.show()
    for i in range(start,0): # last N measurements
        
        # prediction
        x = (F * x) + u
        P = F * P * F.transpose()
        
        # measurement update
        Z = matrix([measurements[i]])
        y = Z.transpose() - (H * x)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * y)
        P = (I - (K * H)) * P
    #print('x: ')
    #x.show()
    #print('P: ')
    #P.show()
    new_state = (x.value[0][0], x.value[1][0])

    return new_state, P
        
        
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we 
    will grade your submission."""
    max_distance = 0.98 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0

    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)
        #print 'separation: %f, turning: %f, distance: %f, %d' % (separation, turning, distance, .555)
        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()

        ctr += 1            
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught



def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    heading = angle_trunc(heading)
    return heading

def naive_next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER):
    """This strategy always tries to steer the hunter directly towards where the target last
    said it was and then moves forwards at full speed. This strategy also keeps track of all 
    the target measurements, hunter positions, and hunter headings over time, but it doesn't 
    do anything with that information."""
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = [hunter_position]
        hunter_headings = [hunter_heading]
        OTHER = (measurements, hunter_positions, hunter_headings) # now I can keep track of history
    else: # not the first time, update my history
        OTHER[0].append(target_measurement)
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        measurements, hunter_positions, hunter_headings = OTHER # now I can always refer to these variables
    
    heading_to_target = get_heading(hunter_position, target_measurement)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = max_distance # full speed ahead!
    return turning, distance, OTHER

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
    broken_robot.pendown()
    chaser_robot.pendown()
    #End of Visualization
    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
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
measurement_noise = .05*target.distance
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

#print demo_grading(hunter, target, next_move)
turtle_demo(hunter, target, next_move, None)

