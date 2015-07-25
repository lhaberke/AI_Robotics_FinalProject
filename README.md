This is my Final Project assignment for the Udacity AI for Robotics course.

Here's a link to it working: https://youtu.be/afsXm6Om7ck
This is obviously a work in progress...

I decided to try an extended kalman filter on it using a constant velocity/yaw rate model.
I used the following tutorial to help me with this filter.  
	https://github.com/balzer82/Kalman
Thank you for the great work, Dresden!

For catching up to the rogue bot, I just use the EKF motion portion to predict its next N positions and use the hunter's max speed to figure out how to intercept it.

If you are in this class, please don't just copy the code.
 - you'll get more out of it by going through the struggle and...
 - you can do a heck of a lot better than this.
 - my first version used a PID with no kalman filtering at all and did just as well.