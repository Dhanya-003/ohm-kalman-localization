"""Motion model and measurement models of the lab, for the state x = (x, y, vx, vy) in the world frame.

    cv_model(dt)                       A of the constant-velocity model             task 1
    process_noise(dt, sigma_a)         Q: what the model does not know (given)      task 1
    gps_measurement(msg)               z, H, R of one GPS fix                       task 1
    odom_measurement(msg, sigma_v)     z, H, R of one odometry message              task 2

Replace each `raise NotImplementedError("STUDENT ...")` by your code: the comment above it says in words
what goes there, the task in the handout gives the equation, and the `return` below shows the names to
use. The messages are ROS messages: `ros2 interface show <type>` lists their fields.
"""
import math

import numpy as np


def cv_model(dt):
    """A of the constant-velocity model over dt seconds (4 x 4)."""
    # the identity, plus dt where a position grows with its velocity
    # TODO STUDENT 1.1
    raise NotImplementedError("STUDENT 1.1")
    return A


def process_noise(dt, sigma_a):
    """Q over dt seconds: an unknown acceleration with standard deviation sigma_a (m/s^2) on each axis.

    The acceleration is constant over the step, so it moves the position by a dt^2 / 2 and the velocity by
    a dt: Q = sigma_a^2 G G^T per axis, with G = (dt^2 / 2, dt). sigma_a = 0 is Q = 0.
    """
    g = np.array([dt * dt / 2.0, dt])
    q = sigma_a ** 2 * np.outer(g, g)
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q                  # x and vx
    Q[np.ix_([1, 3], [1, 3])] = q                  # y and vy
    return Q


def gps_measurement(msg):
    """z, H, R of one fix (geometry_msgs/msg/PoseWithCovarianceStamped on /<robot>/gps_cov)."""
    # z: the measured position; H: picks x and y out of the state; R: the variances of x and y from the
    # covariance of the message (6 x 6, row by row, in the order x y z roll pitch yaw)
    # TODO STUDENT 1.5
    raise NotImplementedError("STUDENT 1.5")
    return z, H, R


def odom_measurement(msg, sigma_v):
    """z, H, R of one odometry message (nav_msgs/msg/Odometry on /<robot>/odom).

    The twist is the velocity in the frame of the robot (child_frame_id). The filter wants it in the world
    frame, so it is rotated by the heading of the odometry.
    """
    theta = yaw(msg.pose.pose.orientation)
    vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
    # z: the velocity of the robot rotated into the world frame by theta; H: picks vx and vy out of the
    # state; R: sigma_v squared on the diagonal
    # TODO STUDENT 2.1
    raise NotImplementedError("STUDENT 2.1")
    return z, H, R


def yaw(q):
    """The heading of a quaternion (geometry_msgs/msg/Quaternion) that turns only about z."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
