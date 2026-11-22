"""Motion models and measurement models of the lab.

Tasks 1 and 2, the linear filter, state x = (x, y, vx, vy) in the world frame:
    cv_model(dt)                       A of the constant-velocity model             task 1
    process_noise(dt, sigma_a)         Q: what the model does not know (given)      task 1
    gps_measurement(msg)               z, H, R of one GPS fix                       task 1
    odom_measurement(msg, sigma_v)     z, H, R of one odometry message              task 2

Task 3, the EKF, state x = (x, y, theta), input u = (vx, vy, omega) of the odometry in the frame of the robot:
    motion_model(x, u, dt)             f(x, u): the state after dt seconds          task 3
    motion_jacobian(x, u, dt)          F = df/dx at x                               task 3
    input_noise(x, dt, ...)            Q from the noise of u (given)                task 3

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


def motion_model(x, u, dt):
    """f(x, u): drive dt seconds with the velocity u of the robot, starting at x = (x, y, theta)."""
    px, py, theta = x
    vx, vy, omega = u
    # the velocity of the robot turned into the world frame by theta moves the position, omega turns theta
    # TODO STUDENT 3.1
    raise NotImplementedError("STUDENT 3.1")
    x_new[2] = math.atan2(math.sin(x_new[2]), math.cos(x_new[2]))      # theta stays in -pi .. pi
    return x_new


def motion_jacobian(x, u, dt):
    """F = df/dx of motion_model at x (3 x 3): how a small change of x, y, theta changes the new state."""
    theta = x[2]
    vx, vy, _ = u
    # the identity, plus the derivatives of the new x and y by theta in the third column
    # TODO STUDENT 3.2
    raise NotImplementedError("STUDENT 3.2")
    return F


def input_noise(x, dt, sigma_v, sigma_w):
    """Q of one EKF step: the noise of u (sigma_v on vx and vy, sigma_w on omega) carried into the state.

    Q = G M G^T with M = diag(sigma_v^2, sigma_v^2, sigma_w^2) and G = df/du, the derivative of the motion model
    by the input. f is linear in u, so column j of G is f(x, e_j) - f(x, 0): uses your motion_model.
    """
    base = motion_model(x, (0.0, 0.0, 0.0), dt)
    G = np.column_stack([motion_model(x, e, dt) - base for e in np.eye(3)])
    G[2] = np.arctan2(np.sin(G[2]), np.cos(G[2]))              # theta near +-pi: the difference, not the jump
    M = np.diag([sigma_v ** 2, sigma_v ** 2, sigma_w ** 2])
    return G @ M @ G.T


def yaw(q):
    """The heading of a quaternion (geometry_msgs/msg/Quaternion) that turns only about z."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
