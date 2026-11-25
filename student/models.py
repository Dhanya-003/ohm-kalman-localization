"""Motion models and measurement models of the lab.

Tasks 1 and 2, the linear filter, state x = (x, y, vx, vy) in the world frame:
    cv_model(dt)                       A of the constant-velocity model             task 1
    process_noise(dt, sigma_a)         Q: what the model does not know (given)      task 1
    gps_measurement(msg)               z, H, R of one GPS fix                       task 1
    odom_measurement(msg, sigma_v)     z, H, R of one odometry message              task 2

Task 4, the EKF, state x = (x, y, theta), input u = (vx, vy, omega) of the odometry in the frame of the robot:
    motion_model(x, u, dt)             f(x, u): the state after dt seconds          task 4
    motion_jacobian(x, u, dt)          F = df/dx at x                               task 4
    input_noise(x, dt, ...)            Q from the noise of u (given)                task 4

Replace each `raise NotImplementedError("STUDENT ...")` by your code. The red box STUDENT ... in the handout
says what is given there, what to write and which test checks it; the `return` below shows the names to use.
The messages are ROS messages: `ros2 interface show <type>` lists their fields.
"""
import math

import numpy as np


def cv_model(dt):
    """Transition matrix A of the constant-velocity model (task 1, eq. 3.13 of the script).

    The position grows by velocity times dt, the velocity stays as it is.

    Args:
        dt: Length of the time step in s.

    Returns:
        A: 4 x 4 numpy array. A @ x maps the state x = (x, y, vx, vy) at time t to the state at t + dt.
    """
    # A of eq. (3.13) for a step of dt seconds, 4 x 4
    # TODO STUDENT 1.1
    raise NotImplementedError("STUDENT 1.1")
    A = np.array([
    [1.0, 0.0, dt,  0.0],
    [0.0, 1.0, 0.0, dt ],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],], dtype=float)
    return A


def process_noise(dt, sigma_a):
    """Process noise Q of the constant-velocity model over one step (given).

    The model keeps the velocity constant; the real robot brakes and turns. Q treats this as an unknown
    acceleration a with standard deviation sigma_a, constant over the step. It moves the position by
    a dt^2 / 2 and the velocity by a dt, so per axis Q = sigma_a^2 g g^T with g = (dt^2 / 2, dt).

    Args:
        dt: Length of the time step in s.
        sigma_a: Standard deviation of the unknown acceleration in m/s^2. 0 gives Q = 0.

    Returns:
        Q: 4 x 4 numpy array, covariance added to P in every prediction. x and y are independent.
    """
    g = np.array([dt * dt / 2.0, dt])
    q = sigma_a ** 2 * np.outer(g, g)
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q                  # x and vx
    Q[np.ix_([1, 3], [1, 3])] = q                  # y and vy
    return Q


def gps_measurement(msg):
    """Measurement model of one GPS fix (task 1, STUDENT 1.5): measurement z, observation matrix H, noise R.

    Args:
        msg: One fix, geometry_msgs/msg/PoseWithCovarianceStamped from /<robot>/gps_cov. The position is
            in msg.pose.pose.position, its covariance in msg.pose.covariance (6 x 6, row by row, in the
            order x y z roll pitch yaw).

    Returns:
        A tuple (z, H, R) of numpy arrays:
        z: the measured position (x, y) in m, 2 values.
        H: 2 x 4 observation matrix. H @ x is the position the filter expects for the state
            x = (x, y, vx, vy).
        R: 2 x 2 covariance of z in m^2, the variances of x and y of the fix.
    """
    # z, H, R of the GPS (handout, STUDENT 1.5): z the measured position, H takes x and y out of the state,
    # R the variances of x and y from msg.pose.covariance (6 x 6, row by row: x y z roll pitch yaw)
    # TODO STUDENT 1.5
    raise NotImplementedError("STUDENT 1.5")
    z = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float)
    H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],], dtype=float)
    R = np.diag([msg.pose.covariance[0], msg.pose.covariance[7]])
    return z, H, R


def odom_measurement(msg, sigma_v):
    """Measurement model of one odometry message (task 2, STUDENT 2.1): z, H and R of the velocity.

    The odometry gives the velocity in the frame of the robot (child_frame_id). The state holds the velocity
    in the world frame, so the measurement is that velocity rotated by the heading theta of the odometry.

    Args:
        msg: One message nav_msgs/msg/Odometry from /<robot>/odom. Heading in msg.pose.pose.orientation,
            velocity of the robot in msg.twist.twist.linear.
        sigma_v: Standard deviation of each velocity component in m/s, the node parameter sigma_v.

    Returns:
        A tuple (z, H, R) of numpy arrays:
        z: the velocity (vx, vy) in the world frame in m/s, 2 values.
        H: 2 x 4 observation matrix. H @ x is the velocity the filter expects for the state
            x = (x, y, vx, vy).
        R: 2 x 2 covariance of z in m^2/s^2, sigma_v^2 on the diagonal.
    """
    theta = yaw(msg.pose.pose.orientation)
    vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
    # z, H, R of the odometry (handout, STUDENT 2.1): z the velocity rotated into the world frame by theta,
    # H takes vx and vy out of the state, R = sigma_v^2 I
    # TODO STUDENT 2.1
    raise NotImplementedError("STUDENT 2.1")
    c, s = math.cos(theta), math.sin(theta)
    z = np.array([c * vx - s * vy, s * vx + c * vy], dtype=float)
    H = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],], dtype=float)
    R = np.diag([sigma_v ** 2, sigma_v ** 2])
    return z, H, R


def motion_model(x, u, dt):
    """Motion model f(x, u) of the EKF (task 4, STUDENT 4.1): the state after driving dt seconds.

    The velocity of the robot, rotated into the world frame by theta, moves the position; omega turns theta.

    Args:
        x: State (x, y, theta) at time t: position in m, heading in rad. Any sequence of 3 numbers.
        u: Input (vx, vy, omega) from the odometry: velocity in the frame of the robot in m/s, turn rate
            in rad/s.
        dt: Length of the time step in s.

    Returns:
        x_new: numpy array (x, y, theta) at time t + dt, theta wrapped to -pi .. pi.
    """
    px, py, theta = x
    vx, vy, omega = u
    # x_new = f(x, u) of the handout (STUDENT 4.1), a numpy array of three numbers
    # TODO STUDENT 4.1
    raise NotImplementedError("STUDENT 4.1")
    c, s = math.cos(theta), math.sin(theta)
    x_new = np.array([
    px + (vx * c - vy * s) * dt,
    py + (vx * s + vy * c) * dt,
    theta + omega * dt,], dtype=float)
    #x_new[2] = math.atan2(math.sin(x_new[2]), math.cos(x_new[2]))      # theta stays in -pi .. pi
    return x_new


def motion_jacobian(x, u, dt):
    """Jacobian F = df/dx of the motion model at x (task 4, STUDENT 4.2).

    F[i, j] says how much entry i of f(x, u) changes per unit change of entry j of x. It carries the
    covariance through the prediction: P <- F P F^T + Q.

    Args:
        x: State (x, y, theta) at which F is evaluated, the estimate before the prediction.
        u: Input (vx, vy, omega) from the odometry, as in motion_model.
        dt: Length of the time step in s.

    Returns:
        F: 3 x 3 numpy array.
    """
    theta = x[2]
    vx, vy, _ = u
    # F = df/dx of the handout (STUDENT 4.2), 3 x 3: the identity plus the derivatives by theta
    # TODO STUDENT 4.2
    raise NotImplementedError("STUDENT 4.2")
    c, s = math.cos(theta), math.sin(theta)
    F = np.array([
    [1.0, 0.0, (-vx * s - vy * c) * dt],
    [0.0, 1.0, ( vx * c - vy * s) * dt],
    [0.0, 0.0, 1.0],], dtype=float)
    return F


def input_noise(x, dt, sigma_v, sigma_w):
    """Process noise Q of one EKF step, from the noise of the input u (given).

    Q = G M G^T with M = diag(sigma_v^2, sigma_v^2, sigma_w^2), the covariance of u, and G = df/du, the
    derivative of the motion model by the input. f is linear in u, so column j of G is f(x, e_j) - f(x, 0):
    the function uses your motion_model.

    Args:
        x: State (x, y, theta) before the prediction.
        dt: Length of the time step in s.
        sigma_v: Standard deviation of vx and vy of the odometry in m/s.
        sigma_w: Standard deviation of the turn rate omega in rad/s.

    Returns:
        Q: 3 x 3 numpy array, covariance added to P in the prediction.
    """
    base = motion_model(x, (0.0, 0.0, 0.0), dt)
    G = np.column_stack([motion_model(x, e, dt) - base for e in np.eye(3)])
    G[2] = np.arctan2(np.sin(G[2]), np.cos(G[2]))              # wrap: a step across +-pi stays small
    M = np.diag([sigma_v ** 2, sigma_v ** 2, sigma_w ** 2])
    return G @ M @ G.T


def yaw(q):
    """Heading of a quaternion that turns only about the z axis.

    Args:
        q: geometry_msgs/msg/Quaternion with the fields x, y, z, w.

    Returns:
        The angle about z in rad, -pi .. pi.
    """
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
