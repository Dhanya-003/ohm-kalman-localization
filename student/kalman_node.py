#!/usr/bin/env python3
"""Lab "Kalman filter": the node that runs your filter on GPS and odometry.

    ros2 launch ohm_kalman_localization kalman.launch.py                          # terminal 1
    ros2 run ohm_kalman_localization kalman_node.py --ros-args -p task:=1        # terminal 2

The node is given; the filter it runs is yours, in `kalman_filter.py` and `models.py` next to this file.

    task 1   GPS only:            x = (x, y, vx, vy), constant velocity; predict every 1/rate s, correct with every
                                  fix on /<robot>/gps_cov
    task 2   GPS and odometry:    task 1, plus a correction with the velocity of every /<robot>/odom
    task 3   EKF:                 x = (x, y, theta); predict with the velocity of every /<robot>/odom as input,
                                  correct with every fix

The filter starts at the first fix. Results: /<robot>/kf/pose (position and its covariance),
/<robot>/kf/twist (velocity in the world frame and its covariance), /<robot>/kf/info (the settings as
JSON), and every 5 s one line in the terminal.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))              # student/, also via the installed symlink
for path in (HERE, os.path.dirname(HERE)):                     # your files, then the package next door
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np                                            # noqa: E402
import rclpy                                                  # noqa: E402
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped   # noqa: E402
from nav_msgs.msg import Odometry                             # noqa: E402
from rclpy.executors import ExternalShutdownException         # noqa: E402
from rclpy.node import Node                                   # noqa: E402
from rclpy.parameter import Parameter                         # noqa: E402
from std_msgs.msg import String                               # noqa: E402

import kalman_filter                                          # noqa: E402
import models                                                 # noqa: E402

PARAMETERS = {
    "robot": "muster",          # topics /<robot>/gps_cov, /<robot>/odom and /<robot>/kf/...
    "task": 1,                  # 1 GPS only, 2 GPS and odometry, 3 EKF
    "sigma_a": 0.3,             # m/s^2: unknown acceleration in Q; 0 means Q = 0
    "sigma_v": 0.05,            # m/s: noise of the odometry velocity, R of task 2, in Q of task 3
    "sigma_w": 0.05,            # rad/s: noise of the odometry turn rate, in Q of task 3
    "theta0": 0.0,              # rad: initial heading of task 3
    "sigma_theta0": 0.5,        # rad: its initial standard deviation
    "sigma_v0": 1.0,            # m/s: initial standard deviation of the velocity
    "rate": 10.0,               # Hz: prediction steps and reports between two fixes
}
RESTART = 5.0                   # s: a longer step (or a step back) means the simulator was restarted


class KalmanNode(Node):

    def __init__(self):
        super().__init__("kalman", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.p = {name: self.declare_parameter(name, value).value for name, value in PARAMETERS.items()}
        robot, self.task = self.p["robot"], int(self.p["task"])
        self.kf, self.t, self.u = None, 0.0, (0.0, 0.0, 0.0)
        self.fixes, self.k_gps, self.last_log = 0, 0.0, -1e9
        self.create_subscription(PoseWithCovarianceStamped, f"/{robot}/gps_cov", self.on_gps, 10)
        if self.task >= 2:
            self.create_subscription(Odometry, f"/{robot}/odom", self.on_odom, 50)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, f"/{robot}/kf/pose", 10)
        self.pub_twist = self.create_publisher(TwistWithCovarianceStamped, f"/{robot}/kf/twist", 10)
        self.pub_info = self.create_publisher(String, f"/{robot}/kf/info", 10)
        self.create_timer(1.0 / float(self.p["rate"]), self.on_timer)
        sensors = {1: "GPS", 2: "GPS + odometry", 3: "EKF, GPS + odometry"}[self.task]
        noise = f"sigma_a {self.p['sigma_a']} m/s^2" if self.task < 3 else f"sigma_w {self.p['sigma_w']} rad/s"
        self.get_logger().info(f"task {self.task} ({sensors}), {noise}"
                               + (f", sigma_v {self.p['sigma_v']} m/s" if self.task >= 2 else "")
                               + f"; waiting for the first fix on /{robot}/gps_cov")

    # ------------------------------------------------------------------ time

    def advance(self, t: float) -> bool:
        """Predict from the filter's time to t. False if the filter had to be dropped (time jumped)."""
        dt = t - self.t
        if dt > RESTART or dt < -RESTART:
            self.get_logger().warn(f"time jumped by {dt:.1f} s: filter dropped, waiting for the next fix")
            self.kf = None
            return False
        if dt > 0.0 and self.task == 3:          # EKF: the model f with the newest odometry as input u
            x = self.kf.x
            F = models.motion_jacobian(x, self.u, dt)
            Q = models.input_noise(x, dt, float(self.p["sigma_v"]), float(self.p["sigma_w"]))
            self.kf.predict_nonlinear(models.motion_model(x, self.u, dt), F, Q)
            self.t = t
        elif dt > 0.0:
            A = models.cv_model(dt)
            Q = models.process_noise(dt, float(self.p["sigma_a"]))
            self.kf.predict(A, Q)
            self.t = t
        return True

    # ------------------------------------------------------------------ callbacks

    def on_gps(self, msg: PoseWithCovarianceStamped) -> None:
        t = stamp(msg)
        z, H, R = models.gps_measurement(msg)
        if self.task == 3:
            H = H[:, :3]                       # the EKF state is (x, y, theta): the columns of x and y stay
        if self.kf is None:                    # start at the fix: velocity unknown (tasks 1, 2), heading theta0 (3)
            if self.task == 3:
                x0, var = [z[0], z[1], float(self.p["theta0"])], [float(self.p["sigma_theta0"]) ** 2]
            else:
                x0, var = [z[0], z[1], 0.0, 0.0], [float(self.p["sigma_v0"]) ** 2] * 2
            self.kf, self.t = kalman_filter.KalmanFilter(x0, np.diag([R[0, 0], R[1, 1]] + var)), t
            self.get_logger().info(f"start at the fix ({z[0]:.2f}, {z[1]:.2f}) m, t = {t:.1f} s, "
                                   f"sigma_gps {math.sqrt(R[0, 0]):.2f} m")
            self.publish()
            return
        if not self.advance(t):
            return
        self.fixes += 1
        self.kf.correct(z, H, R)
        self.k_gps = float(self.kf.K[0, 0])
        if self.task == 3:
            self.kf.x[2] = math.atan2(math.sin(self.kf.x[2]), math.cos(self.kf.x[2]))

    def on_odom(self, msg: Odometry) -> None:
        if self.task == 3:                     # EKF: the odometry is the input u, from now until the next message
            if self.kf is not None and self.advance(stamp(msg)):
                tw = msg.twist.twist
                self.u = (tw.linear.x, tw.linear.y, tw.angular.z)
            return
        if self.kf is None or not self.advance(stamp(msg)):
            return
        z, H, R = models.odom_measurement(msg, float(self.p["sigma_v"]))
        self.kf.correct(z, H, R)

    def on_timer(self) -> None:
        if self.kf is None:
            return
        if self.task == 3 or self.advance(self.get_clock().now().nanoseconds * 1e-9):
            self.publish()                     # the EKF predicts with every odometry message, not here

    # ------------------------------------------------------------------ results

    def publish(self) -> None:
        x, P, ekf = self.kf.x, self.kf.P, self.task == 3
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp.sec, pose.header.stamp.nanosec = int(self.t), int((self.t % 1.0) * 1e9)
        pose.pose.pose.position.x, pose.pose.pose.position.y = float(x[0]), float(x[1])
        if ekf:
            heading = float(x[2])
            velocity = (models.motion_model(x, self.u, 1.0) - x)[:2]      # what f adds in 1 s: world velocity
            velocity_cov = np.eye(2) * float(self.p["sigma_v"]) ** 2
        else:
            heading = math.atan2(x[3], x[2])                       # the state has no heading: the course
            velocity, velocity_cov = x[2:4], P[2:4, 2:4]
        pose.pose.pose.orientation.z, pose.pose.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
        pose.pose.covariance = cov6(P[0:2, 0:2], P[2, 2] if ekf else 1e-6)
        self.pub_pose.publish(pose)
        twist = TwistWithCovarianceStamped()
        twist.header = pose.header
        twist.twist.twist.linear.x, twist.twist.twist.linear.y = float(velocity[0]), float(velocity[1])
        twist.twist.covariance = cov6(velocity_cov)
        self.pub_twist.publish(twist)
        info = {"task": self.task, "sigma_a": self.p["sigma_a"], "sigma_v": self.p["sigma_v"],
                "fixes": self.fixes, "k_gps": self.k_gps}
        if ekf:
            info.update(sigma_w=self.p["sigma_w"], theta0=self.p["theta0"], sigma_theta0=self.p["sigma_theta0"])
        self.pub_info.publish(String(data=json.dumps(info)))
        if self.t - self.last_log >= 5.0:
            self.last_log = self.t
            sigma = math.sqrt(max(P[0, 0], P[1, 1]))
            state = (f"theta {math.degrees(x[2]):7.1f} deg (sigma {math.degrees(math.sqrt(P[2, 2])):5.1f})" if ekf
                     else f"v ({x[2]:5.2f}, {x[3]:5.2f}) m/s")
            self.get_logger().info(
                f"t {self.t:6.1f} s | x ({x[0]:6.2f}, {x[1]:6.2f}) m | {state} | "
                f"sigma {sigma:.2f} m | K {self.k_gps:.2f} | fixes {self.fixes}")


def stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def cov6(block, yaw=1e-6) -> list:
    """A 2 x 2 covariance into the 6 x 6 of ROS (x y z roll pitch yaw); z, roll, pitch tiny, yaw as given."""
    c = [0.0] * 36
    c[0], c[1], c[6], c[7] = float(block[0, 0]), float(block[0, 1]), float(block[1, 0]), float(block[1, 1])
    c[14] = c[21] = c[28] = 1e-6
    c[35] = float(yaw)
    return c


def main():
    rclpy.init()
    node = KalmanNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
