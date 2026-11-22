#!/usr/bin/env python3
"""The driver of the lab (given): the robot drives the loop of route.py, lap after lap.

The driver steers by the exact pose on /<robot>/truth, like a person who sees where the robot is: it drives
along the tangent of the loop, corrects sideways back onto the loop, and turns with the curvature of the loop
(feed-forward) plus a small correction of the heading. Your
filter never reads /truth: it has to find the robot from GPS and odometry alone. Every lap the robot
brakes on the left straight, stands for `stop` seconds and starts again, with `accel` m/s^2 both ways.

    ros2 run ohm_kalman_localization route_node.py --ros-args -p speed:=0.5      (the launch file starts it)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import rclpy                                                  # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist              # noqa: E402
from rclpy.executors import ExternalShutdownException         # noqa: E402
from rclpy.node import Node                                   # noqa: E402

from ohm_kalman_localization import route                    # noqa: E402

STEP = 0.05                     # m between two points of the loop
LEVER, RADIUS = 0.27, 0.05      # m: lx + ly and wheel radius of the simulated mecanum robot


class RouteNode(Node):
    """ROS node that drives the robot along the loop of route.py, steered by the true pose (given).

    Parameters: speed (m/s), stop (s per lap), accel (m/s^2), k_cross and k_heading (1/s, the two
    corrections), preview (m), wheel_limit (rad/s), robot (name).

    Attributes:
        path: The loop as a list of (x, y, tangent), STEP m apart.
        stop_index: Index of the point of path closest to route.STOP.
        index: Index of the point of path the robot passed last.
        wait_until: Simulation time in s until which the robot stands at the stop.
        go: Simulation time in s at which the robot started moving the last time.
        laps: Number of completed laps.
    """

    def __init__(self):
        """Declare the parameters, build the loop and subscribe to /<robot>/truth."""
        super().__init__("route")
        self.speed = self.declare_parameter("speed", 0.5).value            # m/s along the loop
        self.stop = self.declare_parameter("stop", 4.0).value              # s standing still per lap
        self.accel = self.declare_parameter("accel", 0.3).value            # m/s^2 braking and starting
        self.k_cross = self.declare_parameter("k_cross", 0.5).value        # 1/s: back to the loop, sideways
        self.k_heading = self.declare_parameter("k_heading", 0.8).value    # 1/s: heading onto the tangent
        self.preview = self.declare_parameter("preview", 0.15).value       # m: tangent and curvature this far ahead
        self.wheel_limit = self.declare_parameter("wheel_limit", 11.0).value   # rad/s, the simulator clips at 12
        robot = self.declare_parameter("robot", "muster").value
        self.path = route.loop(STEP)
        self.stop_index = min(range(len(self.path)), key=lambda i: math.dist(self.path[i][:2], route.STOP))
        self.index, self.wait_until, self.go, self.laps = 0, -1.0, None, 0
        self.pub = self.create_publisher(Twist, f"/{robot}/cmd_vel", 10)
        self.create_subscription(PoseStamped, f"/{robot}/truth", self.on_truth, 10)
        self.get_logger().info(f"driving the loop ({route.length():.1f} m, {self.speed} m/s, stop {self.stop} s "
                               f"per lap) by /{robot}/truth")

    def on_truth(self, msg: PoseStamped) -> None:
        """Callback of /<robot>/truth: compute and publish one velocity command on /<robot>/cmd_vel.

        Finds the nearest point of the loop, handles the stop and the speed ramps, and steers along the tangent
        with a sideways and a heading correction. The command is scaled down as a whole if a wheel would exceed
        wheel_limit.

        Args:
            msg: The true pose, geometry_msgs/msg/PoseStamped.
        """
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x, y = msg.pose.position.x, msg.pose.position.y
        q = msg.pose.orientation
        theta = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        n = len(self.path)
        window = range(self.index, self.index + int(2.0 / STEP))           # search 2 m ahead only
        nearest = min(window, key=lambda i: math.dist(self.path[i % n][:2], (x, y))) % n
        advanced = (nearest - self.index) % n
        if 0 < (self.stop_index - self.index) % n <= advanced:
            self.wait_until = t + self.stop                                # passed the stop point
        if nearest < self.index:
            self.laps += 1
        self.index = nearest
        cmd = Twist()
        if t < self.wait_until:
            self.pub.publish(cmd)
            return
        if self.go is None or self.go < self.wait_until:
            self.go = max(t, self.wait_until)                              # moving off from here
        to_stop = ((self.stop_index - nearest - 1) % n + 1) * STEP           # m, the stop just passed: a lap
        speed = min(self.speed, max(0.05, math.sqrt(2.0 * self.accel * to_stop)),   # braking
                    0.05 + self.accel * (t - self.go))                                 # starting
        px, py, tangent = self.path[nearest]
        cross = -(x - px) * math.sin(tangent) + (y - py) * math.cos(tangent)     # m, > 0: left of the loop
        side = max(-0.15, min(0.15, -self.k_cross * cross))                      # m/s back towards the loop
        vx = speed * math.cos(tangent) - side * math.sin(tangent)                # world frame
        vy = speed * math.sin(tangent) + side * math.cos(tangent)
        k = int(self.preview / STEP)
        ahead, beyond = self.path[(nearest + k) % n][2], self.path[(nearest + k + 1) % n][2]
        curvature = math.atan2(math.sin(beyond - ahead), math.cos(beyond - ahead)) / STEP   # 1/m, 1 in a corner
        error = math.atan2(math.sin(ahead - theta), math.cos(ahead - theta))
        cmd.linear.x = math.cos(theta) * vx + math.sin(theta) * vy        # into the frame of the robot
        cmd.linear.y = -math.sin(theta) * vx + math.cos(theta) * vy
        cmd.angular.z = max(-1.2, min(1.2, speed * curvature + self.k_heading * error))   # turn with the loop
        # The simulator clips every wheel on its own, and that eats the turn first. Scale the whole command
        # instead, so that the fastest wheel stays below wheel_limit and the direction stays as it is.
        fastest = (abs(cmd.linear.x) + abs(cmd.linear.y) + LEVER * abs(cmd.angular.z)) / RADIUS
        if fastest > self.wheel_limit:
            scale = self.wheel_limit / fastest
            cmd.linear.x, cmd.linear.y = cmd.linear.x * scale, cmd.linear.y * scale
            cmd.angular.z *= scale
        self.pub.publish(cmd)


def main():
    """Start ROS, spin the node until Ctrl-C or shutdown, then clean up."""
    rclpy.init()
    node = RouteNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
