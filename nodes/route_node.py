#!/usr/bin/env python3
"""The driver of the lab (given): the robot drives the loop of route.py, lap after lap.

The driver steers by the exact pose on /<robot>/truth, like a person who sees where the robot is. Your
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


class RouteNode(Node):

    def __init__(self):
        super().__init__("route")
        self.speed = self.declare_parameter("speed", 0.5).value            # m/s along the loop
        self.stop = self.declare_parameter("stop", 4.0).value              # s standing still per lap
        self.accel = self.declare_parameter("accel", 0.3).value            # m/s^2 braking and starting
        self.lookahead = self.declare_parameter("lookahead", 0.6).value    # m ahead on the loop
        robot = self.declare_parameter("robot", "muster").value
        self.path = route.loop(STEP)
        self.stop_index = min(range(len(self.path)), key=lambda i: math.dist(self.path[i][:2], route.STOP))
        self.index, self.wait_until, self.go, self.laps = 0, -1.0, None, 0
        self.pub = self.create_publisher(Twist, f"/{robot}/cmd_vel", 10)
        self.create_subscription(PoseStamped, f"/{robot}/truth", self.on_truth, 10)
        self.get_logger().info(f"driving the loop ({route.length():.1f} m, {self.speed} m/s, stop {self.stop} s "
                               f"per lap) by /{robot}/truth")

    def on_truth(self, msg: PoseStamped) -> None:
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
        ahead = self.path[(nearest + int(self.lookahead / STEP)) % n]
        dx, dy = ahead[0] - x, ahead[1] - y
        d = math.hypot(dx, dy) or 1.0
        vx, vy = speed * dx / d, speed * dy / d                            # towards the point ahead, world
        heading = self.path[(nearest + int(0.5 * self.lookahead / STEP)) % n][2]
        error = math.atan2(math.sin(heading - theta), math.cos(heading - theta))
        cmd.linear.x = math.cos(theta) * vx + math.sin(theta) * vy        # into the frame of the robot
        cmd.linear.y = -math.sin(theta) * vx + math.cos(theta) * vy
        cmd.angular.z = max(-1.2, min(1.2, 2.5 * error))
        self.pub.publish(cmd)


def main():
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
