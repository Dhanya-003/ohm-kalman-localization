#!/usr/bin/env python3
"""Task 3 (given): the output of robot_localization on the topics of your filter, so the monitor compares both.

    /<robot>/odometry/filtered (nav_msgs/msg/Odometry)  ->  /<robot>/kf/pose, /<robot>/kf/twist, /<robot>/kf/info

kf/twist holds the velocity in the world frame, as in your filter: the twist of the Odometry message is in the
frame of the robot and is rotated with the quaternion of the estimate (v' = q v q*, as tf2 does it).
launch/ekf.launch.py starts this node.
"""
import json
import os

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class EkfBridge(Node):
    """ROS node that puts the estimate of robot_localization on the topics of your filter (task 3, given).

    Parameters: robot (name), params (name of the parameter file, only for kf/info and the CSV name).

    Attributes:
        params: File name of the robot_localization parameters, e.g. ekf.yaml.
    """

    def __init__(self):
        """Declare the parameters, create the publishers and subscribe to /<robot>/odometry/filtered."""
        super().__init__("ekf_bridge")
        robot = self.declare_parameter("robot", "muster").value
        self.params = os.path.basename(self.declare_parameter("params", "ekf.yaml").value)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, f"/{robot}/kf/pose", 10)
        self.pub_twist = self.create_publisher(TwistWithCovarianceStamped, f"/{robot}/kf/twist", 10)
        self.pub_info = self.create_publisher(String, f"/{robot}/kf/info", 10)
        self.create_subscription(Odometry, f"/{robot}/odometry/filtered", self.on_odometry, 10)
        self.get_logger().info(f"/{robot}/odometry/filtered -> /{robot}/kf/pose, kf/twist ({self.params})")

    def on_odometry(self, msg: Odometry) -> None:
        """Callback of /<robot>/odometry/filtered: republish it as kf/pose, kf/twist and kf/info.

        The velocity is rotated from the frame of the robot into the world frame with the quaternion of the
        estimate; its covariance is passed on unrotated.

        Args:
            msg: The estimate of the ekf_node, nav_msgs/msg/Odometry.
        """
        pose = PoseWithCovarianceStamped()
        pose.header.stamp, pose.header.frame_id = msg.header.stamp, msg.header.frame_id
        pose.pose = msg.pose
        self.pub_pose.publish(pose)
        q = msg.pose.pose.orientation                         # two_d_mode: a turn about z only
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        twist = TwistWithCovarianceStamped()
        twist.header = pose.header
        twist.twist.twist.linear.x = (1.0 - 2.0 * q.z * q.z) * vx - 2.0 * q.w * q.z * vy
        twist.twist.twist.linear.y = 2.0 * q.w * q.z * vx + (1.0 - 2.0 * q.z * q.z) * vy
        twist.twist.covariance = msg.twist.covariance       # left in the frame of the robot
        self.pub_twist.publish(twist)
        self.pub_info.publish(String(data=json.dumps({"task": "ekf", "params": self.params})))


def main():
    """Start ROS, spin the node until Ctrl-C or shutdown, then clean up."""
    rclpy.init()
    node = EkfBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
