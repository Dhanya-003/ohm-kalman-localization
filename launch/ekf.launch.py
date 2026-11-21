"""Lab "Kalman filter", task 4: the EKF of robot_localization instead of your node, on the same topics.

    ros2 launch ohm_kalman_localization kalman.launch.py                 # terminal 1, as always
    ros2 launch ohm_kalman_localization ekf.launch.py                    # terminal 2, instead of kalman_node.py
    ros2 launch ohm_kalman_localization ekf.launch.py params:=my.yaml    # another parameter file

`--show-args` lists the arguments; ohm_kalman_localization/launching.py does the work.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # checkout, even via a symlink
if HERE not in sys.path:
    sys.path.append(HERE)
from ohm_kalman_localization.launching import describe_ekf      # noqa: E402


def generate_launch_description():
    return describe_ekf()
