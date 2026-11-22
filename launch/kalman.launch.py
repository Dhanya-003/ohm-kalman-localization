"""Lab "Kalman filter": the simulator in the hall `open`, the robot drives a loop, RViz shows GPS and estimate.

    ros2 launch ohm_kalman_localization kalman.launch.py
    ros2 run ohm_kalman_localization kalman_node.py --ros-args -p task:=1       # second terminal
    ros2 launch ohm_kalman_localization kalman.launch.py gps:=tunnel              # task 2: no fix in the tunnel
    ros2 launch launch/kalman.launch.py                                           # from the checkout

`--show-args` lists the arguments; ohm_kalman_localization/launching.py does the work.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # checkout, even via a symlink
if HERE not in sys.path:
    sys.path.append(HERE)
from ohm_kalman_localization.launching import describe          # noqa: E402


def generate_launch_description():
    """Entry point of ros2 launch: simulator, driver, monitor and RViz.

    Returns:
        launch.LaunchDescription, built by ohm_kalman_localization.launching.describe.
    """
    return describe()
