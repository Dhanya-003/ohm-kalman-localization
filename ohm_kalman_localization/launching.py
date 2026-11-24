"""What `launch/kalman.launch.py` starts: the simulator in the hall `open`, the driver, the monitor, RViz.

The simulator's own `lab.launch.py` (repository `ir-mecanum-lab`) is included with one of config/*.json and
with its own RViz switched off: this lab brings an RViz config of its own (`rviz/kalman.rviz`). The driver
(nodes/route_node.py) drives the loop, the monitor (nodes/monitor_node.py) prints how good GPS and filter
are. The students' node is not started here; they run it themselves in a second terminal.
"""
import os
import shutil
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo, \
    OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from . import paths

ARGS = [
    ("robot", "muster", "robot name"),
    ("gps", "open", "GPS: open (everywhere) | tunnel (no fix in the upper left corner)"),
    ("speed", "0.5", "speed of the driver along the loop, m/s"),
    ("rviz", "auto", "RViz 2 with the lab's view: auto | true | false"),
    ("headless", "false", "without the window (SDL dummy driver)"),
    ("seconds", "", "end after N s of simulation time (empty = until Ctrl-C)"),
]
CONFIGS = ("open", "tunnel")
NODE = "ros2 run ohm_kalman_localization kalman_node.py --ros-args -p task:=1"


def rviz_config(robot: str) -> str:
    """Write the RViz config of the lab for one robot.

    Args:
        robot: Robot name, replaces the placeholder {robot} in rviz/kalman.rviz.

    Returns:
        Path of the written file in /tmp.
    """
    with open(paths.lab_file("rviz", "kalman.rviz"), encoding="utf-8") as handle:
        text = handle.read()
    target = f"/tmp/ohm_kalman_localization_{robot}.rviz"
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text.replace("{robot}", robot))
    return target


def given_node(name: str, robot: str, *parameters: str) -> ExecuteProcess:
    """Launch action for one of the given nodes in nodes/.

    Runs it from the checkout, or from lib/ of the install after a colcon build without --symlink-install.

    Args:
        name: File name of the node, e.g. "route_node.py".
        robot: Robot name, passed as the parameter robot.
        *parameters: Further parameters as "name:=value".

    Returns:
        launch.actions.ExecuteProcess that starts the node with the Python of this launch.
    """
    path = paths.lab_file("nodes", name)
    if not os.path.isfile(path):
        from ament_index_python.packages import get_package_prefix
        path = os.path.join(get_package_prefix(paths.PACKAGE), "lib", paths.PACKAGE, name)
    command = [sys.executable, path, "--ros-args", "-p", f"robot:={robot}"]
    for parameter in parameters:
        command += ["-p", parameter]
    return ExecuteProcess(cmd=command, name=name[:-3], output="screen", emulate_tty=True)


def _after_exit(event, _context):
    """Event handler: when the simulator ends (window closed, seconds:= reached), end the whole launch.

    Args:
        event: launch.events.process.ProcessExited of any process of the launch.
        _context: The launch context (unused).

    Returns:
        [Shutdown] if the process was the simulator, else None.
    """
    if event.action.name.startswith("mecanum_sim"):
        return [Shutdown(reason="the simulator ended")]
    return None


def _start(context):
    """Build the actions of kalman.launch.py once the launch arguments are known.

    Args:
        context: The launch context, to read the arguments.

    Returns:
        List of launch actions: simulator, driver, monitor, exit handler and (if wanted and installed) RViz.

    Raises:
        ValueError: if gps:= names no config.
    """
    arg = lambda name: LaunchConfiguration(name).perform(context)   # noqa: E731
    simulator = paths.simulator_root()
    if arg("gps") not in CONFIGS:
        raise ValueError(f"gps:={arg('gps')} : choose one of: {', '.join(CONFIGS)}")
    config, robot = paths.lab_file("config", arg("gps") + ".json"), arg("robot")
    forwarded = {"config": config, "robot": robot, "rviz": "false", "truth": "true", "headless": arg("headless")}
    if arg("seconds"):
        forwarded["seconds"] = arg("seconds")
    actions = [LogInfo(msg=f"[kalman] robot '{robot}' drives the loop in the hall 'open', GPS '{arg('gps')}' "
                           f"(simulator {simulator}); start your node in a second terminal: {NODE}"),
               IncludeLaunchDescription(
                   PythonLaunchDescriptionSource(os.path.join(simulator, "launch", "lab.launch.py")),
                   launch_arguments=forwarded.items()),
               given_node("route_node.py", robot, f"speed:={arg('speed')}"),
               given_node("monitor_node.py", robot, f"config:={config}"),
               RegisterEventHandler(OnProcessExit(on_exit=_after_exit))]
    wanted = arg("rviz").lower()
    if wanted == "true" or (wanted == "auto" and shutil.which("rviz2")):
        actions.append(ExecuteProcess(cmd=["rviz2", "--display-config", rviz_config(robot),
                                           "--ros-args", "-p", "use_sim_time:=true"],
                                      name="rviz2", output="screen"))
    elif wanted == "auto":
        actions.append(LogInfo(msg="[kalman] rviz2 is not installed: sudo apt install ros-$ROS_DISTRO-rviz2"))
    return actions


EKF_ARGS = [
    ("robot", "muster", "robot name"),
    ("params", "", "parameter file of the ekf_node (empty = config/ekf.yaml of this lab)"),
]


def _start_ekf(context):
    """Build the actions of ekf.launch.py: robot_localization's ekf_node in the namespace of the robot and the bridge.

    Args:
        context: The launch context, to read the arguments.

    Returns:
        List of launch actions.
    """
    from launch_ros.actions import Node
    arg = lambda name: LaunchConfiguration(name).perform(context)   # noqa: E731
    robot, params = arg("robot"), arg("params") or paths.lab_file("config", "ekf.yaml")
    frames = {"use_sim_time": True, "map_frame": "map", "odom_frame": f"{robot}/odom",
              "base_link_frame": f"{robot}/base_link"}
    return [LogInfo(msg=f"[kalman] robot_localization ekf_node with {params}"),
            Node(package="robot_localization", executable="ekf_node", name="ekf", namespace=robot,
                 parameters=[params, frames], output="screen"),
            given_node("ekf_bridge.py", robot, f"params:={params}")]


def describe_ekf() -> LaunchDescription:
    """Launch description of ekf.launch.py (task 3), arguments in EKF_ARGS.

    Returns:
        launch.LaunchDescription.
    """
    return LaunchDescription([DeclareLaunchArgument(name, default_value=default, description=text)
                              for name, default, text in EKF_ARGS]
                             + [OpaqueFunction(function=_start_ekf)])


def describe() -> LaunchDescription:
    """Launch description of kalman.launch.py, arguments in ARGS.

    Returns:
        launch.LaunchDescription.
    """
    return LaunchDescription([DeclareLaunchArgument(name, default_value=default, description=text)
                              for name, default, text in ARGS]
                             + [OpaqueFunction(function=_start)])
