# ohm_kalman_localization

Lab *Kalman filter* for the simulator [ir-mecanum-lab](https://github.com/christianpfitzner/ir-mecanum-lab):
a linear Kalman filter that finds a driving robot, first from GPS alone, then from GPS and odometry.
Handout: `docs/kalman.pdf` (`make` in `docs/`).

## Install

```bash
sudo apt install ros-jazzy-robot-localization
cd ~/ros2_ws/src
git clone git@github.com:christianpfitzner/ir-mecanum-lab.git
git clone git@github.com:christianpfitzner/ohm-kalman-localization.git
cd ~/ros2_ws
colcon build --symlink-install --packages-select mecanum_lab_interfaces mecanum_lab ohm_kalman_localization
source install/setup.bash
```

## Run

| command | what |
|---|---|
| `ros2 launch ohm_kalman_localization kalman.launch.py` | simulator in the hall `open`, the robot drives a loop, monitor, RViz |
| `ros2 run ohm_kalman_localization kalman_node.py --ros-args -p task:=1` | your filter: task 1 GPS, task 2 GPS and odometry, task 3 EKF |
| `ros2 launch ohm_kalman_localization ekf.launch.py` | task 4: robot_localization's ekf_node with `config/ekf.yaml` instead of your node |
| `ros2 launch ohm_kalman_localization kalman.launch.py gps:=tunnel` | no fix in the tunnel; `gps:=multipath` adds a biased zone |
| `ros2 run ohm_kalman_localization plot_run.py` | plots of the newest run in `runs/` |
| `python3 hint.py 1.3` | the next of three hints for `STUDENT 1.3`; without a block: the hints opened so far |

Your code: `student/kalman_filter.py`, `student/models.py`. Launch arguments: `gps:=`, `speed:=`, `robot:=`,
`rviz:=`, `headless:=`, `seconds:=`.

## Parameters of the node

| parameter | default | |
|---|---|---|
| `task` | 1 | 1 GPS only · 2 GPS and odometry · 3 EKF |
| `sigma_a` | 0.3 | m/s², process noise Q; 0 = Q = 0 |
| `sigma_v` | 0.05 | m/s, noise of the odometry velocity (tasks 2, 3) |
| `sigma_w` | 0.05 | rad/s, noise of the odometry turn rate (task 3) |
| `theta0`, `sigma_theta0` | 0, 0.5 | rad, initial heading of the EKF and its σ |
| `gate` | 0 | extra task: reject fixes with d² above this, e.g. 9.21 |
| `sigma_v0`, `rate` | 1.0, 10 | initial σ of the velocity, Hz of the prediction |

## Tests

```bash
python3 -m pytest test -q -k task1
```
