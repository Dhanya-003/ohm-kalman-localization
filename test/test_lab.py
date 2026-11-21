"""Tests of the lab repository: the configs fit the loop and the simulator next door, the given tools work.

    python3 -m pytest test -q          (the launch and node tests need ROS 2 sourced, the rest does not)
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ohm_kalman_localization import paths, route                 # noqa: E402

CONFIGS = ("open", "tunnel", "multipath")


def config(name):
    with open(paths.lab_file("config", name + ".json"), encoding="utf-8") as handle:
        return json.load(handle)


def load(rel):
    spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3], os.path.join(ROOT, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def on_loop(rect):
    x0, y0, x1, y1 = rect
    return [p for p in route.loop() if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]


def test_the_loop_is_closed_starts_at_the_spawn_and_keeps_off_the_walls():
    points = route.loop()
    assert points[0][:2] == pytest.approx(route.START[:2], abs=0.05)
    steps = [math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:] + points[:1])]
    assert max(steps) < 0.06
    assert sum(steps) == pytest.approx(route.length(), rel=0.01)
    assert all(2.0 < x < 28.0 and 2.0 < y < 18.0 for x, y, _ in points)          # hall open: 30 x 20 m


@pytest.mark.parametrize("name", CONFIGS)
def test_every_config_drives_the_same_loop_with_the_same_gps(name):
    cfg = config(name)
    assert cfg["world"] == "open" and cfg["debug_truth"] is True
    assert cfg["spawns"] == [list(route.START)]
    assert cfg["gps"]["rate"] == 5.0 and cfg["gps"]["sigma_xy"] == 0.5


def test_the_zones_lie_on_the_loop():
    tunnel, multipath = config("multipath")["gps"]["zones"]
    assert config("tunnel")["gps"]["zones"] == [tunnel] and config("open")["gps"]["zones"] == []
    inside = on_loop(tunnel["rect"])
    assert tunnel["block"] and len(inside) * 0.05 == pytest.approx(6.6, abs=0.1)             # corner, 6.6 m
    x0, y0, x1, y1 = tunnel["rect"]
    assert x0 < route.STOP[0] < x1 and y0 < route.STOP[1] < y1                                # the stop is inside
    inside = on_loop(multipath["rect"])
    assert all(p[0] == pytest.approx(21.0) for p in inside) and len(inside) * 0.05 > 2.5   # right straight
    assert on_loop([route.STOP[0] - 0.5, route.STOP[1] - 0.5, route.STOP[0] + 0.5, route.STOP[1] + 0.5])


@pytest.mark.parametrize("name", CONFIGS)
def test_the_simulator_reads_the_config(name):
    paths.use_simulator()
    from mecanum_lab import sensors, worlds
    from mecanum_lab.types import load_config
    cfg = load_config(paths.lab_file("config", name + ".json"))
    world = worlds.apply_config(worlds.load_world("open", cfg=cfg), cfg)
    pose = world.spawn_pose(0)
    assert (pose.x, pose.y, pose.theta) == pytest.approx(route.START)
    assert len(sensors._zones(cfg["gps"]["zones"])) == len(config(name)["gps"]["zones"])


def test_the_monitor_finds_the_zones_and_the_plot_finds_the_outages(tmp_path):
    pytest.importorskip("rclpy")
    monitor = load("nodes/monitor_node.py")
    zones = monitor.load_zones(paths.lab_file("config", "multipath.json"))
    assert [z["name"] for z in zones] == ["tunnel", "multipath"] and zones[0]["block"]
    assert monitor.inside(zones[0], *route.STOP) and not monitor.inside(zones[0], *route.START[:2])
    plot = load("nodes/plot_run.py")
    run = tmp_path / "task1.csv"
    rows = ["# " + json.dumps({"task": 1, "sigma_a": 0.1}), ",".join(monitor.COLUMNS)]
    rows += [f"gps,{t},1,1,0.5,0.5,0,,,,,1,1,0.5,0" for t in (0, 1, 2, 6, 7)]
    rows += [f"kf,{t},1,1,0.3,0.3,0,0.5,0,0.1,0.1,1,1,0.5,0" for t in range(8)]
    run.write_text("\n".join(rows) + "\n")
    settings, kf, gps = plot.load(str(run))
    assert settings == {"task": 1, "sigma_a": 0.1} and len(kf) == 8 and len(gps) == 5
    assert plot.outages(gps, 0.0, 7.0) == [(2.0, 6.0)]


def test_the_ekf_config_fuses_gps_position_and_odometry_velocity():
    yaml = pytest.importorskip("yaml")
    with open(paths.lab_file("config", "ekf.yaml"), encoding="utf-8") as handle:
        text = handle.read()
    p = yaml.safe_load(text)["/**"]["ros__parameters"]
    assert p["two_d_mode"] and not p["publish_tf"] and p["world_frame"] == "map"
    assert (p["pose0"], p["odom0"]) == ("gps_cov", "odom")
    assert [i for i, on in enumerate(p["pose0_config"]) if on] == [0, 1]                 # x, y
    assert [i for i, on in enumerate(p["odom0_config"]) if on] == [6, 7, 11]             # vx, vy, vyaw
    # the two lines task 4 switches on: they must parse, and the matrix must have one type (ROS 2 parameters)
    on = text.replace("    # pose0_rejection_threshold", "    pose0_rejection_threshold") \
             .replace("    # initial_estimate_covariance:", "    initial_estimate_covariance:").replace("\n    #   ", "\n        ")
    p = yaml.safe_load(on)["/**"]["ros__parameters"]
    assert p["pose0_rejection_threshold"] == 3.0
    matrix = p["initial_estimate_covariance"]
    assert len(matrix) == 225 and {type(v) for v in matrix} == {float}
    assert [matrix[16 * i] for i in (0, 1, 5)] == [100.0, 100.0, 1.0]


def test_the_launch_file_has_its_arguments():
    if not shutil.which("ros2"):
        pytest.skip("ROS 2 is not sourced")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]))   # this checkout,
    out = subprocess.run(["ros2", "launch", os.path.join(ROOT, "launch", "kalman.launch.py"), "--show-args"],
                         capture_output=True, text=True, timeout=60, env=env).stdout                # not an install
    for name in ("robot", "gps", "speed", "rviz", "headless", "seconds"):
        assert f"'{name}'" in out
    out = subprocess.run(["ros2", "launch", os.path.join(ROOT, "launch", "ekf.launch.py"), "--show-args"],
                         capture_output=True, text=True, timeout=60, env=env).stdout
    assert "'robot'" in out and "'params'" in out


def test_the_nodes_are_executable():
    for rel in ("student/kalman_node.py", "nodes/route_node.py", "nodes/monitor_node.py", "nodes/plot_run.py",
                "nodes/ekf_bridge.py"):
        path = os.path.join(ROOT, rel)
        assert os.access(path, os.X_OK), rel
        with open(path, encoding="utf-8") as handle:
            assert handle.readline().startswith("#!/usr/bin/env python3"), rel
