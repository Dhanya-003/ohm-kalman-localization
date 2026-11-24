#!/usr/bin/env python3
"""The monitor of the lab (given): how good is the estimate? Compares GPS and filter with the truth.

    ros2 run ohm_kalman_localization monitor_node.py      (the launch file starts it)

Every `period` s one line over the last `window` s of simulation time:

    t       simulation time
    GPS     RMSE of the raw fixes against the true position
    filter  RMSE and largest error of /<robot>/kf/pose, and how many times better than GPS
    sigma   mean standard deviation the filter claims (sqrt of the diagonal of P)
    NEES    mean of e^T P^-1 e / 2 over the estimates: about 1 if sigma is honest, > 1 overconfident
    heading RMS error of the heading, for an estimate with a heading of its own (the EKF of task 4)

and after every pass through a GPS zone of the config (the tunnel) its largest error and sigma. Each run of
your node (a new start of kalman_node.py) is written to runs/<task>_<settings>_<time>.csv for plot_run.py.
RViz gets the true path, the estimated path, the fixes and the zones on /<robot>/kalman_lab/*.
"""
import bisect
import collections
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import rclpy                                                  # noqa: E402
from geometry_msgs.msg import (Point, PoseStamped, PoseWithCovarianceStamped,   # noqa: E402
                               TwistWithCovarianceStamped)
from nav_msgs.msg import Path                                 # noqa: E402
from rclpy.executors import ExternalShutdownException         # noqa: E402
from rclpy.node import Node                                   # noqa: E402
from std_msgs.msg import ColorRGBA, String                    # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray        # noqa: E402

from ohm_kalman_localization import paths, route              # noqa: E402

COLUMNS = ["kind", "t", "x", "y", "sx", "sy", "sxy", "vx", "vy", "svx", "svy",
           "x_true", "y_true", "vx_true", "vy_true"]
KEEP = 60.0                     # s of path shown in RViz
NEW_RUN = 2.0                   # s without /kf/pose: the next estimate is a new run


class MonitorNode(Node):
    """ROS node that compares GPS and filter with the truth, writes each run to a CSV and draws RViz markers.

    Parameters: robot (name), window (s averaged per report), period (s between reports), config (simulator
    config with the GPS zones).

    Attributes:
        truth: Recent true poses (t, x, y, theta).
        gps: (t, squared error) of the fixes in the current window.
        est: (t, squared error, NEES, sigma) of the estimates in the current window.
        heading: (t, squared heading error) of estimates with a heading of their own.
        fixes, path: (t, x, y) of the fixes and of the estimate, for RViz.
        pending: Estimates and fixes waiting for a newer truth sample.
        zones: The GPS zones of the config, see load_zones.
        inside: The zone the robot is in, with its largest error and sigma so far; None outside.
    """

    def __init__(self):
        """Declare the parameters, subscribe to truth, GPS and the filter, create publishers and timer."""
        super().__init__("monitor")
        robot = self.declare_parameter("robot", "muster").value
        self.window = self.declare_parameter("window", 30.0).value
        self.period = self.declare_parameter("period", 10.0).value
        self.zones = load_zones(self.declare_parameter("config", "").value)
        self.truth = collections.deque(maxlen=4000)            # (t, x, y)
        self.gps = collections.deque()                         # (t, squared error)
        self.est = collections.deque()                         # (t, squared error, nees, sigma)
        self.heading = collections.deque()                     # (t, squared heading error) of an EKF
        self.fixes = collections.deque()                       # (t, x, y) for RViz
        self.path = collections.deque()                        # (t, x, y) of the estimate
        self.pending = collections.deque()                     # estimates newer than the newest truth
        self.twist, self.info, self.writer, self.file = None, None, None, None
        self.last_kf, self.last_report, self.inside = -1e9, None, None
        self.create_subscription(PoseStamped, f"/{robot}/truth", self.on_truth, 50)
        self.create_subscription(PoseWithCovarianceStamped, f"/{robot}/gps_cov", self.on_gps, 10)
        self.create_subscription(PoseWithCovarianceStamped, f"/{robot}/kf/pose", self.on_kf, 50)
        self.create_subscription(TwistWithCovarianceStamped, f"/{robot}/kf/twist", self.on_twist, 50)
        self.create_subscription(String, f"/{robot}/kf/info", self.on_info, 10)
        self.pub_truth = self.create_publisher(Path, f"/{robot}/kalman_lab/truth_path", 1)
        self.pub_kf = self.create_publisher(Path, f"/{robot}/kalman_lab/kf_path", 1)
        self.pub_markers = self.create_publisher(MarkerArray, f"/{robot}/kalman_lab/markers", 1)
        self.create_timer(0.5, self.draw)
        self.get_logger().info(f"comparing /{robot}/gps_cov and /{robot}/kf/pose with /{robot}/truth; "
                               f"start your node: ros2 run ohm_kalman_localization kalman_node.py")

    # ------------------------------------------------------------------ truth

    def on_truth(self, msg: PoseStamped) -> None:
        """Callback of /<robot>/truth: store the true pose and evaluate the estimates that waited for it.

        Args:
            msg: The true pose, geometry_msgs/msg/PoseStamped.
        """
        t = stamp(msg)
        if self.truth and t < self.truth[-1][0]:
            self.truth.clear()                                 # simulator restarted
        q = msg.pose.orientation
        self.truth.append((t, msg.pose.position.x, msg.pose.position.y, math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)))
        while self.pending and self.pending[0][0] <= t:
            self.evaluate(*self.pending.popleft())

    def true_at(self, t: float):
        """True state at time t, interpolated linearly between the two neighbouring samples.

        Args:
            t: Simulation time in s.

        Returns:
            (x, y, vx, vy, theta) in m, m/s and rad, or None if t is outside the buffer or in a gap > 0.5 s.
        """
        times = [s[0] for s in self.truth]
        i = bisect.bisect_left(times, t)
        if i == 0 or i >= len(times) or times[i] - times[i - 1] > 0.5:
            return None
        (t0, x0, y0, _), (t1, x1, y1, th1) = self.truth[i - 1], self.truth[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return x0 + a * (x1 - x0), y0 + a * (y1 - y0), (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0), th1

    # ------------------------------------------------------------------ measurements and estimates

    def on_gps(self, msg: PoseWithCovarianceStamped) -> None:
        """Callback of /<robot>/gps_cov: keep the fix for RViz and evaluate it against the truth.

        Args:
            msg: The fix, geometry_msgs/msg/PoseWithCovarianceStamped.
        """
        t, x, y = stamp(msg), msg.pose.pose.position.x, msg.pose.pose.position.y
        self.fixes.append((t, x, y))
        self.later(t, "gps", msg)

    def later(self, t: float, kind: str, msg, twist=None) -> None:
        """Evaluate a fix or an estimate now if the truth at t is known, otherwise as soon as it arrives.

        Args:
            t: Time stamp of the message in s.
            kind: "gps" or "kf".
            msg: The fix or the estimate, geometry_msgs/msg/PoseWithCovarianceStamped.
            twist: The latest kf/twist (estimates only) or None.
        """
        if self.truth and t <= self.truth[-1][0]:
            self.evaluate(t, kind, msg, twist)
        else:
            self.pending.append((t, kind, msg, twist))

    def evaluate(self, t: float, kind: str, msg, twist) -> None:
        """Compare one fix or estimate with the truth: error, NEES, sigma, heading; write a CSV row.

        Args:
            t: Time stamp in s.
            kind: "gps" or "kf".
            msg: The fix or the estimate, geometry_msgs/msg/PoseWithCovarianceStamped.
            twist: The latest kf/twist (geometry_msgs/msg/TwistWithCovarianceStamped) or None.
        """
        true = self.true_at(t)
        if true is None:
            return
        x, y, c = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.covariance
        if kind == "gps":
            self.gps.append((t, (x - true[0]) ** 2 + (y - true[1]) ** 2))
            self.row("gps", t, x, y, math.sqrt(c[0]), math.sqrt(c[7]), c[1], true=true)
            return
        ex, ey = x - true[0], y - true[1]
        det = c[0] * c[7] - c[1] * c[6]
        nees = (c[7] * ex * ex - (c[1] + c[6]) * ex * ey + c[0] * ey * ey) / det / 2.0 if det > 0 else float("inf")
        sigma = math.sqrt((max(c[0], 0.0) + max(c[7], 0.0)) / 2.0)
        self.est.append((t, ex * ex + ey * ey, nees, sigma))
        if c[35] > 1e-5:                                       # the estimate has a heading of its own (EKF)
            q = msg.pose.pose.orientation
            d = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z) - true[4]
            self.heading.append((t, math.atan2(math.sin(d), math.cos(d)) ** 2))
        tw = twist.twist if twist else None
        self.row("kf", t, x, y, math.sqrt(max(c[0], 0.0)), math.sqrt(max(c[7], 0.0)), c[1],
                 tw.twist.linear.x if tw else "", tw.twist.linear.y if tw else "",
                 math.sqrt(max(tw.covariance[0], 0.0)) if tw else "",
                 math.sqrt(max(tw.covariance[7], 0.0)) if tw else "", true=true)
        self.zone_pass(t, true, math.hypot(ex, ey), sigma)
        self.report(t)

    def on_twist(self, msg: TwistWithCovarianceStamped) -> None:
        """Callback of /<robot>/kf/twist: keep the latest velocity estimate for the next CSV row.

        Args:
            msg: geometry_msgs/msg/TwistWithCovarianceStamped.
        """
        self.twist = msg

    def on_info(self, msg: String) -> None:
        """Callback of /<robot>/kf/info: store the settings of the filter and open the CSV of the run.

        A new value of "run" (the filter was reset with /kalman/reset) starts a new run, as a pause would.

        Args:
            msg: std_msgs/msg/String with the settings as JSON.
        """
        try:
            info = json.loads(msg.data)
        except ValueError:
            return
        if self.info is not None and info.get("run") != self.info.get("run"):
            self.new_run(self.last_kf)
        self.info = info
        if self.writer is None and self.file is None:
            self.open_csv()

    def on_kf(self, msg: PoseWithCovarianceStamped) -> None:
        """Callback of /<robot>/kf/pose: start a new run after a pause, keep the path, evaluate the estimate.

        Args:
            msg: The estimate, geometry_msgs/msg/PoseWithCovarianceStamped.
        """
        t = stamp(msg)
        if abs(t - self.last_kf) > NEW_RUN:
            self.new_run(t)
        self.last_kf = t
        self.path.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y))
        self.later(t, "kf", msg, self.twist)

    # ------------------------------------------------------------------ statistics

    def new_run(self, t: float) -> None:
        """Close the CSV of the last run and clear all statistics.

        Args:
            t: Time of the first estimate of the new run in s.
        """
        if self.file:
            self.file.close()
            self.get_logger().info(f"run written to {os.path.relpath(self.file.name)}")
        self.file, self.writer, self.info = None, None, None
        self.gps.clear(), self.est.clear(), self.path.clear(), self.pending.clear(), self.heading.clear()
        self.last_report, self.inside = t, None
        self.get_logger().info(f"new run of the filter at t = {t:.1f} s")

    def report(self, t: float) -> None:
        """Print one line over the last `window` seconds, at most every `period` seconds.

        Args:
            t: Current simulation time in s.
        """
        if t - self.last_report < self.period:
            return
        self.last_report = t
        for queue in (self.gps, self.est, self.heading):
            while queue and queue[0][0] < t - self.window:
                queue.popleft()
        if not self.est:
            return
        rmse = math.sqrt(sum(e[1] for e in self.est) / len(self.est))
        worst = math.sqrt(max(e[1] for e in self.est))
        sigma = sum(e[3] for e in self.est) / len(self.est)
        nees = sum(min(e[2], 1e6) for e in self.est) / len(self.est)
        if self.gps:
            gps = math.sqrt(sum(e[1] for e in self.gps) / len(self.gps))
            text = f"t {t:6.1f} s | GPS {gps:.2f} m | filter {rmse:.2f} m, max {worst:.2f} m, {gps / rmse:.1f}x"
        else:
            text = f"t {t:6.1f} s | GPS: no fix | filter {rmse:.2f} m, max {worst:.2f} m"
        text += f" | sigma {sigma:.2f} m | NEES {nees:.1f}"
        if self.heading:
            text += f" | heading {math.degrees(math.sqrt(sum(e[1] for e in self.heading) / len(self.heading))):.1f} deg"
        self.get_logger().info(text)

    def zone_pass(self, t: float, true, error: float, sigma: float) -> None:
        """Track the largest error and sigma inside a GPS zone; print them when the robot leaves the zone.

        Args:
            t: Time of the estimate in s.
            true: True state at t, as returned by true_at.
            error: Distance between estimate and truth in m.
            sigma: Standard deviation the filter claims in m.
        """
        zone = next((z for z in self.zones if inside(z, true[0], true[1])), None)
        if self.inside and (zone is None or zone["name"] != self.inside["name"]):
            z = self.inside
            self.get_logger().info(f"{z['name']}: {t - z['t0']:.1f} s inside, largest error {z['max']:.2f} m, "
                                   f"largest sigma {z['sigma']:.2f} m")
            self.inside = None
        if zone is not None:
            if self.inside is None:
                self.inside = {"name": zone["name"], "t0": t, "max": 0.0, "sigma": 0.0}
            self.inside["max"] = max(self.inside["max"], error)
            self.inside["sigma"] = max(self.inside["sigma"], sigma)

    # ------------------------------------------------------------------ csv

    def open_csv(self) -> None:
        """Open runs/<task>_<settings>_<time>.csv; first line the settings as JSON, then the header."""
        i = self.info
        name = f"task{i.get('task', 0)}_a{i.get('sigma_a', 0)}"
        if i.get("task") == "ekf":
            name = "ekf_" + os.path.splitext(i.get("params", "ekf.yaml"))[0]
        elif i.get("task") == 4:
            name = f"task4_v{i.get('sigma_v', 0)}_w{i.get('sigma_w', 0)}"
            if i.get("theta0"):
                name += f"_theta{i.get('theta0')}_s{i.get('sigma_theta0')}"
        elif i.get("task", 1) >= 2:
            name += f"_v{i.get('sigma_v', 0)}"
        folder = paths.lab_file("runs")
        os.makedirs(folder, exist_ok=True)
        self.file = open(os.path.join(folder, f"{name}_{time.strftime('%H%M%S')}.csv"), "w", newline="")
        self.writer = csv.writer(self.file)
        self.file.write("# " + json.dumps(i) + "\n")
        self.writer.writerow(COLUMNS)

    def row(self, kind, t, x, y, sx, sy, sxy, vx="", vy="", svx="", svy="", true=None) -> None:
        """Write one row to the CSV of the run (nothing if no CSV is open).

        Args:
            kind: "gps" or "kf".
            t: Time stamp in s.
            x, y: Position in m.
            sx, sy, sxy: Standard deviations of x and y in m and their covariance in m^2.
            vx, vy, svx, svy: Velocity in the world frame and its standard deviations in m/s ("" for a fix).
            true: True state at t; its first four values (x, y, vx, vy) are written.
        """
        if self.writer is None:
            return
        values = [kind, t, x, y, sx, sy, sxy, vx, vy, svx, svy] + list(true[:4])
        self.writer.writerow([v if isinstance(v, str) else f"{v:.4f}" for v in values])

    # ------------------------------------------------------------------ RViz

    def draw(self) -> None:
        """Timer, twice per second: publish the true and estimated path, the fixes, the loop and the zones."""
        if not self.truth:
            return
        now = self.truth[-1][0]
        for queue in (self.fixes, self.path):
            while queue and queue[0][0] < now - KEEP:
                queue.popleft()
        truth = [s[:3] for s in self.truth if s[0] >= now - KEEP]
        self.pub_truth.publish(make_path(truth, now))
        self.pub_kf.publish(make_path(list(self.path), now))
        markers = MarkerArray()
        fixes = marker(0, "gps fixes", Marker.POINTS, now, (0.2, 0.75, 0.3, 1.0), 0.12)
        fixes.points = [Point(x=x, y=y, z=0.02) for _, x, y in self.fixes]
        loop = marker(1, "loop", Marker.LINE_STRIP, now, (0.6, 0.6, 0.6, 0.5), 0.02)
        loop.points = [Point(x=x, y=y) for x, y, _ in route.loop(0.25)]
        loop.points.append(loop.points[0])
        markers.markers += [fixes, loop]
        for k, z in enumerate(self.zones):
            x0, y0, x1, y1 = z["rect"]
            color = (0.3, 0.3, 0.35, 0.45) if z["block"] else (0.85, 0.55, 0.0, 0.3)
            box = marker(10 + k, "zones", Marker.CUBE, now, color, 1.0)
            box.pose.position.x, box.pose.position.y, box.pose.position.z = (x0 + x1) / 2, (y0 + y1) / 2, -0.02
            box.scale.x, box.scale.y, box.scale.z = x1 - x0, y1 - y0, 0.02
            label = marker(20 + k, "zone names", Marker.TEXT_VIEW_FACING, now, (1.0, 1.0, 1.0, 0.9), 0.35)
            label.pose.position.x, label.pose.position.y, label.pose.position.z = (x0 + x1) / 2, (y0 + y1) / 2, 0.3
            label.text = z["name"]
            markers.markers += [box, label]
        self.pub_markers.publish(markers)


def stamp(msg) -> float:
    """Time stamp of a ROS message in s.

    Args:
        msg: Any message with a std_msgs/msg/Header in msg.header.

    Returns:
        sec + nanosec * 1e-9 of the header stamp.
    """
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def make_path(samples, now: float) -> Path:
    """Build a nav_msgs/msg/Path in the frame map from samples (every second sample is used).

    Args:
        samples: Sequence of (t, x, y).
        now: Time stamp of the path in s.

    Returns:
        The nav_msgs/msg/Path.
    """
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp.sec, path.header.stamp.nanosec = int(now), int((now % 1.0) * 1e9)
    for t, x, y in samples[::2]:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


def marker(ident, ns, kind, now, rgba, size) -> Marker:
    """Build a visualization_msgs/msg/Marker in the frame map with the common fields set.

    Args:
        ident: Marker id, unique within its namespace.
        ns: Namespace of the marker.
        kind: Marker type, e.g. Marker.POINTS.
        now: Time stamp in s.
        rgba: Colour as (r, g, b, a), each 0 .. 1.
        size: Scale in m in x, y and z.

    Returns:
        The Marker, without points or pose.
    """
    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp.sec, m.header.stamp.nanosec = int(now), int((now % 1.0) * 1e9)
    m.ns, m.id, m.type, m.action = ns, ident, kind, Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = size
    m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
    return m


def load_zones(config: str) -> list:
    """Read the GPS zones of a simulator config.

    Args:
        config: Path of the JSON config the launch file started the simulator with; "" for none.

    Returns:
        List of dicts {"name": str, "rect": [x0, y0, x1, y1] in m, "block": True if no fix inside}.
    """
    if not config or not os.path.isfile(config):
        return []
    with open(config, encoding="utf-8") as handle:
        zones = json.load(handle).get("gps", {}).get("zones", [])
    return [{"name": z.get("name", "zone"), "rect": [float(v) for v in z["rect"]], "block": bool(z.get("block"))}
            for z in zones]


def inside(zone, x: float, y: float) -> bool:
    """Whether a point lies in the rectangle of a zone.

    Args:
        zone: A zone as returned by load_zones.
        x, y: The point in m.

    Returns:
        True if the point is inside or on the border.
    """
    x0, y0, x1, y1 = zone["rect"]
    return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)


def main():
    """Start ROS, spin the monitor until Ctrl-C or shutdown, close the CSV and clean up."""
    rclpy.init()
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.file:
            node.file.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
