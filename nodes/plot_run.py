#!/usr/bin/env python3
"""Plots of one run of your filter, from the CSV the monitor wrote (given).

    ros2 run ohm_kalman_localization plot_run.py                 the newest run in runs/
    ros2 run ohm_kalman_localization plot_run.py runs/task1_a0.1_141502.csv
    ros2 run ohm_kalman_localization plot_run.py --save          also save it as .png next to the .csv

Three plots: the path (truth, GPS, estimate), the estimated velocity against the true one, and the
standard deviations from the diagonal of P over time. Grey bands: no fix for more than 2 s.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from ohm_kalman_localization import paths                    # noqa: E402


def load(path: str):
    """(settings, rows of the filter, rows of the GPS) of one CSV."""
    with open(path, encoding="utf-8") as handle:
        first = handle.readline()
        settings = json.loads(first[1:]) if first.startswith("#") else {}
        if not settings:
            handle.seek(0)
        rows = list(csv.DictReader(handle))
    number = lambda r: {k: (float(v) if v not in ("", None) and k != "kind" else v) for k, v in r.items()}  # noqa
    kf = [number(r) for r in rows if r["kind"] == "kf"]
    gps = [number(r) for r in rows if r["kind"] == "gps"]
    return settings, kf, gps


def outages(gps, start: float, end: float, gap: float = 2.0) -> list:
    """(t0, t1) of every stretch longer than `gap` without a fix."""
    times = [start] + [r["t"] for r in gps] + [end]
    return [(a, b) for a, b in zip(times, times[1:]) if b - a > gap]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", help="a run from runs/ (default: the newest)")
    ap.add_argument("--save", action="store_true", help="save the figure as .png next to the .csv")
    args = ap.parse_args()
    path = args.csv
    if path is None:
        runs = sorted(glob.glob(paths.lab_file("runs", "*.csv")), key=os.path.getmtime)
        if not runs:
            print("no run in runs/ yet: start the launch file and your node first", file=sys.stderr)
            return 1
        path = runs[-1]
    settings, kf, gps = load(path)
    if not kf:
        print(f"{path}: no estimate in it", file=sys.stderr)
        return 1
    import matplotlib
    if args.save and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = kf[0]["t"]
    t = [r["t"] - t0 for r in kf]
    fig = plt.figure(figsize=(13, 7.5))
    fig.suptitle(f"{os.path.basename(path)}   {json.dumps(settings)}", fontsize=9)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot([r["x_true"] for r in kf], [r["y_true"] for r in kf], color="0.3", lw=1.5, label="truth")
    ax.plot([r["x"] for r in gps], [r["y"] for r in gps], "x", color="tab:green", ms=4, label="GPS")
    ax.plot([r["x"] for r in kf], [r["y"] for r in kf], color="tab:red", lw=1.2, label="estimate")
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    bands = outages(gps, kf[0]["t"], kf[-1]["t"])
    v = fig.add_subplot(2, 2, 2)
    for name, color in (("vx", "tab:blue"), ("vy", "tab:orange")):
        v.plot(t, [r[name + "_true"] for r in kf], color=color, lw=1, alpha=0.5, label=f"{name} true")
        v.plot(t, [r[name] if r[name] != "" else float("nan") for r in kf], color=color, lw=1.4,
               label=f"{name} estimate")
    v.set_ylabel("velocity / (m/s)")
    v.legend(loc="upper right", fontsize=8, ncol=2)
    v.grid(alpha=0.3)

    s = fig.add_subplot(2, 2, 4, sharex=v)
    for name, label, color in (("sx", "sigma x", "tab:red"), ("sy", "sigma y", "tab:purple"),
                               ("svx", "sigma vx", "tab:blue"), ("svy", "sigma vy", "tab:orange")):
        s.plot(t, [r[name] if r[name] != "" else float("nan") for r in kf], color=color, lw=1.2, label=label)
    s.set_yscale("log")
    s.set_xlabel("time since the start of the filter / s")
    s.set_ylabel("sqrt of the diagonal of P")
    s.legend(loc="upper right", fontsize=8, ncol=2)
    s.grid(alpha=0.3, which="both")
    for axis in (v, s):
        for a, b in bands:
            axis.axvspan(a - t0, b - t0, color="0.85", zorder=0)
    fig.tight_layout()
    if args.save:
        target = os.path.splitext(path)[0] + ".png"
        fig.savefig(target, dpi=110)
        print(f"saved {target}")
    if os.environ.get("DISPLAY"):
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
