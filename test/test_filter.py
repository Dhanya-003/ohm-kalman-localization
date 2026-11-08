"""Self check of the student files, one task at a time, without ROS and without the simulator.

    python3 -m pytest test/test_filter.py -q -k task1
    python3 -m pytest test/test_filter.py -q -k task2
    python3 -m pytest test/test_filter.py -q -k extra
"""
import math
import os
import random
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "student"))

import kalman_filter as KF                                    # noqa: E402
import models as M                                            # noqa: E402


def gps_msg(x, y, var_x, var_y):
    cov = [0.0] * 36
    cov[0], cov[7], cov[14], cov[35] = var_x, var_y, 1e-12, 0.01
    return NS(pose=NS(pose=NS(position=NS(x=x, y=y, z=0.0)), covariance=cov))


def odom_msg(theta, vx, vy):
    q = NS(x=0.0, y=0.0, z=math.sin(theta / 2), w=math.cos(theta / 2))
    return NS(pose=NS(pose=NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=q)),
              twist=NS(twist=NS(linear=NS(x=vx, y=vy, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))))


# ------------------------------------------------------------------ task 1

def test_task1_1_cv_model_moves_the_position_with_the_velocity():
    A = M.cv_model(0.5)
    assert A.shape == (4, 4)
    assert A @ np.array([0.0, 0.0, 2.0, 4.0]) == pytest.approx([1.0, 2.0, 2.0, 4.0])
    assert A @ np.array([3.0, -1.0, 0.0, 0.0]) == pytest.approx([3.0, -1.0, 0.0, 0.0])
    assert M.cv_model(0.0) == pytest.approx(np.eye(4))


def test_task1_2_predict_moves_the_state_and_lets_P_grow():
    f = KF.KalmanFilter([0.0, 0.0, 1.0, 0.5], np.diag([0.0, 0.0, 2.0, 2.0]))
    f.predict(M.cv_model(1.0), np.zeros((4, 4)))
    assert f.x == pytest.approx([1.0, 0.5, 1.0, 0.5])
    # script eq. 3.5: one step from diag(p, q) gives p + q dt^2 and the coupling q dt
    assert f.P[0, 0] == pytest.approx(2.0) and f.P[0, 2] == pytest.approx(2.0) and f.P[2, 0] == pytest.approx(2.0)
    Q = M.process_noise(1.0, 0.5)
    f.predict(M.cv_model(1.0), Q)
    assert f.P == pytest.approx(f.P.T)
    assert f.P[0, 0] == pytest.approx(8.0 + Q[0, 0])


def test_task1_3_4_correct_is_the_example_of_the_script():
    # script, figure of the covariance ellipse: P = [2.56 1.2; 1.2 1.0], only x measured with sigma 0.6
    f = KF.KalmanFilter([0.0, 0.0], [[2.56, 1.2], [1.2, 1.0]])
    assert f.correct([1.0], np.array([[1.0, 0.0]]), np.array([[0.36]]))
    assert f.K[:, 0] == pytest.approx([0.877, 0.411], abs=1e-3)                       # STUDENT 1.3
    assert f.x == pytest.approx([0.877, 0.411], abs=1e-3)                             # STUDENT 1.4
    assert f.P == pytest.approx(np.array([[0.3156, 0.1479], [0.1479, 0.5068]]), abs=1e-4)


def test_task1_3_4_the_worked_example_finds_the_velocity():
    # script, worked example: start known, velocity (1, 0.5) unknown, R = 20, Q = 0, 100 fixes
    rng = random.Random(4)
    f = KF.KalmanFilter([0.0, 0.0, 0.0, 0.0], np.diag([0.0, 0.0, 2.0, 2.0]))
    H, R = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), 20.0 * np.eye(2)
    for n in range(1, 101):
        f.predict(M.cv_model(1.0), np.zeros((4, 4)))
        f.correct([n * 1.0 + rng.gauss(0, math.sqrt(20)), n * 0.5 + rng.gauss(0, math.sqrt(20))], H, R)
    assert f.x[2:] == pytest.approx([1.0, 0.5], abs=0.1)
    assert math.sqrt(f.P[2, 2]) < 0.05


def test_task1_5_gps_measurement_reads_z_H_and_R_from_the_message():
    z, H, R = M.gps_measurement(gps_msg(3.0, 4.0, 0.25, 0.36))
    assert np.asarray(z) == pytest.approx([3.0, 4.0])
    assert H == pytest.approx(np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    assert R == pytest.approx(np.diag([0.25, 0.36])), "R holds variances: entry 0 and 7, not squared again"


def test_task1_5_the_filter_smooths_noisy_fixes():
    rng = random.Random(1)
    f, errors, gps = None, [], []
    for n in range(200):
        t = 0.2 * n
        truth = (t * 0.5, 1.0 + 0.2 * t)
        msg = gps_msg(truth[0] + rng.gauss(0, 0.5), truth[1] + rng.gauss(0, 0.5), 0.25, 0.25)
        z, H, R = M.gps_measurement(msg)
        if f is None:
            f = KF.KalmanFilter([z[0], z[1], 0, 0], np.diag([0.25, 0.25, 1.0, 1.0]))
            continue
        f.predict(M.cv_model(0.2), M.process_noise(0.2, 0.05))
        f.correct(z, H, R)
        if n > 50:
            errors.append(math.dist(f.x[:2], truth))
            gps.append(math.dist(z, truth))
    assert sum(errors) / len(errors) < 0.5 * sum(gps) / len(gps)


# ------------------------------------------------------------------ task 2

@pytest.mark.parametrize("theta, body, world", [
    (0.0, (0.5, 0.0), (0.5, 0.0)),
    (math.pi / 2, (0.5, 0.0), (0.0, 0.5)),
    (math.pi, (0.4, 0.0), (-0.4, 0.0)),
    (math.pi / 2, (0.0, 0.3), (-0.3, 0.0)),
    (math.radians(30), (0.5, 0.1), (0.5 * math.cos(math.radians(30)) - 0.1 * math.sin(math.radians(30)),
                                     0.5 * math.sin(math.radians(30)) + 0.1 * math.cos(math.radians(30)))),
])
def test_task2_1_odom_measurement_rotates_the_velocity_into_the_world(theta, body, world):
    z, H, R = M.odom_measurement(odom_msg(theta, *body), 0.05)
    assert np.asarray(z) == pytest.approx(world, abs=1e-9)
    assert H == pytest.approx(np.array([[0, 0, 1.0, 0], [0, 0, 0, 1.0]]))
    assert R == pytest.approx(np.diag([0.0025, 0.0025])), "R = sigma_v squared"


def test_task2_the_odometry_pins_the_velocity_and_carries_the_position():
    f = KF.KalmanFilter([0.0, 0.0, 0.0, 0.0], np.diag([0.25, 0.25, 1.0, 1.0]))
    for _ in range(100):                                    # 2 s at 50 Hz, driving north
        f.predict(M.cv_model(0.02), M.process_noise(0.02, 0.5))
        f.correct(*M.odom_measurement(odom_msg(math.pi / 2, 0.5, 0.0), 0.05))
    assert f.x[2:] == pytest.approx([0.0, 0.5], abs=0.02)
    assert f.x[1] == pytest.approx(1.0, abs=0.1)
    assert math.sqrt(f.P[3, 3]) < 0.05


# ------------------------------------------------------------------ extra task

def test_extra_1_mahalanobis_distance():
    assert KF.mahalanobis2(np.array([1.0, 0.0]), np.diag([0.25, 0.25])) == pytest.approx(4.0)
    assert KF.mahalanobis2(np.array([0.3, -0.4]), np.eye(2)) == pytest.approx(0.25)


def test_extra_1_the_gate_rejects_a_fix_far_off_and_keeps_a_close_one():
    f = KF.KalmanFilter([0.0, 0.0, 0.0, 0.0], np.diag([0.01, 0.01, 0.01, 0.01]))
    z, H, R = M.gps_measurement(gps_msg(3.0, 0.0, 0.25, 0.25))
    assert not f.correct(z, H, R, gate=9.21)
    assert f.x == pytest.approx([0.0, 0.0, 0.0, 0.0])
    z, H, R = M.gps_measurement(gps_msg(0.5, 0.0, 0.25, 0.25))
    assert f.correct(z, H, R, gate=9.21) and f.x[0] > 0.0
