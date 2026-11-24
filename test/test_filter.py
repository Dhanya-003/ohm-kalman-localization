"""Self check of the student files, one task at a time, without ROS and without the simulator.

    python3 -m pytest test/test_filter.py -q -k task1
    python3 -m pytest test/test_filter.py -q -k task2
    python3 -m pytest test/test_filter.py -q -k task4        (3.4 and 3.5 need a sourced ROS 2)
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
    assert np.shape(A) == (4, 4), "A is a 4 x 4 numpy array"
    assert A @ np.array([0.0, 0.0, 2.0, 4.0]) == pytest.approx([1.0, 2.0, 2.0, 4.0]), \
        "A @ (0, 0, vx, vy) must move the position by (vx dt, vy dt): dt where x meets vx and y meets vy"
    assert A @ np.array([3.0, -1.0, 0.0, 0.0]) == pytest.approx([3.0, -1.0, 0.0, 0.0]), "standing still: x stays"
    assert M.cv_model(0.0) == pytest.approx(np.eye(4)), "dt = 0 gives the identity"


def test_task1_2_predict_moves_the_state_and_lets_P_grow():
    f = KF.KalmanFilter([0.0, 0.0, 1.0, 0.5], np.diag([0.0, 0.0, 2.0, 2.0]))
    f.predict(M.cv_model(1.0), np.zeros((4, 4)))
    assert f.x == pytest.approx([1.0, 0.5, 1.0, 0.5]), "self.x = A x of eq. (3.6)"
    # script eq. 3.5: one step from diag(p, q) gives p + q dt^2 and the coupling q dt
    assert f.P[0, 0] == pytest.approx(2.0) and f.P[0, 2] == pytest.approx(2.0) and f.P[2, 0] == pytest.approx(2.0), \
        "self.P = A P A^T + Q of eq. (3.8)"
    Q = M.process_noise(1.0, 0.5)
    f.predict(M.cv_model(1.0), Q)
    assert f.P == pytest.approx(f.P.T), "P must stay symmetric: A P A^T, not A P"
    assert f.P[0, 0] == pytest.approx(8.0 + Q[0, 0]), "Q is added once per prediction"


def test_task1_3_kalman_gain_is_the_example_of_the_script():
    # script, figure of the covariance ellipse: P = [2.56 1.2; 1.2 1.0], only x measured with sigma 0.6
    K = KF.kalman_gain(np.array([[2.56, 1.2], [1.2, 1.0]]), np.array([[1.0, 0.0]]), np.array([[0.36]]))
    assert np.shape(K) == (2, 1), "K has one row per entry of the state, one column per measured value"
    assert K[:, 0] == pytest.approx([0.877, 0.411], abs=1e-3), "K = P H^T S^-1 with S = H P H^T + R, eq. (3.10)"


def test_task1_3_kalman_gain_of_a_gps_fix_is_4_by_2():
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    K = KF.kalman_gain(np.diag([0.25, 0.25, 1.0, 1.0]), H, 0.25 * np.eye(2))
    assert np.shape(K) == (4, 2), "P is 4 x 4, H^T is 4 x 2, S^-1 is 2 x 2: K is 4 x 2"
    assert K == pytest.approx(np.array([[0.5, 0], [0, 0.5], [0, 0], [0, 0]])), \
        "P and R equal: the gain of the position is 0.5"


def test_task1_4_correct_is_the_example_of_the_script():
    # the same example: the measurement x = 1 moves x and, through the tilted ellipse, also v
    f = KF.KalmanFilter([0.0, 0.0], [[2.56, 1.2], [1.2, 1.0]])
    f.correct([1.0], np.array([[1.0, 0.0]]), np.array([[0.36]]))
    assert f.x == pytest.approx([0.877, 0.411], abs=1e-3), "self.x = x + K (innovation), eq. (3.11)"
    assert f.P == pytest.approx(np.array([[0.3156, 0.1479], [0.1479, 0.5068]]), abs=1e-4), \
        "self.P = (I - K H) P, eq. (3.12): P must shrink after the fix"


def test_task1_4_the_worked_example_finds_the_velocity():
    # script, worked example: start known, velocity (1, 0.5) unknown, R = 20, Q = 0, 100 fixes; uses 1.1 to 1.4
    rng = random.Random(4)
    f = KF.KalmanFilter([0.0, 0.0, 0.0, 0.0], np.diag([0.0, 0.0, 2.0, 2.0]))
    H, R = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), 20.0 * np.eye(2)
    for n in range(1, 101):
        f.predict(M.cv_model(1.0), np.zeros((4, 4)))
        f.correct([n * 1.0 + rng.gauss(0, math.sqrt(20)), n * 0.5 + rng.gauss(0, math.sqrt(20))], H, R)
    assert f.x[2:] == pytest.approx([1.0, 0.5], abs=0.1), "after 100 fixes the velocity is (1, 0.5)"
    assert math.sqrt(f.P[2, 2]) < 0.05, "sigma of the velocity must shrink with every fix: is self.P updated?"


def test_task1_5_gps_measurement_reads_z_H_and_R_from_the_message():
    z, H, R = M.gps_measurement(gps_msg(3.0, 4.0, 0.25, 0.36))
    assert np.asarray(z) == pytest.approx([3.0, 4.0]), "z = the position of the fix, msg.pose.pose.position"
    assert H == pytest.approx(np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])), "H x must give (x, y) of the state"
    assert R == pytest.approx(np.diag([0.25, 0.36])), "R holds variances: entry 0 and 7, not squared again"


def test_task1_all_blocks_together_smooth_noisy_fixes():
    # the whole filter of task 1 (1.1 to 1.5) on 40 s of fixes, sigma 0.5 m
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
    assert sum(errors) / len(errors) < 0.5 * sum(gps) / len(gps), \
        "the filter is not twice as good as the fixes: run the tests of 1.1 to 1.5 one by one"


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
    assert np.asarray(z) == pytest.approx(world, abs=1e-9), \
        f"z: the velocity of the robot rotated by theta = {theta:.2f} into the world frame, eq. (2)"
    assert H == pytest.approx(np.array([[0, 0, 1.0, 0], [0, 0, 0, 1.0]])), "H x must give (vx, vy) of the state"
    assert R == pytest.approx(np.diag([0.0025, 0.0025])), "R = sigma_v squared"


def test_task2_the_odometry_pins_the_velocity_and_carries_the_position():
    f = KF.KalmanFilter([0.0, 0.0, 0.0, 0.0], np.diag([0.25, 0.25, 1.0, 1.0]))
    for _ in range(100):                                    # 2 s at 50 Hz, driving north
        f.predict(M.cv_model(0.02), M.process_noise(0.02, 0.5))
        f.correct(*M.odom_measurement(odom_msg(math.pi / 2, 0.5, 0.0), 0.05))
    assert f.x[2:] == pytest.approx([0.0, 0.5], abs=0.02), "driving north at 0.5 m/s: velocity (0, 0.5)"
    assert f.x[1] == pytest.approx(1.0, abs=0.1), "after 2 s the robot is 1 m further north"
    assert math.sqrt(f.P[3, 3]) < 0.05, "the odometry pins the velocity: sigma of vy below 0.05 m/s"


# ------------------------------------------------------------------ task 4

@pytest.mark.parametrize("x, u, dt, new", [
    ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), 2.0, (1.0, 0.0, 0.0)),                       # straight ahead
    ((1.0, 2.0, math.pi / 2), (0.5, 0.0, 0.0), 2.0, (1.0, 3.0, math.pi / 2)),       # heading north
    ((0.0, 0.0, math.pi / 2), (0.0, 0.3, 0.0), 1.0, (-0.3, 0.0, math.pi / 2)),      # sideways, to the left
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), 1.0, (0.0, 0.0, 0.5)),                       # turning on the spot
    ((0.0, 0.0, 3.0), (0.0, 0.0, 0.5), 1.0, (0.0, 0.0, 3.5 - 2 * math.pi)),         # theta stays in -pi .. pi
])
def test_task3_1_motion_model_drives_with_the_velocity_of_the_robot(x, u, dt, new):
    assert M.motion_model(np.array(x), u, dt) == pytest.approx(new, abs=1e-9), \
        f"f(x, u) of the handout for x = {x}, u = {u}, dt = {dt}"


@pytest.mark.parametrize("theta", [0.0, 0.7, 2.5, -1.9])
def test_task3_2_the_jacobian_is_the_derivative_of_the_motion_model(theta):
    x, u, dt, eps = np.array([3.0, -1.0, theta]), (0.4, 0.1, 0.3), 0.5, 1e-6
    numeric = np.column_stack([(M.motion_model(x + eps * e, u, dt) - M.motion_model(x - eps * e, u, dt)) / (2 * eps)
                               for e in np.eye(3)])
    assert M.motion_jacobian(x, u, dt) == pytest.approx(numeric, abs=1e-6), \
        "F is not the derivative of your motion_model: compare the third column with your preparation"


def test_task3_3_predict_nonlinear_takes_f_and_carries_P_with_F():
    f = KF.KalmanFilter([0.0, 0.0, 0.0], np.diag([0.1, 0.1, 0.2]))
    F = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    Q = np.diag([0.01, 0.01, 0.001])
    f.predict_nonlinear(np.array([1.0, 2.0, 0.3]), F, Q)
    assert f.x == pytest.approx([1.0, 2.0, 0.3]), "self.x = x_new"
    assert f.P == pytest.approx(F @ np.diag([0.1, 0.1, 0.2]) @ F.T + Q), "self.P = F P F^T + Q"


def test_task3_the_ekf_finds_a_wrong_heading_from_gps_alone():
    rng, dt, heading, u = random.Random(2), 0.02, 0.5, (0.5, 0.0, 0.0)
    f, pose = KF.KalmanFilter([2.0, 1.0, -0.5], np.diag([0.25, 0.25, 1.0])), np.array([2.0, 1.0, heading])
    for n in range(1, 1501):                                        # 30 s at 50 Hz, a fix every 0.2 s
        pose = pose + np.array([math.cos(heading), math.sin(heading), 0.0]) * 0.5 * dt
        x = f.x
        f.predict_nonlinear(M.motion_model(x, u, dt), M.motion_jacobian(x, u, dt), M.input_noise(x, dt, 0.05, 0.02))
        if n % 10 == 0:
            z, H, R = M.gps_measurement(gps_msg(pose[0] + rng.gauss(0, 0.5), pose[1] + rng.gauss(0, 0.5), 0.25, 0.25))
            f.correct(z, H[:, :3], R)
    assert abs(math.degrees(f.x[2] - heading)) < 3.0
    assert math.dist(f.x[:2], pose[:2]) < 0.4




@pytest.fixture
def node():
    """The KalmanNode, on a ROS domain of its own so that it meets no running simulator."""
    rclpy = pytest.importorskip("rclpy")
    import kalman_node
    rclpy.init(domain_id=231)
    n = kalman_node.KalmanNode()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_task3_4_the_node_offers_the_reset_service(node):
    names = [node.resolve_service_name(service.srv_name) for service in node.services]
    assert "/kalman/reset" in names, "no service /kalman/reset: STUDENT 4.4 in create_reset_service"


def test_task3_5_reset_drops_the_filter_and_reads_the_parameters(node):
    from rclpy.parameter import Parameter
    from std_srvs.srv import Trigger
    node.kf = KF.KalmanFilter([1.0, 2.0, 0.0], np.eye(3))
    node.set_parameters([Parameter("theta0", value=1.5), Parameter("sigma_theta0", value=2.0)])
    response = node.on_reset(Trigger.Request(), Trigger.Response())
    assert isinstance(response, Trigger.Response), "return the response"
    assert response.success and response.message
    assert node.kf is None, "drop the filter: it starts again at the next fix"
    assert node.p["theta0"] == 1.5 and node.p["sigma_theta0"] == 2.0, "read the parameters again"
