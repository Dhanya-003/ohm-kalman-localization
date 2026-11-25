"""The Kalman filter of the lab: prediction and correction, the same equations for every sensor.

predict is the linear prediction of tasks 1 and 2, predict_nonlinear the one of the EKF in task 4.

    x   state, numpy vector of n numbers        (x, y, vx, vy) in tasks 1 and 2, (x, y, theta) in task 4
    P   covariance of the state, n x n
    A   state transition matrix, n x n          Q  process noise, n x n
    z   measurement, m numbers                  H  observation matrix, m x n    R  measurement noise, m x m

In the script the measurement z is y-hat, the predicted measurement H x is y-tilde (eq. 3.7).

Replace each `raise NotImplementedError("STUDENT ...")` by your code. The red box STUDENT ... in the handout
says what is given there, what to write and which test checks it.

numpy (examples in the handout, section 4.1):
    np.array([[1.0, 2.0], [3.0, 4.0]])     a 2 x 2 matrix, row by row; np.array([1.0, 2.0]) a vector
    A @ B, A @ x                           matrix times matrix, matrix times vector (A * B is entry by entry)
    A.T                                    the transpose
    np.linalg.inv(S)                       the inverse
    np.eye(4)                              the 4 x 4 identity
"""
import numpy as np


class KalmanFilter:
    """Kalman filter: a linear prediction (tasks 1, 2), an EKF prediction (task 4), one correction for all.

    The filter knows no sensor and no model: the caller passes A, Q (or f, F, Q) and z, H, R in every step.

    Attributes:
        x: State estimate, numpy vector of n numbers.
        P: Covariance of x, n x n numpy array.
        K: Gain of the last correction, n x m numpy array (empty before the first one), for the report.
    """

    def __init__(self, x0, P0):
        """Start the filter at a first estimate.

        Args:
            x0: Initial state, sequence of n numbers.
            P0: Initial covariance, n x n (nested sequence or numpy array).
        """
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.K = np.zeros((len(self.x), 0))       # the last gain, for the node's report

    def predict(self, A, Q):
        """Linear prediction over one time step, eq. (3.6) and (3.8) without the input B u (STUDENT 1.2).

        The state moves with the model, its covariance grows by Q. Overwrites self.x and self.P.

        Args:
            A: n x n state transition matrix of this step.
            Q: n x n process noise of this step.
        """
        # overwrite self.x and self.P with the prediction, eq. (3.6) and (3.8), here without B u
        # TODO STUDENT 1.2
        raise NotImplementedError("STUDENT 1.2")
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q

    def predict_nonlinear(self, x_new, F, Q):
        """EKF prediction (task 4, STUDENT 4.3): the state through the non-linear model, the covariance through F.

        Overwrites self.x and self.P.

        Args:
            x_new: f(x, u), the state after the step, already computed by the caller (n numbers).
            F: n x n Jacobian of f by x at the old estimate.
            Q: n x n process noise of this step.
        """
        # overwrite self.x with x_new = f(x, u) and self.P with the EKF covariance: as in predict, F in place of A
        # TODO STUDENT 4.3
        raise NotImplementedError("STUDENT 4.3")
        self.x = x_new
        self.P = F @ self.P @ F.T + Q

    def correct(self, z, H, R):
        """Correction with one measurement, eq. (3.10) to (3.12) (STUDENT 1.4; the gain is STUDENT 1.3).

        Overwrites self.x and self.P and stores the gain in self.K.

        Args:
            z: Measurement, m numbers (the script's y-hat).
            H: m x n observation matrix: H @ x is the measurement expected for the state x.
            R: m x m covariance of the measurement.
        """
        z = np.asarray(z, dtype=float)
        innovation = z - H @ self.x                # measured minus predicted measurement, y-hat - y-tilde
        K = kalman_gain(self.P, H, R)               # eq. (3.10), your function below
        # two lines: self.x with eq. (3.11), then self.P with eq. (3.12)
        # TODO STUDENT 1.4
        raise NotImplementedError("STUDENT 1.4")
        self.K = K                                # kept for the report of the node
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P


def kalman_gain(P, H, R):
    """Kalman gain K of eq. (3.10) (STUDENT 1.3).

    Args:
        P: n x n covariance of the predicted state, the script's P-tilde_n.
        H: m x n observation matrix.
        R: m x m covariance of the measurement.

    Returns:
        K: n x m numpy array, one row per entry of the state, one column per measured value.
    """
    # first S = H P H^T + R, the bracket of eq. (3.10), then K = P H^T S^-1
    # TODO STUDENT 1.3
    raise NotImplementedError("STUDENT 1.3")
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return K
