"""The Kalman filter of the lab: prediction and correction, the same equations for every sensor.

predict is the linear prediction of tasks 1 and 2, predict_nonlinear the one of the EKF in task 3.

    x   state, numpy vector of n numbers        (x, y, vx, vy) in tasks 1 and 2, (x, y, theta) in task 3
    P   covariance of the state, n x n
    A   state transition matrix, n x n          Q  process noise, n x n
    z   measurement, m numbers                  H  observation matrix, m x n    R  measurement noise, m x m

In the script the measurement z is y-hat, the predicted measurement H x is y-tilde (eq. 3.7).

Replace each `raise NotImplementedError("STUDENT ...")` by your code. The red box STUDENT ... in the handout
says what is given there, what to write and which test checks it. numpy: `A @ B` is the matrix product,
`A.T` the transpose, `np.linalg.inv(S)` the inverse, `np.eye(4)` the identity.
"""
import numpy as np


class KalmanFilter:
    """Kalman filter: a linear prediction (tasks 1, 2), an EKF prediction (task 3), one correction for all.

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

    def predict_nonlinear(self, x_new, F, Q):
        """EKF prediction (task 3, STUDENT 3.3): the state through the non-linear model, the covariance through F.

        Overwrites self.x and self.P.

        Args:
            x_new: f(x, u), the state after the step, already computed by the caller (n numbers).
            F: n x n Jacobian of f by x at the old estimate.
            Q: n x n process noise of this step.
        """
        # overwrite self.x with x_new = f(x, u) and self.P with the EKF covariance: as in predict, F in place of A
        # TODO STUDENT 3.3
        raise NotImplementedError("STUDENT 3.3")

    def correct(self, z, H, R):
        """Correction with one measurement, eq. (3.10) to (3.12) (STUDENT 1.3 and 1.4).

        Overwrites self.x and self.P and stores the gain in self.K.

        Args:
            z: Measurement, m numbers (the script's y-hat).
            H: m x n observation matrix: H @ x is the measurement expected for the state x.
            R: m x m covariance of the measurement.
        """
        z = np.asarray(z, dtype=float)
        innovation = z - H @ self.x                # measured minus predicted measurement
        S = H @ self.P @ H.T + R                   # its covariance, the bracket of the gain
        # the gain K of eq. (3.10); S, its bracket, is computed above
        # TODO STUDENT 1.3
        raise NotImplementedError("STUDENT 1.3")
        # overwrite self.x and self.P with eq. (3.11) and (3.12); the innovation is computed above
        # TODO STUDENT 1.4
        raise NotImplementedError("STUDENT 1.4")
        self.P = 0.5 * (self.P + self.P.T)          # keeps P symmetric against rounding
        self.K = K
