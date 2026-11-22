"""The Kalman filter of the lab: prediction and correction, the same equations for every sensor.

predict is the linear prediction of tasks 1 and 2, predict_nonlinear the one of the EKF in task 3.

    x   state, numpy vector of n numbers        (x, y, vx, vy) in tasks 1 and 2, (x, y, theta) in task 3
    P   covariance of the state, n x n
    A   state transition matrix, n x n          Q  process noise, n x n
    z   measurement, m numbers                  H  observation matrix, m x n    R  measurement noise, m x m

In the script the measurement z is y-hat, the predicted measurement H x is y-tilde (eq. 3.7).

Replace each `raise NotImplementedError("STUDENT ...")` by your code: the comment above it says in words
what goes there, the task in the handout gives the equation. numpy: `A @ B` is the matrix product,
`A.T` the transpose, `np.linalg.inv(S)` the inverse, `np.eye(4)` the identity.
"""
import numpy as np


class KalmanFilter:

    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.K = np.zeros((len(self.x), 0))       # the last gain, for the node's report

    def predict(self, A, Q):
        """Prediction over one time step: the state moves with the model, its uncertainty grows."""
        # the state through the model (no input B u here), then the covariance through the model plus Q
        # TODO STUDENT 1.2
        raise NotImplementedError("STUDENT 1.2")

    def predict_nonlinear(self, x_new, F, Q):
        """EKF prediction (task 3): the state through the non-linear model, the covariance through its Jacobian."""
        # x_new = f(x, u) is already computed: take it as the state; the covariance as in predict, with F for A
        # TODO STUDENT 3.3
        raise NotImplementedError("STUDENT 3.3")

    def correct(self, z, H, R, gate=0.0):
        """Correction with one measurement. False if the gate (extra task Z.1) rejects it."""
        z = np.asarray(z, dtype=float)
        innovation = z - H @ self.x                # measured minus predicted measurement
        S = H @ self.P @ H.T + R                   # its covariance, the bracket of the gain
        if gate > 0.0 and mahalanobis2(innovation, S) > gate:
            return False
        # the gain: how much of the innovation goes into the state
        # TODO STUDENT 1.3
        raise NotImplementedError("STUDENT 1.3")
        # the state moves by the gain times the innovation, then the covariance shrinks
        # TODO STUDENT 1.4
        raise NotImplementedError("STUDENT 1.4")
        self.P = 0.5 * (self.P + self.P.T)          # keeps P symmetric against rounding
        self.K = K
        return True


def mahalanobis2(innovation, S):
    """Extra task Z.1: the squared Mahalanobis distance of the innovation, whose covariance is S."""
    # the innovation in units of its own uncertainty: innovation transposed, times S inverse, times innovation
    # TODO STUDENT Z.1
    raise NotImplementedError("STUDENT Z.1")
    return d2
