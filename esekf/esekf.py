import numpy as np
import scipy

import utils
from .state import NominalState, ErrorState
from orientation import RotationQuaterion
from utils import skew
from .models import ImuModel, ImuMeasurement, Measurement


class ESEFK:
    def __init__(
        self,
        x0: NominalState,
        P0: np.ndarray,
        model: ImuModel,
    ):
        self.x = x0
        self.x_err = ErrorState.from_vec(np.zeros(x0.dof()))
        self.P = P0
        self.model = model

    def Hx(self):
        """"Jacobian of the true state with respect to the error state"""
        n, qx, qy, qz = self.x.ori.as_vec()
        Q_d_theta = .5 * np.array(
            [
                [-qx, -qy, -qz,],
                [n, -qz, qy],
                [qz, n, -qx,],
                [-qy, qx, n],
            ]
        )

        Hx = np.zeros((self.x.dof() + 1, self.x.dof()))
        Hx[:4, :3] = Q_d_theta
        Hx[4:, 3:] = np.eye(12)
        return Hx


    def propagate(self, u: np.ndarray, dt: float):
        """
        ESEKF prediction
        """
        self.x = self.model.predict_nominal(self.x, u, dt)
        self.P = self.model.predict_err(self.x, self.P, u, dt)
        # self.injection_and_reset()

    def update(self, measurement: Measurement, dt: float):
        H = measurement.H(self.x) @ self.Hx()

        P = self.P
        R = measurement.R

        S = H @ P @ H.T + R
        K = (P @ H.T @ np.linalg.inv(S))
        innovation = measurement.z - measurement.h(self.x)
        innovation = np.atleast_1d(innovation).reshape(-1, 1)

        x_err = K @ innovation
        print(x_err)
        self.x_err = ErrorState.from_vec(x_err.flatten())

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T  # Joseph form for more stability
        self.injection_and_reset()

    def injection_and_reset(self):
        """
        Injects the error state into the nominal state.
        All info is now stored in the nominal state.
        The error state is reset after injection.
        The posterior covariance is then updated to reflect this reset
        Joan Solà. Quaternion kinematics for the error-state Kalman filter.
        Chapter 6.2 eq. 282-286
        """
        dq = np.hstack(([1], 0.5 * self.x_err.theta.flatten()))

        self.x = NominalState(
            ori=RotationQuaterion.multiply(self.x.ori, dq),
            pos=self.x.pos + self.x_err.pos,
            vel=self.x.vel + self.x_err.vel,
            gyro_bias=self.x.gyro_bias + self.x_err.gyro_bias,
            acc_bias=self.x.acc_bias + self.x.acc_bias,
        )
        # self.x.g = self.x.g + self.x_err.g

        G = np.eye(15)   # Can be left as eye
        G[:3, :3] = np.eye(3) - self.x_err.theta

        self.P = G @ self.P @ G.T
        self.x_err = ErrorState.from_vec(np.zeros(self.x.dof()))
