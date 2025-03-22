import numpy as np
import scipy

import utils
from .state import NominalState, ErrorState
from orientation import RotationQuaterion
from utils import skew
from .models import ImuMeasurement, ImuModel
from models import Measurement


class ESEFK:
    def __init__(
        self,
        x0: NominalState,
        P0: np.ndarray,
        model: ImuModel,
        gyro_bias_std: np.ndarray,
        acc_bias_std: np.ndarray,
        gyro_bias_p: np.ndarray,
        acc_bias_p: np.ndarray,
    ):
        self.x = NominalState
        self.x_err = np.zeros(18)
        self.P = P0
        self.model = model
        self.gyro_bias_std = gyro_bias_std
        self.acc_bias_std = acc_bias_std
        self.gyro_bias_p = gyro_bias_p
        self.acc_bias_p = acc_bias_p
        self.acc_correction = np.eye(3)
        self.gyro_correction = np.eye(3)

        def diag3(x):
            return np.diag([x] * 3)

        self.Q_c = scipy.linalg.block_diag(
            self.accm_corr @ diag3(self.accm_std**2) @ self.accm_corr.T,
            self.gyro_corr @ diag3(self.gyro_std**2) @ self.gyro_corr.T,
            diag3(self.accm_bias_std**2),
            diag3(self.gyro_bias_std**2),
        )

    def predict(self, u: ImuMeasurement, dt: float):
        """
        ESEKF prediction
        """
        self.x = self.model.predict_nominal(self.x, )
        self.P = self.model.predict_err(self.x, self.P, u, dt)

    def update(self, measurement: Measurement):
        H = measurement.H(self.x)

        H = self.H()
        P = self.error_state.covariance
        R = measurement.R

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        innovation = measurement.z - measurement.h()
        self.x_err = K @ innovation

        I_KH = np.eye(18) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T  # Joseph form for more stability

    def injection_and_reset(self):
        """
        Injects the error state into the nominal state.
        All info is now stored in the nominal state.
        The error state is reset after injection.
        The posterior covariance is then updated to reflect this reset
        Joan Solà. Quaternion kinematics for the error-state Kalman filter.
        Chapter 6.2 eq. 282-286
        """
        dq = np.hstack(([1], 0.5 * self.x_err.theta))
        self.x.ori = RotationQuaterion.multiply(self.ori, dq)
        self.x.pos = self.x.pos + self.x_err.pos
        self.x.vel = self.x.vel + self.x_err.vel
        self.x.gyro_bias = self.x.gyro_bias + self.x_err.gyro_bias
        self.x.acc_bias = self.x.acc_bias + self.x.acc_bias
        self.x.g = self.x.g + self.x_err.g

        G = np.eye(18)   # Can be left as eye
        G[:3, :3] = np.eye(3) - self.x_err.theta

        self.error_state.covariance = G @ self.P @ G.T
