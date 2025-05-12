import numpy as np
import scipy

import utils
from .state import NominalState, ErrorState
from orientation import RotationQuaterion
from utils import skew
from .models import ImuModel
from messages_pb2 import UkfState
import manifpy as manif

from measurement_models import Measurement


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
        self.S_inv = None
        self.innovation = None

    def Hx(self):
        """"Jacobian of the true state with respect to the error state"""
        qx, qy, qz,n = self.x.q
        Q_d_theta = .5 * np.array(
            [
                [-qx, -qy, -qz,],
                [n, -qz, qy],
                [qz, n, -qx,],
                [-qy, qx, n],
            ]
        )

        Hx = np.zeros((self.x.dof() + 1, self.x.dof()))
        Hx[:6, :6] = np.eye(6)
        Hx[6:10, 6:9] = Q_d_theta
        Hx[10:, 9:] = np.eye(6)
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
        self.S_inv = np.linalg.inv(S)
        K = (P @ H.T @ self.S_inv)
        z_pred = measurement.h(self.x)
        innovation = measurement.z - z_pred
        self.innovation = np.atleast_1d(innovation)
        innovation = self.innovation.reshape(-1, 1)

        x_err = K @ innovation
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

        theta = self.x_err.theta.flatten()
        theta_norm = np.linalg.norm(theta)
        # if theta_norm > 1e-8:
        #     dq = np.hstack([
        #         np.cos(theta_norm / 2),
        #         np.sin(theta_norm) * theta / theta_norm
        #     ])
        # else:
        # dq = RotationQuaterion(eta=1, epsilon=0.5 * self.x_err.theta.flatten())
        tangent = manif.SO3Tangent(self.x_err.theta)
        ori = self.x.ori.rplus(tangent)
        
        # dq /= np.linalg.norm(dq)

        self.x = NominalState(
            ori=ori, #dq.multiply(self.x.ori), #.multiply(dq),
            pos=self.x.pos + self.x_err.pos,
            vel=self.x.vel + self.x_err.vel,
            gyro_bias=self.x.gyro_bias + self.x_err.gyro_bias,
            acc_bias=self.x.acc_bias + self.x_err.acc_bias,
        )
        # self.x.g = self.x.g + self.x_err.g

        G = np.eye(self.x.dof())   # Can be left as eye

        self.P = G @ self.P @ G.T
        self.x_err = ErrorState.from_vec(np.zeros(self.x.dof()))

    def nees_pos(self, true_pos):
        idx = 2
        x_hat = self.x.position[:idx]
        x = true_pos[:idx]
        return (x_hat - x).T @ np.linalg.inv(self.P[:idx, :idx]) @ (x_hat - x)

    def nis(self):
        # Compute the NIS
        return self.innovation.T @ self.S_inv @ self.innovation

    def to_proto_msg(self):
        quat = self.x.q

        vel = self.x.R.T @ self.x.velocity
        roll, pitch, yaw = self.x.euler

        return UkfState(
            position_x=self.x.position[0],
            position_y=self.x.position[1],
            position_z=self.x.position[2],
            quaternion_w=quat[0],
            quaternion_x=quat[1],
            quaternion_y=quat[2],
            quaternion_z=quat[3],
            velocity_x=vel[0],
            velocity_y=vel[1],
            velocity_z=vel[2],
            heading=yaw,
            roll=roll,
            pitch=pitch,
            gyro_bias_x=self.x.gyroscope_bias[0],
            gyro_bias_y=self.x.gyroscope_bias[1],
            gyro_bias_z=self.x.gyroscope_bias[2],
            accel_bias_x=self.x.accelerometer_bias[0],
            accel_bias_y=self.x.accelerometer_bias[1],
            accel_bias_z=self.x.accelerometer_bias[2],
            covariance=np.diag(self.P)
        )

