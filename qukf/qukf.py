import numpy as np
import scipy
from orientation import RotationQuaterion, AttitudeError, average, quaternion_weighted_average
from sigma_points import SigmaPoints
# from dataclasses import dataclass
# from typing import Callable
from .models import ImuModel
from .state import State
from scipy.spatial.transform import Rotation as Rot
from sigma_points import MwereSigmaPoints
from messages_pb2 import UkfState


class QUKF:
    def __init__(
        self,
        model,
        dim_q,
        x0,
        P0,
        sigma_points,
    ):
        self.model = model
        self.dim_x = x0.dof()
        self.dim_q = dim_q
        self.w = np.zeros(dim_q)  # Noise mean
        self.x = x0
        self.P = P0
        n = self.dim_x + self.dim_q
        self.P_aug = np.zeros((n, n))
        self.P_aug[:self.dim_x, :self.dim_x] = self.P
        self.P_aug[self.dim_x:, self.dim_x:] = self.model.Q
        self.sigma_points = sigma_points
        self.propgated_sigmas = np.tile(self.x.as_vec(), (self.sigma_points.num_sigmas, 1))
        self.delta_sigmas = np.zeros((self.sigma_points.num_sigmas, self.dim_x))
        self.S_inv = None
        self.innovation = None

    def propagate(self, u, dt):
        # self.P += np.eye(self.dim_x) * 1e-8
        Q = self.model.Q

        # Augment the covariance matrix
        self.P_aug[:self.dim_x, :self.dim_x] = self.P
        self.P_aug[self.dim_x:, self.dim_x:] = Q

        # Generate zero mean error sigma points
        delta_x_aug_sigmas = self.sigma_points.compute_sigma_points(self.P_aug)
        sigmas = self.propgated_sigmas
        # sigmas[0] = self.model.f(self.x, dt, np.zeros(self._dim_q)).as_vec()
        for i, delta_x_aug in enumerate(delta_x_aug_sigmas):
            delta_x = delta_x_aug[:self.dim_x]
            w = delta_x_aug[self.dim_x:]
            xi = State.add_error_state(self.x, delta_x)
            sigmas[i] = self.model.f(xi, u, dt, w)

        q_bar = quaternion_weighted_average(sigmas[:, :4], self.sigma_points.Wm)
        # print("q_bar", q_bar)

        x_bar_ = np.dot(self.sigma_points.Wm.flatten(), sigmas[:, 4:])
        x_bar = np.concatenate([q_bar, x_bar_])

        P = np.zeros_like(self.P)
        for i, sigma in enumerate(sigmas):
            # x = State.from_vec(sigma)
            # dx = x.to_error_state(x_bar)
            dx = State.to_error_state(sigma, x_bar)
            P += self.sigma_points.Wc[i] * np.outer(dx, dx)
            self.delta_sigmas[i] = dx

        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals[eigvals < 0] = 0
        P_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.P = (P_psd + P_psd.T) / 2
        # TODO: when biases are added, we need to add the noise for the biases to the biases
        # self.P = (P + P.T) / 2
        self.x = State.from_vec(x_bar)
        self.propgated_sigmas = sigmas

    def update(self, measurement, dt: float, R=None):
        # self.P += np.eye(self.dim_x) * 1e-8

        R = measurement.R
        z = measurement.z

        xis = self.sigma_points.compute_sigma_points(self.P_aug)
        sigmas_z = np.zeros((self.sigma_points.num_sigmas, measurement.dim))
        for i, sigma in enumerate(self.propgated_sigmas):
            # x = State.add_error_state(self.x, sigma)
            x = State.from_vec(sigma)
            sigmas_z[i] = measurement.h(x)

        z_hat = np.dot(self.sigma_points.Wm, sigmas_z)

        # y = sigmas_z - z_hat[np.newaxis, :]
        Wc_diag = np.diag(self.sigma_points.Wc)
        # S = np.dot(y.T, np.dot(Wc_diag, y)) + measurement.R
        # S_inv = np.linalg.inv(S)
        # print(S_inv)
        Pxz = np.zeros((self.delta_sigmas.shape[1], sigmas_z.shape[1]))

        S = np.zeros((measurement.dim, measurement.dim))
        x = self.x.as_vec()
        sigmas = self.propgated_sigmas
        # print(self.sigma_points.Wc)
        for i in range(self.sigma_points.num_sigmas):
            dx = State.to_error_state(sigmas[i], x)
            dz = sigmas_z[i] - z_hat
            Pxz += self.sigma_points.Wc[i] * np.outer(dx, dz)
            S += self.sigma_points.Wc[i] * np.outer(dz, dz)

        S = S + measurement.R
        # dz = sigmas_z - z_hat[np.newaxis, :]
        # Pxz = (self.sigma_points.Wc * self.delta_sigmas) @ dz
        # S_inv = np.linalg.inv(S)
        self.S_inv = np.linalg.inv(S)
        # self.S_inv = np.linalg.inv(S)
        self.innovation = z - z_hat

        K = Pxz @ self.S_inv

        dx = K @ self.innovation
        self.x = State.add_error_state(self.x, dx)
        P = self.P - K @ S @ K.T
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals[eigvals < 0] = 0
        P_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.P = (P_psd + P_psd.T) / 2

    def nees_pos(self, true_pos):
        idx = 2
        x_hat = self.x.position[:idx]
        x = true_pos[:idx]
        return (x_hat - x).T @ np.linalg.inv(self.P[3:3 + idx, 3:3 + idx]) @ (x_hat - x)

    def nis(self):
        # Compute the NIS
        return self.innovation.T @ self.S_inv @ self.innovation

    def to_proto_msg(self):
        quat = self.x.q

        pos = self.x.position
        vel = self.x.R.T @ self.x.velocity
        rot = Rot.from_matrix(self.x.R)
        roll, pitch, yaw = rot.as_euler("xyz", degrees=True)

        return UkfState(
            position_x=pos[0],
            position_y=pos[1],
            position_z=pos[2],
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


