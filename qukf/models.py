from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from .state import State
import scipy
from abc import ABC, abstractmethod
from typing import Optional
from utils import skew
import manifpy as manif

@dataclass
class ImuModel:
    """The IMU is considered a dynamic model instead of a sensar.
    This works as an IMU measures the change between two states,
    and not the state itself.."""

    accel_std: float
    accel_bias_std: float
    accel_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    accel_correction: 'np.ndarray[3, 3]' = field(default_factory=lambda: np.eye(3))
    gyro_correction: 'np.ndarray[3, 3]' = field(default_factory=lambda: np.eye(3))

    g: 'np.ndarray[3]' = field(default_factory=lambda: np.array([0, 0, 9.819]))

    Q: 'np.ndarray[12, 12]' = field(init=False, repr=False)

    def __post_init__(self):
        def diag3(x):
            return np.diag([x] * 3)

        accm_corr = self.accel_correction
        gyro_corr = self.gyro_correction

        self.Q = scipy.linalg.block_diag(
            accm_corr @ diag3(self.accel_std**2) @ accm_corr.T,
            gyro_corr @ diag3(self.gyro_std**2) @ gyro_corr.T,
            diag3(self.gyro_bias_std**2),
            diag3(self.accel_bias_std**2),
        )

    def f(self, state: State, u: np.ndarray, dt: float, w: np.ndarray):
        """Predict the nominal state, given an IMU mesurement """

        omega = u[:3] + w[:3] - state.gyro_bias
        a_m = (u[3:6] * 9.80665) + w[3:6] - state.acc_bias
        # a_m[:2] *= -1

        Rq = state.R
        # acc = a_m + Rq.T @ self.g
        acc = Rq @ a_m + self.g

        # omega[:2] = 0.0
        theta = omega * dt
        d_vel = acc * dt

        pos_pred = state.pos + (dt * state.vel) #+ 0.5 * dt**2 * acc
        vel_pred = state.vel + d_vel

        tangent = manif.SO3Tangent(theta)
        q = state.ori.rplus(tangent).coeffs()

        gyro_bias_pred = state.gyro_bias + w[9:12]
        acc_bias_pred = state.acc_bias + w[6:9]

        # delta_rot = AttitudeError.from_rodrigues_param(theta)
        # ori_pred = state.ori.multiply(delta_rot)
        
        # q = (Rot.from_matrix(Rq) * Rot.from_rotvec(theta)).as_quat()
        # ori_pred = RotationQuaterion(q[-1], q[:3])

        # acc_bias_pred = np.exp(-dt * self.accel_bias_p) * state.acc_bias + w[6:9]
        # gyro_bias_pred = np.exp(-dt * self.gyro_bias_p) * state.gyro_bias + w[9:12]
        return np.hstack([q, pos_pred, vel_pred, gyro_bias_pred, acc_bias_pred])
