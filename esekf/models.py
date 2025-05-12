import numpy as np
import scipy
from dataclasses import dataclass, field
from .state import NominalState
from orientation import RotationQuaterion, AttitudeError
from utils import skew
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as Rot
import manifpy as manif


@dataclass
class ImuMeasurement:
    gyro: np.ndarray
    acc: np.ndarray


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

    Q_c: 'np.ndarray[12, 12]' = field(init=False, repr=False)

    def __post_init__(self):
        def diag3(x):
            return np.diag([x] * 3)

        accm_corr = self.accel_correction
        gyro_corr = self.gyro_correction

        self.Q_c = scipy.linalg.block_diag(
            accm_corr @ diag3(self.accel_std**2) @ accm_corr.T,
            gyro_corr @ diag3(self.gyro_std**2) @ gyro_corr.T,
            diag3(self.accel_bias_std**2),
            diag3(self.gyro_bias_std**2)
        )

    def predict_nominal(self, state: NominalState, u: np.ndarray, dt: float) -> NominalState:
        """Predict the nominal state, given an IMU mesurement

        We assume the change in orientation is negligable when caculating
        predicted position and velicity
        """
        omega = u[:3] #- state.gyro_bias
        a_m = (u[3:6] * 9.80665)# - state.acc_bias
        # a_m[:2] *= -1

        Rq = state.R
        # acc = a_m + Rq.T @ self.g
        acc = Rq @ a_m + self.g

        # omega[:2] = 0.0
        theta = omega * dt
        d_vel = acc * dt

        pos_pred = state.pos + (dt * state.vel) #+ 0.5 * dt**2 * acc
        vel_pred = state.vel + d_vel * dt

        tangent = manif.SO3Tangent(theta)
        ori_pred = state.ori.rplus(tangent)

        # delta_rot = AttitudeError.from_rodrigues_param(theta)
        # ori_pred = state.ori.multiply(delta_rot)
        
        # q = (Rot.from_matrix(Rq) * Rot.from_rotvec(theta)).as_quat()
        # ori_pred = RotationQuaterion(q[-1], q[:3])

        acc_bias_pred = np.exp(-dt * self.accel_bias_p) * state.acc_bias
        gyro_bias_pred = np.exp(-dt * self.gyro_bias_p) * state.gyro_bias

        x_pred = NominalState(ori=ori_pred, pos=pos_pred, vel=vel_pred, gyro_bias=gyro_bias_pred,
                              acc_bias=acc_bias_pred)
        return x_pred

    def F_c(self,
            x_est_nom: NominalState,
            u: ImuMeasurement,
            ) -> 'np.ndarray[15, 15]':
        """
        Get the continuous time state transition matrix, A_c
        """
        F_c = np.zeros((15, 15))
        Rq = x_est_nom.R
        omega = u[:3]
        acc = u[3:]
        S_acc = skew(acc - x_est_nom.acc_bias)
        S_omega = skew(omega - x_est_nom.gyro_bias)
        pa = self.accel_bias_p
        pw = self.gyro_bias_p

        def O(x): return np.zeros((x, x))
        def I(x): return np.eye(x)
        F_c = np.block([[O(3), I(3), O(3), O(3), O(3)],
                        [O(3), O(3), -Rq @ S_acc, -Rq @ self.accel_correction, O(3)],
                        [O(3), O(3), -S_omega, O(3), -self.gyro_correction],
                        [O(3), O(3), O(3), -pa * I(3), O(3)],
                        [O(3), O(3), O(3), O(3), -pw * I(3)]])

        return F_c

    def get_error_G_c(self,
                      state: NominalState,
                      ) -> 'np.ndarray[15, 15]':
        """The continous noise covariance matrix, G, in (10.68)

        """
        G_c = np.zeros((15, 12))
        Rq = state.R

        O3 = np.zeros((3, 3))
        I3 = np.eye(3)

        G_c = np.block([[O3, O3, O3, O3],
                        [-Rq.T, O3, O3, O3],
                        [O3, -I3, O3, O3],
                        [O3, O3, I3, O3],
                        [O3, O3, O3, I3]])

        return G_c

    def get_discrete_error_diff(self,
                                state: NominalState,
                                u: ImuMeasurement,
                                dt: float
                                ) -> Tuple['np.ndarray[15, 15]',
                                           'np.ndarray[15, 15]']:
        """Get the discrete equivalents of F and GQGT

        We use scipy.linalg.expm to get the matrix exponential

        Then the descrete time matrices are extraxted using van loans method

        """
        F_c = self.F_c(state, u)
        G_c = self.get_error_G_c(state)
        GQGT_c = G_c @ self.Q_c @ G_c.T

        VanLoanMatrix = scipy.linalg.expm(np.block([[-F_c, GQGT_c],
                                                    [np.zeros((15, 15)), F_c.T]]) * dt)

        A_d = VanLoanMatrix[15:, 15:].T
        GQGT_d = A_d @ VanLoanMatrix[:15, 15:]

        return A_d, GQGT_d

    def predict_err(self,
                    state: NominalState,
                    P: np.ndarray,
                    u: ImuMeasurement,
                    dt: float,
                    ):
        """
        Predict the error state.
        Since the error state is initialized to zero, the linear equations awlays returns to zero,
        predicting the error state is therefore skipped
        """

        Ad, GQGTd = self.get_discrete_error_diff(state, u, dt)
        # x_err = Ad @ x_err_prev
        P_pred = Ad @ P @ Ad.T + GQGTd
        return P_pred


