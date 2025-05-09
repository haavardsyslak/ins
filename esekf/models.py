import numpy as np
import scipy
from dataclasses import dataclass, field
from .state import NominalState
from orientation import RotationQuaterion, AttitudeError
from utils import skew
from typing import Tuple, Optional
from abc import ABC, abstractmethod


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

    g: 'np.ndarray[3]' = field(default_factory=lambda: np.array([0, 0, 9.82]))

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
        omega = u[:3] - state.gyro_bias
        a_m = (u[3:6] * 9.78) - state.acc_bias
        # a_m[:2] *= -1
        Rq = state.ori.R
        acc = a_m - Rq.T @ self.g
        # print(acc)

        theta = omega * dt
        d_vel = acc * dt

        pos_pred = state.pos + Rq @ (dt * state.vel)  # + 0.5 * dt**2 * acc
        vel_pred = state.vel + d_vel

        delta_rot = AttitudeError.from_rodrigues_param(theta)
        ori_pred = state.ori.multiply(delta_rot)

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
        Rq = x_est_nom.ori.R
        omega = u[:3]
        acc = u[3:]
        S_acc = skew(acc)
        S_omega = skew(omega)
        pa = self.accel_bias_p
        pw = self.gyro_bias_p

        O3 = np.zeros((3, 3))
        I3 = np.eye(3)
        F_c = np.block([[O3, I3, O3, O3, O3],
                        [O3, O3, -S_omega, O3, -self.gyro_correction],
                        [O3, O3, -Rq @ S_acc, -Rq @ self.accel_correction, O3],
                        [O3, O3, O3, -pa * I3, O3],
                        [O3, O3, O3, O3, -pw * I3]])
        return F_c

    def get_error_G_c(self,
                      state: NominalState,
                      ) -> 'np.ndarray[15, 15]':
        """The continous noise covariance matrix, G, in (10.68)

        """
        G_c = np.zeros((15, 12))
        Rq = state.ori.R

        O3 = np.zeros((3, 3))
        I3 = np.eye(3)

        G_c = np.block([[O3, O3, O3, O3],
                        [-Rq, O3, O3, O3],
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


class Measurement(ABC):
    def __init__(self, R, z: Optional[np.ndarray] = None):
        self.R = R

    @abstractmethod
    def h(self):
        """
        Measurement function
        """
        pass

    @abstractmethod
    def H(self) -> np.ndarray:
        """
        Measurement jacobian
        """
        pass


class GNSSMeasurement(Measurement):
    def H(self):
        pass


class DvlMeasurement(Measurement):
    def __init__(self, R, z: Optional[np.ndarray] = None):
        self.R = R
        self.dim = 3
        if z is None:
            z = np.zeros(3)
        elif len(z) != 3:
            raise ValueError("Dvl measurement must have 3 elements")
        else:
            self._z = z

    @property
    def z(self):
        return self._z

    def H(self, state: NominalState):

        # n, e1, e2, e3 = state.ori.as_vec()
        # v_x, v_y, v_z = state.vel
        #
        # # Jacobian wrt. error state (chain rule)
        # X_dq = np.zeros((3, 15))
        # Q_delta_theta = 0.5 * np.array([
        #     [-e1, -e2, -e2],
        #     [n, -e2, e2],
        #     [e2, n, -e1],
        #     [-e2, e1, n],
        # ])
        # X_dq[:6, :6] = np.eye(6)
        # X_dq[6:9, 6:9] = Q_delta_theta
        # # Initialize the Jacobian matrices
        #
        # # fmt: off
        v_x, v_y, v_z = state.vel
        n, e1, e2, e3 = state.ori.as_vec()
        H_q = np.array([
            [-2*e2*v_z + 2*e3*v_y, 2*e2*v_y + 2*e3*v_z, 2*e1*v_y - 4*e2*v_x - 2*n*v_z, 2*e1*v_z - 4*e3*v_x + 2*n*v_y],
            [2*e1*v_z - 2*e3*v_x, -4*e1*v_y + 2*e2*v_x + 2*n*v_z, 2*e1*v_x + 2*e3*v_z, 2*e2*v_z - 4*e3*v_y - 2*n*v_x],
            [-2*e1*v_y + 2*e2*v_x, -4*e1*v_z + 2*e3*v_x - 2*n*v_y, -4*e2*v_z + 2*e3*v_y + 2*n*v_x, 2*e1*v_x + 2*e2*v_y]
                ])
        # #
        # #
        H_v = np.array([
            [-2*e2**2 - 2*e3**2 + 1, 2*e1*e2 + 2*e3*n, 2*e1*e3 - 2*e2*n],
            [2*e1*e2 - 2*e3*n, -2*e1**2 - 2*e3**2 + 1, 2*e1*n + 2*e2*e3], 
            [2*e1*e3 + 2*e2*n, -2*e1*n + 2*e2*e3, -2*e1**2 - 2*e2**2 + 1]
                ])
        # H_v = state.ori.R.T
        q0, q1, q2, q3 = state.ori.as_vec()

    # ∂h/∂q (3x4)
        # H_q = 2 * np.array([
        #     [-q1*vx - q2*vy - q3*vz,  q0*vx + q2*vz - q3*vy,  q0*vy - q1*vz + q3*vx,  q0*vz + q1*vy - q2*vx],
        #     [-q2*vz + q3*vy + q1*vx,  q2*vx - q0*vz - q3*vy,  q3*vx + q0*vy - q1*vz, -q2*vy + q1*vx + q0*vz],
        #     [-q3*vx + q1*vz - q2*vy,  q3*vx + q0*vy - q1*vz,  q1*vx + q3*vz - q0*vy,  q2*vx + q0*vz - q3*vy]
        #     ])
        # # fmt: on
        #
        # # Assemble the final Jacobian H_x
        H = np.zeros((3, 16))  # 6x6 matrix for H_x
        # H[:, 3:6] = H_v        # velocity
        # H[:, 6:10] = H_q       # quaternion
        # H_x[0:4, :] = H_q
        # H_x[8:10, :] = H_v
        #
        # R = state.ori.R  # Rotation matrix from world to body
        # v = state.vel   # Velocity in world frame

        # Skew-symmetric matrix of velocity

        # H = np.zeros((3, 16))  # 3 outputs (body frame velocities) vs 15 error states

        # Partial derivative wrt orientation error (delta theta)
        H[:, 3:6] = np.eye(3)

        return H

    def h(self, state: NominalState):
        # R = state.ori.R
        # return R.T @ state.vel
        return state.vel


class DepthMeasurement(Measurement):
    def __init__(self, R, z: Optional[float] = None):
        self.R = np.atleast_1d(R).reshape(-1, 1)
        self.dim = 1
        if z is not None:
            self._z = z

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, val):
        self._z = val

    def h(self, state: NominalState) -> float:
        return state.pos[-1]

    def H(self, state: NominalState) -> np.ndarray:
        H = np.zeros((1, 16))
        H[0, 2] = 1  # Only sensitive to z-position error
        return H


class GnssMeasurement(Measurement):
    # GNSS measurement needs to be given in the local frame (x, y)
    def __init__(self, R, z: Optional[np.ndarray] = None):
        self.R = R
        self.dim = 2
        if z is not None:
            self._z = z

    @property
    def z(self) -> np.ndarray:
        return self._z

    @z.setter
    def z(self, val: np.ndarray):
        self._z = val

    def h(self, state: NominalState):
        return state.extended_pose.translation()[:2]
