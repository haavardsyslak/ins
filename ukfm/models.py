from dataclasses import dataclass, field
import numpy as np
from .state import LieState
import scipy
import manifpy as manif


@dataclass
class ImuModel:
    """The IMU is considered a dynamic model instead of a sensar.
    This works as an IMU measures the change between two states,
    and not the state itself.."""

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    accel_std: float
    accel_bias_std: float
    accel_bias_p: float

    Q: "np.ndarray[6, 6]" = field(init=False, repr=False)

    def __post_init__(self):
        self.g = np.array((0, 0, 9.819))
        self.Q = scipy.linalg.block_diag(
            self.gyro_std**2 * np.eye(3),
            self.accel_std**2 * np.eye(3),
            self.gyro_bias_std**2 * np.eye(3),
            self.accel_bias_std**2 * np.eye(3),
        )

    def f(self, state: LieState, u: np.ndarray, dt: float, w: np.ndarray):
        omega = u[:3] + w[:3] - state.gyro_bias
        # print(u[:3])
        a_m = (u[3:6] * 9.80665) + w[3:6]- state.acc_bias
        # print(a_m)
        R = state.extended_pose.rotation()
        acc = a_m + R.T @ state.g
        theta = omega * dt
        d_vel = acc * dt
        d_pos = R.T @ state.extended_pose.linearVelocity() * dt # * dt + .5 * acc * dt**2
        xi = np.concatenate([d_pos, theta, d_vel])
        tangent = manif.SE_2_3Tangent(xi)

        new_extended_pose = state.extended_pose.rplus(tangent)

        gyro_bias = state.gyro_bias + w[6:9]
        acc_bias = state.acc_bias + w[9:12]

        # gyro_bias = np.exp(-dt * self.gyro_bias_p) * state.gyro_bias + w[6:9]
        # acc_bias = np.exp(-dt * self.accel_bias_p) * state.acc_bias + w[9:12]

        # acc_bias = np.zeros(3)
        # gyro_bias[1] = 0
        return LieState(extended_pose=new_extended_pose, gyro_bias=gyro_bias, acc_bias=acc_bias)

    def h(self, state: LieState):
        return state.R.T @ state.vel

    @classmethod
    def phi_up(cls, state, xi):
        # return cls.phi(state, xi)
        tangent = manif.SE_2_3Tangent(xi[:9])
        new_extended_pose = tangent + state.extended_pose
        gyro_bias = state.gyro_bias + xi[9:12]
        acc_bias = state.acc_bias + xi[12:15]
        return LieState(extended_pose=new_extended_pose, gyro_bias=gyro_bias, acc_bias=acc_bias)

    @classmethod
    def phi(cls, state, xi):
        """Takes points in the euclidean space onto the manifold (Exp map)"""
        tangent = manif.SE_2_3Tangent(xi[:9])
        # new_extended_pose = tangent + state.extended_pose
        new_extended_pose = state.extended_pose.lplus(tangent)

        gyro_bias = state.gyro_bias + xi[9:12]
        acc_bias = state.acc_bias + xi[12:15]
        return LieState(extended_pose=new_extended_pose, gyro_bias=gyro_bias, acc_bias=acc_bias)

    @classmethod
    def phi_inv(cls, state, state_hat):
        """Takes points from the manifold onto the Lie algebra (Log map)"""
        # x_new.lminus(propagated)
        tangent = state.extended_pose.lminus(state_hat.extended_pose).coeffs()
        # tangent = state_hat.extended_pose.lminus(state.extended_pose).coeffs()
        d_gyro_bias = state.gyro_bias - state_hat.gyro_bias
        d_acc_bias = state.acc_bias - state_hat.acc_bias

        return np.concatenate([tangent, d_gyro_bias, d_acc_bias])


