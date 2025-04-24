from dataclasses import dataclass, field
import numpy as np
from .state import LieState
from lie import SO3
import scipy
from abc import ABC, abstractmethod
from typing import Optional
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
        self.g = np.array((0, 0, 9.8))
        self.Q = scipy.linalg.block_diag(
            self.gyro_std**2 * np.eye(3),
            self.accel_std**2 * np.eye(3),
            self.gyro_bias_std**2 * np.eye(3),
            # 1e-5 * np.eye(3),
            # self.gyro_bias_std**2 * np.eye(3),
        )

    def f(self, state: LieState, u: np.ndarray, dt: float, w: np.ndarray):
        omega = u[:3] + w[:3] - state.gyro_bias 
        a_m = (-u[3:6] * 9.78) - state.acc_bias + w[3:6]
        R = state.extended_pose.rotation()
        Rt = state.extended_pose.rotation().T
        acc = a_m + Rt @ state.g

        theta = omega * dt
        d_vel = acc * dt
        d_pos = Rt @ state.extended_pose.linearVelocity() * dt + .5 * R @ acc * dt**2
        xi = np.concatenate([d_pos, theta, d_vel])
        tangent = manif.SE_2_3Tangent(xi)

        new_extended_pose = state.extended_pose.rplus(tangent)
        return LieState(extended_pose=new_extended_pose, gyro_bias=state.gyro_bias)

    def h(self, state: LieState):
        return state.R.T @ state.vel

    @classmethod
    def phi(cls, state, xi):
        """Takes points in the euclidean space onto the manifold (Exp map)"""
        tangent = manif.SE_2_3Tangent(xi[:9])
        new_extended_pose = state.extended_pose.rplus(tangent)
        # gyro_bias = state.gyro_bias + xi[9:12]
        # acc_bias = state.acc_bias + xi[12:15]
        return LieState(extended_pose=new_extended_pose)#, gyro_bias=gyro_bias)

    @classmethod
    def phi_inv(cls, state, state_hat):
        """Takes points from the manifold onto the Lie algebra (Log map)"""
        # tangent = state.extended_pose.rminus(state_hat.extended_pose).coeffs()
        tangent = state_hat.extended_pose.rminus(state.extended_pose).coeffs()
        return tangent
        # d_gyro_bias = state.gyro_bias - state_hat.gyro_bias
        # d_acc_bias = state.acc_bias - state_hat.acc_bias
        # return np.hstack([tangent, d_gyro_bias])

        # return np.hstack([tangent, dg])

class Measurement(ABC):
    def __init__(self, R):
        self.R = R

    @abstractmethod
    def h(self):
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

    @z.setter
    def z(self, val: np.ndarray):
        self._z = val

    def h(self, state: LieState) -> np.ndarray:
        R = state.extended_pose.rotation()
        # return R.T @ state.extended_pose.linearVelocity()
        return R.T @ state.extended_pose.linearVelocity()


class Magneotmeter(Measurement):
    def __init__(self, R, z: Optional[np.ndarray] = None):
        self.R = R
        if z is None:
            z = np.zeros(3)
        elif len(z) != 3:
            raise ValueError("Magnetometer measurement must have 3 elements")
        else:
            self._z = z

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, val: np.ndarray):
        self._z = val

    def h(self, state: LieState) -> np.ndarray:
        pass


class DepthMeasurement(Measurement):
    def __init__(self, R, z: Optional[float] = None):
        self.R = R
        self.dim = 1
        if z is not None:
            self._z = z

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, val):
        self._z = val

    def h(self, state: LieState) -> float:
        return state.extended_pose.translation()[-1]


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

    def h(self, state: LieState):
        return state.extended_pose.translation()[:2]


