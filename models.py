from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from state import LieState
from lie import SO3
import scipy
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class ImuModel:
    """The IMU is considered a dynamic model instead of a sensar.
    This works as an IMU measures the change between two states,
    and not the state itself.."""

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    acc_std: float
    acc_bias_std: float
    acc_bias_p: float

    Q: "np.ndarray[6, 6]" = field(init=False, repr=False)

    def __post_init__(self):
        self.g = np.array((0, 0, 9.822))
        self.Q = scipy.linalg.block_diag(
            self.gyro_std**2 * np.eye(3),
            self.acc_std**2 * np.eye(3),
            # self.gyro_bias_std**2 * np.eye(3),
        )

    def f(self, state: LieState, u: np.ndarray, dt: float, w: np.ndarray):
        omega = u[:3] + w[:3]
        acc = (u[3:6] * 9.81) + w[3:6]
        R = state.R @ SO3.exp(omega * dt)
        p = state.pos + state.vel * dt + 0.5 * ((state.R @ acc) - self.g) * dt**2
        v = state.vel + ((state.R @ acc) - self.g) * dt
        return LieState(R=R, vel=v, pos=p)

    def h(self, state: LieState):
        return state.R.T @ state.vel

    @classmethod
    def phi(cls, state, xi):
        """Takes points in the euclidean space onto the manifold (Exp map)"""
        R = SO3.exp(xi[:3]) @ state.R
        v = state.vel + xi[3:6]
        p = state.pos + xi[6:9]
        # g = state.g + xi[-1]
        return LieState(R=R, vel=v, pos=p)

    @classmethod
    def phi_inv(cls, state, state_hat):
        """Takes points from the manifold onto the Lie algebra (Log map)"""
        rot = SO3.log(state_hat.R.dot(state.R.T))
        v = state_hat.vel - state.vel
        p = state_hat.pos - state.pos

        return np.hstack([rot, v, p])

    def left_phi(cls, state, xi):
        delta_rot = SO3.exp(xi[:3])
        J = SO3.left_jacobian(xi[:3])
        R = state.R.dot(delta_rot)
        v = state.R.dot(J.dot(xi[3:6])) + state.vel
        p = state.R.dot(J.dot(xi[6:9])) + state.pos

        return LieState(R=R, vel=v, pos=p)

    def left_phi_inv(cls, state, state_hat):
        dR = state.R.T @ state_hat.R
        phi = SO3.log(dR)
        J = SO3.left_jacobian_inv(phi)
        dv = state.R.T.dot(state.vel - state_hat.vel)
        dp = state.R.T.dot(state.pos - state_hat.pos)

        return np.hstack([phi, J.dot(dv), J.dot(dp)])


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
        return state.R.T @ state.vel


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
        return state.pos[-1]
