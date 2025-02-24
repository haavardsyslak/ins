from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from state import LieState
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

    acc_std: float
    acc_bias_std: float
    acc_bias_p: float

    Q: "np.ndarray[6, 6]" = field(init=False, repr=False)

    def __post_init__(self):
        self.g = np.array((0, 0, 9.822))
        self.Q = scipy.linalg.block_diag(
            self.gyro_std**2 * np.eye(3),
            self.acc_std**2 * np.eye(3),
            # np.array([0, 0, 0.00001]),
        )

    def f(self, state: LieState, u: np.ndarray, dt: float, w: np.ndarray):
        omega = u[:3] + w[:3]
        acc = (u[3:6] * 9.81) #+ w[3:6]
        acc *=-1
        acc += w[3:6]

        
        R = state.state.rotation()
        # acc_b = acc + (R.T @ self.g)
        # acc_n = R @ (acc_b)
        acc_b = acc + R.T @ np.array([0, 0, state.g])
        # acc_n = (R @ acc) - self.g
        # acc_b = R.T @ acc
        # xi = np.concatenate(
        #     [omega * dt, acc_n * dt, state.state.linearVelocity() + .5 * acc_n * dt**2])
        d_vel = acc_b * dt
        d_pos = state.state.linearVelocity() * dt + .5 * acc_b * dt**2
        d_rot = omega * dt
        xi = np.concatenate(
            [R @ d_pos, d_rot, d_vel])
        xi_hat = manif.SE_2_3Tangent(xi)

        new_state = state.state.rplus(xi_hat)

        # R = state.R @ SO3.exp(omega * dt)
        # p = state.pos + state.vel * dt + 0.5 * ((state.R @ acc) - self.g) * dt**2
        # v = state.vel + ((state.R @ acc) - self.g) * dt
        # return LieState(R=R, vel=v, pos=p)
        return LieState(new_state, g=state.g)

    def h(self, state: LieState):
        return state.R.T @ state.vel

    @classmethod
    def phi(cls, state, xi):
        """Takes points in the euclidean space onto the manifold (Exp map)"""
        lie_xi = manif.SE_2_3Tangent(xi[:9])
        dg = state.g + xi[-1]
        return LieState(state.state.rplus(lie_xi), g=dg)

    @classmethod
    def phi_inv(cls, state, state_hat):
        """Takes points from the manifold onto the Lie algebra (Log map)"""
        lie_state = state.state.rminus(state_hat.state).coeffs()
        dg = state.g - state_hat.g

        return np.concatenate([lie_state, np.array([dg])])

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
        # return state.state.rotation().transpose() @ state.state.linearVelocity()
        # return state.state.rotation().transpose() @ state.state.linearVelocity()
        return state.state.linearVelocity()


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
        return state.state.z()
