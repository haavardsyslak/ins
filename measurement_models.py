from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from istate import IState
from utils import skew


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

        self._nis = None
        # Vector form the IMU frame to the DVL frame
        self.lever_arm = np.array([-17e-2, 0, 26e-2])
        self._lever_arm_comp = np.zeros(3)

    @property
    def lever_arm_comp(self):
        return self._lever_arm_comp

    @lever_arm_comp.setter
    def lever_arm_comp(self, value):
        self._lever_arm_comp = value

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, val: np.ndarray):
        self._z = val

    def h(self, state: IState, omega=np.zeros(3)) -> np.ndarray:
        # return state.extended_pose.linearVelocity()
        # return state.R.T @ state.velocity - self.lever_arm_comp
        return state.R.T @ state.velocity - self.lever_arm_comp

    def H(self, state: IState):
        """
        Compute measurement Jacobian H for h(x) = R(q)^T v.
        Assumes scalar-first quaternion q = [qw, qx, qy, qz].
        Returns: H ∈ ℝ^{3×19}
        """
        H_out = np.zeros((3, 16))

        # Extract quaternion and velocity from state
        q = state.q  # [qw, qx, qy, qz]
        v = state.velocity  # [vx, vy, vz]
        vx, vy, vz = state.velocity  # shape (3,)

        e1, e2, e3, n = q
        # qw, qx, qy, qz = q

        # Rotation matrix R(q)
        # R = np.array([
        #     [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        #     [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        #     [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        # ])

        # ∂h/∂v = R^T
        H_out[:, 3:6] = state.R.T

        # # Unit vectors
        # epsilon = np.array([qx, qy, qz])
        # e1 = np.array([1, 0, 0])
        # e2 = np.array([0, 1, 0])
        # e3 = np.array([0, 0, 1])

        # ∂h/∂q (analytical)
        H_q = np.zeros((3, 4))
        # H_q[:, 0] = (4 * qw * np.eye(3) + 2 * skew(epsilon)) @ v
        # H_q[:, 1] = 2 * ((np.outer(e1, epsilon) + np.outer(epsilon, e1) + qw * skew(e1)) @ v)
        # H_q[:, 2] = 2 * ((np.outer(e2, epsilon) + np.outer(epsilon, e2) + qw * skew(e2)) @ v)
        # H_q[:, 3] = 2 * ((np.outer(e3, epsilon) + np.outer(epsilon, e3) + qw * skew(e3)) @ v)

        # H_q = np.hstack([dR_dqw, dR_dqx, dR_dqy, dR_dqz])

        H_q = np.array(
            [
                [
                    -2 * e2 * vz + 2 * e3 * vy,
                    2 * e2 * vy + 2 * e3 * vz,
                    2 * e1 * vy - 4 * e2 * vx - 2 * n * vz,
                    2 * e1 * vz - 4 * e3 * vx + 2 * n * vy,
                ],
                [
                    2 * e1 * vz - 2 * e3 * vx,
                    -4 * e1 * vy + 2 * e2 * vx + 2 * n * vz,
                    2 * e1 * vx + 2 * e3 * vz,
                    2 * e2 * vz - 4 * e3 * vy - 2 * n * vx,
                ],
                [
                    -2 * e1 * vy + 2 * e2 * vx,
                    -4 * e1 * vz + 2 * e3 * vx - 2 * n * vy,
                    -4 * e2 * vz + 2 * e3 * vy + 2 * n * vx,
                    2 * e1 * vx + 2 * e2 * vy,
                ],
            ]
        )
        H_out[:, 6:10] = H_q
        # H_out[:, 3:6] = np.eye(3)

        return H_out

    @property
    def nis(self):
        return self._nis

    @nis.setter
    def nis(self, nis):
        self._nis = nis


class Magnetometer(Measurement):
    def __init__(self, R, z: Optional[np.ndarray] = None):
        self.R = R
        self.dim = 3
        if z is None:
            self._z = np.zeros(3)
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

    def h(self, state: IState) -> np.ndarray:
        return state.R.T @ state.get_mag_field()


class DepthMeasurement(Measurement):
    def __init__(self, R, z: Optional[float] = None):
        self.R = R
        self.dim = 1
        if z is not None:
            self._z = z
        self._nis = None

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, val):
        self._z = val

    def h(self, state: IState) -> float:
        return state.position[2]

    @property
    def nis(self):
        return self._nis

    @nis.setter
    def nis(self, nis):
        self._nis = nis

    def H(self, state: IState) -> np.ndarray:
        H = np.zeros((1, 16))
        H[0, 2] = 1
        return H


class GnssMeasurement(Measurement):
    # GNSS measurement needs to be given in the local frame (x, y)
    def __init__(self, R, z: Optional[np.ndarray] = None):
        self.R = R
        self.dim = 2
        if z is not None:
            self._z = z
        self._nis = None

        self.lever_arm = np.array([-35e-2, 0, -5e-2])

    @property
    def z(self) -> np.ndarray:
        return self._z

    @z.setter
    def z(self, val: np.ndarray):
        self._z = val

    def h(self, state: IState):
        return state.position[:2] + self.lever_arm[:2]

    def H(self, state: IState):
        H = np.zeros((2, 16))
        H[:, :2] = np.eye(2)
        return H

    @property
    def nis(self):
        return self._nis

    @nis.setter
    def nis(self, nis):
        self._nis = nis
