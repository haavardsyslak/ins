from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from istate import IState

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
        return state.R.T @ state.velocity - self.lever_arm_comp

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
        return state.position[:2]# + self.lever_arm[:2]

    @property
    def nis(self):
        return self._nis

    @nis.setter
    def nis(self, nis):
        self._nis = nis


