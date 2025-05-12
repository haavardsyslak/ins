from dataclasses import dataclass
from orientation import RotationQuaterion
import numpy as np
from typing import Self
import pymap3d as pm
from istate import IState
import manifpy as manif
from scipy.spatial.transform import Rotation as Rot


@dataclass
class NominalState(IState):
    ori: manif.SO3
    vel: np.ndarray
    pos: np.ndarray
    acc_bias: np.ndarray
    gyro_bias: np.ndarray
    # g: np.ndarray

    def dof(self):
        return 15

    @property
    def position(self) -> np.ndarray:
        return self.pos

    @property
    def velocity(self) -> np.ndarray:
        return self.vel

    @property
    def R(self) -> np.ndarray:
        return self.ori.rotation()

    @property
    def q(self) -> np.ndarray:
        # print(self.extended_pose.coeffs()[3:7])
        # input()
        return self.ori.coeffs()
        # return self.extended_pose.coeffs()[3:7]

    @property
    def euler(self) -> np.ndarray:
        return Rot.from_matrix(self.R).as_euler("xyz", degrees=True)

    @property
    def gyroscope_bias(self) -> np.ndarray:
        return self.gyro_bias

    @property
    def accelerometer_bias(self) -> np.ndarray:
        return self.acc_bias


@dataclass
class ErrorState:
    theta: np.ndarray
    vel: np.ndarray
    pos: np.ndarray
    gyro_bias: np.ndarray
    gyro_bias: np.ndarray
    acc_bias: np.ndarray
    # g: np.ndarray

    @staticmethod
    def from_vec(vec: np.ndarray) -> Self:
        return ErrorState(
            pos=vec[:3],
            vel=vec[3:6],
            theta=vec[6:9],
            acc_bias=vec[9:12],
            gyro_bias=vec[12:15],
            # g=vec[15:17],
        )
