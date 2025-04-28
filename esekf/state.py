from dataclasses import dataclass
from orientation import RotationQuaterion
import numpy as np
from typing import Self
import pymap3d as pm


@dataclass
class NominalState:
    ori: RotationQuaterion
    vel: np.ndarray
    pos: np.ndarray
    acc_bias: np.ndarray
    gyro_bias: np.ndarray
    # g: np.ndarray

    def dof(self):
        return 15

    def to_global_position(self, initial_global_pos):
        lat0 = initial_global_pos[0]
        long0 = initial_global_pos[1]
        alt0 = 0.0
        if len(initial_global_pos) == 3:
            alt0 = initial_global_pos[2]

        pos = self.pos
        lat, lon, alt = pm.ned2geodetic(pos[0], pos[1], pos[2], lat0, long0, alt0)

        return lat, lon, alt


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
            theta=vec[:3],
            pos=vec[3:6],
            vel=vec[6:9],
            gyro_bias=vec[9:12],
            acc_bias=vec[12:15],
            # g=vec[15:17],
        )
