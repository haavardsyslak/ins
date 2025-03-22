from dataclasses import dataclass
from orientation import RotationQuaterion
import numpy as np
from typing import Self


@dataclass
class NominalState:
    ori: RotationQuaterion
    vel: np.ndarray
    pos: np.ndarray
    acc_bias: np.ndarray
    gyro_bias: np.ndarray
    g: np.ndarray


@dataclass
class ErrorState:
    theta: np.ndarray
    vel: np.ndarray
    pos: np.ndarray
    gyro_bias: np.ndarray
    gyro_bias: np.ndarray
    acc_bias: np.ndarray
    g: np.ndarray

    @staticmethod
    def from_vec(vec: np.ndarray) -> Self:
        return ErrorState(
            theta=vec[:3],
            pos=vec[3:6],
            vel=vec[6:9],
            gyro_bias=vec[9:12],
            acc_bias=vec[12:15],
            g=vec[15:17],
        )
