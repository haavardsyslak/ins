import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Self
import manifpy as manif
from istate import IState
from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass


@dataclass
class State(IState):
    ori: manif.SO3
    vel: np.ndarray
    pos: np.ndarray
    gyro_bias: np.ndarray
    acc_bias: np.ndarray

    def dof(self) -> int:
        return 15

    @staticmethod
    def add_error_state(state: Self, delta_x) -> Self:
        tangent = manif.SO3Tangent(delta_x[:3])
        ori = state.ori.rplus(tangent)
        pos = state.pos + delta_x[3:6]
        vel = state.vel + delta_x[6:9]
        gyro_bias = state.gyro_bias + delta_x[9:12]
        acc_bias = state.acc_bias + delta_x[12:15]
        return State(ori=ori, pos=pos, vel=vel, gyro_bias=gyro_bias, acc_bias=acc_bias)

    @staticmethod
    def from_vec(x: np.ndarray) -> Self:
        ori = manif.SO3(x[:4])
        pos = x[4:7]
        vel = x[7:10]
        gyro_bias = x[10:13]
        acc_bias = x[13:16]
        return State(ori=ori, pos=pos, vel=vel, gyro_bias=gyro_bias, acc_bias=acc_bias)

    @staticmethod
    def to_error_state(state, state_hat):
        ori = manif.SO3(state[:4])
        ori_hat = manif.SO3(state_hat[:4])
        tangent = ori.rminus(ori_hat).coeffs()

        dx = state[4:] - state_hat[4:]
        return np.concatenate([tangent, dx])

    def as_vec(self) -> np.ndarray:
        return np.concatenate([self.ori.coeffs(), self.pos, self.vel, self.gyro_bias, self.acc_bias])

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
        return Rot.from_matrix(self.R).as_quat()
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
