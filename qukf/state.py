import numpy as np
from dataclasses import dataclass
from orientation import Orientation, RotationQuaterion, AttitudeError, RotationMatrix
import json
from scipy.spatial.transform import Rotation
from lie import SU2
from typing import Self, Union
from abc import ABC, abstractmethod


@dataclass
class LieState:
    ori: RotationMatrix
    vel: np.ndarray
    pos: np.ndarray
    # g: float

    def to_json(self) -> str:
        q = Rotation.from_matrix(self.ori.R).as_quat()
        euler = Rotation.from_matrix(self.ori.R).as_euler("XYZ")

        msg = {
            "velocity": {"x": self.vel[0], "y": self.vel[1], "z": self.vel[2]},
            "pose": {
                "position": {"x": self.pos[0], "y": self.pos[1], "z": self.pos[2]},
                "orientation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
            },
            "euler_angles": {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]},
            # "gyro_bias": {"x": self.gyro_bias[0], "y": self.gyro_bias[1], "z": self.gyro_bias[2]},
            # "acc_bias": {"x": self.acc_bias[0], "y": self.acc_bias[1], "z": self.acc_bias[2]},
            # "gravity": {"g": self.g},
        }

        return json.dumps(msg)


@dataclass
class State:
    ori: RotationQuaterion
    vel: np.ndarray
    pos: np.ndarray

    def as_vec(self):
        q = self.ori.as_vec()
        return np.concatenate([q, self.vel, self.pos])

    # TODO currently not arrays
    @staticmethod
    def add_error_state(state: np.ndarray, error_state: np.ndarray) -> np.ndarray:
        # q = SU2.Exp(error_state[:3], 1)
        q = Rotation.from_quat(state.ori.as_vec(), scalar_first=True) * Rotation.from_rotvec(error_state[:3])
        # q = state.ori @ q
        q = RotationQuaterion.from_vec(q.as_quat(scalar_first=True))
        v = state.vel + error_state[3:6]
        p = state.pos + error_state[6:9]
        return State(ori=q, vel=v, pos=p)

    @classmethod
    def from_vec(cls, state: np.ndarray):
        q = RotationQuaterion.from_vec(state[:4])
        v = state[4:7]
        p = state[7:10]

        return cls(ori=q, vel=v, pos=p)

    def update_from_vec(self, state):
        self.ori = RotationQuaterion.from_vec(state[:4])
        self.vel = state[4:7]
        self.p = state[7:10]

    @staticmethod
    def to_error_state(x, x_bar) -> np.ndarray:
        # print("x: ", x_bar[:4])
        # dq = RotationQuaterion.get_error_quat(x[:4], x_bar[:4])
        # dq = RotationQuaterion.from_vec(x[:4]) @ RotationQuaterion.from_vec(x_bar[:4]).conj()

        # print("dq:", dq)

        # dq = RotationQuaterion.get_error_quat(x[:4], x_bar[:4])
        # dq = x_bar.ori @ self.ori.conj()
        # dr = SU2.log(dq.as_vec())
        # dr = SU2.log(dq)
        # dr = Rotation.from_quat(dq, scalar_first=True).as_rotvec()
        q = Rotation.from_quat(x[:4], scalar_first=True)
        q_bar = Rotation.from_quat(x_bar[:4], scalar_first=True)
        dr = (q_bar.inv() * q).as_rotvec()
        dv = x_bar[4:7] - x[4:7]
        dp = x_bar[7:10] - x[7:10]

        return np.concatenate([dr, dv, dp])

    def to_json(self) -> str:
        q = self.ori.as_vec()
        euler = Rotation.from_quat(q, scalar_first=True).as_euler("XYZ")

        msg = {
            "velocity": {"x": self.vel[0], "y": self.vel[1], "z": self.vel[2]},
            "pose": {
                "position": {"x": self.pos[0], "y": self.pos[1], "z": self.pos[2]},
                "orientation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
            },
            "euler_angles": {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]},
            # "gyro_bias": {"x": self.gyro_bias[0], "y": self.gyro_bias[1], "z": self.gyro_bias[2]},
            # "acc_bias": {"x": self.acc_bias[0], "y": self.acc_bias[1], "z": self.acc_bias[2]},
            # "gravity": {"g": self.g},
        }

        return json.dumps(msg)


@dataclass
class AugmentedState:
    ori: RotationQuaterion
    vel: np.ndarray
    pos: np.ndarray
    noise: np.ndarray

    def as_vec(self) -> np.ndarray:
        q = self.ori.as_vec()
        return np.hstack([q, self.vel, self.pos, self.noise])

    @staticmethod
    def add_error_state(error_state) -> np.ndarray:
        q = SU2.exp(error_state[:3])
        RotationQuaterion.from_vec(q)

