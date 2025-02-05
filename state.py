import numpy as np
from dataclasses import dataclass
from quaternion import RotationQuaterion, AttitudeError
import json

from scipy.spatial.transform import Rotation


@dataclass
class ErrorState:
    pos: np.ndarray
    vel: np.ndarray
    ori: np.ndarray

    def as_vec(self):
        return np.hstack(self.pos, self.vel, self.ori)

    @staticmethod
    def from_vec(vec):
        pos = vec[0:3]
        vel = vec[3:6]
        ori = vec[6:9]
        return ErrorState(pos, vel, ori)


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    ori: RotationQuaterion

    def as_vec(self) -> np.ndarray:
        return np.hstack(self.pos, self.vel, self.ori.as_vec())

    @staticmethod
    def from_vec(vec: np.ndarray):
        pos = vec[0:3]
        vel = vec[3:6]
        ori = RotationQuaterion.from_vec(vec[6:10])

        return State(pos, vel, ori)

    def add_error_state(self, error_state: ErrorState):
        self.pos + error_state.pos
        self.vel + error_state.vel
        self.ori @ AttitudeError.from_rodrigues_param(error_state.ori)

    def split(self):
        euclidean = np.hstack(self.pos, self.vel)
        return (euclidean, self.ori.as_vec())

    def dim_euclidean(self):
        return 9

    def dim_non_euclidean(self):
        return 4


@dataclass
class LieState:
    R: np.ndarray
    pos: np.ndarray
    vel: np.ndarray

    def to_json(self) -> str:
        q = Rotation.from_matrix(self.R).as_quat()
        euler = Rotation.from_matrix(self.R).as_euler("XYZ")

        msg = {
            "velocity": {"x": self.vel[0], "y": self.vel[1], "z": self.vel[2]},
            "pose": {
                "position": {"x": self.pos[0], "y": self.pos[1], "z": self.pos[2]},
                "orientation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
            },
            "euler_angles": {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]}
        }

        return json.dumps(msg)
