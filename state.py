import numpy as np
from dataclasses import dataclass
from quaternion import RotationQuaterion, AttitudeError
import json
from scipy.spatial.transform import Rotation


@dataclass
class LieState:
    R: np.ndarray
    vel: np.ndarray
    pos: np.ndarray
    # g: float

    def to_json(self) -> str:
        q = Rotation.from_matrix(self.R).as_quat()
        euler = Rotation.from_matrix(self.R).as_euler("XYZ")

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
