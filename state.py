import numpy as np
from dataclasses import dataclass
from quaternion import RotationQuaterion, AttitudeError
import json
from scipy.spatial.transform import Rotation
import manifpy as manif


@dataclass
class LieState:
    state: manif.SE_2_3
    g: float

    def to_json(self) -> str:
        R = self.state.rotation()
        q = Rotation.from_matrix(R).as_quat(scalar_first=True)
        euler = Rotation.from_matrix(R).as_euler("XYZ")

        msg = {
            "velocity": {"x": self.state.vx(), "y": self.state.vy(), "z": self.state.vz()},
            "pose": {
                "position": {"x": self.state.x(), "y": self.state.y(), "z": self.state.z()},
                "orientation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
            },
            "euler_angles": {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]},
            # "gyro_bias": {"x": self.gyro_bias[0], "y": self.gyro_bias[1], "z": self.gyro_bias[2]},
            # "acc_bias": {"x": self.acc_bias[0], "y": self.acc_bias[1], "z": self.acc_bias[2]},
            "gravity": {"g": self.g},
        }

        return json.dumps(msg)
