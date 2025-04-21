import numpy as np
import manifpy as manif
from dataclasses import dataclass, field
import pymap3d as pm


@dataclass
class LieState:
    extended_pose: manif.SE_2_3
    # gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    g: np.ndarray = field(default_factory=lambda: np.array([0, 0, 9.822]))

    def dof(self):
        return 9 + len(self.g)

    def to_global_position(self, initial_global_pos):
        lat0 = initial_global_pos[0]
        long0 = initial_global_pos[1]
        alt0 = 0.0
        if len(initial_global_pos) == 3:
            alt0 = initial_global_pos[2]

        pos = self.extended_pose.translation()
        lat, lon, alt = pm.ned2geodetic(pos[0], pos[1], pos[2], lat0, long0, alt0)

        return lat, lon, alt

