import numpy as np
import manifpy as manif
from dataclasses import dataclass, field


@dataclass
class LieState:
    extended_pose: manif.SE3
    # gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    g: np.ndarray = field(default_factory=lambda: np.array([0, 0, 9.822]))

    def dof(sefl):
        return 6 + 3

