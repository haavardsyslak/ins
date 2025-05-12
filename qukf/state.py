import numpy as np
from dataclasses import dataclass
from orientation import Orientation, RotationQuaterion, AttitudeError, RotationMatrix
import json
from scipy.spatial.transform import Rotation
from typing import Self, Union
import manifpy as manif
from istate import IState


@dataclass
class State(IState):
    ori: manif.SO3
    vel: np.ndarray
    pos: np.ndarray

    def dof(self):
        return 9



