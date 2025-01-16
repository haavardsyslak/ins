import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from quaternion import RotationQuaterion, AttitudeError


def test_to_rodrigues_param():
    q = np.array([1, 0, 0, 0])
    rp = AttitudeError.to_rodrigues_param(q)
    assert rp.all() == 0
