import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from orientation import RotationQuaterion

def test_form_quat_to_rotation_matrix():
    # Define multiple test cases as (euler_angles, tolerance) tuples
    test_cases = [
        ([np.pi / 2, 0, 0], 1e-8, 1e-9),  # 90-degree rotation around X-axis
        ([0, np.pi / 2, 0], 1e-8, 1e-9),  # 90-degree rotation around Y-axis
        ([0, 0, np.pi / 2], 1e-8, 1e-9),  # 90-degree rotation around Z-axis
        ([np.pi / 4, np.pi / 4, np.pi / 4], 1e-8, 1e-9),  # Combined rotation
        ([np.pi, np.pi, np.pi], 1e-8, 1e-9),  # 180-degree rotation around all axes
        ([0, 0, 0], 1e-8, 1e-9),  # Identity rotation
        ([np.pi / 3, -np.pi / 6, np.pi / 2], 1e-8, 1e-9),  # Arbitrary rotation
        ([np.pi / 2, np.pi / 2, np.pi / 2], 1e-8, 1e-9),  # 90-degree pitch (singularity)
    ]

    for euler_angles, rtol, atol in test_cases:
        # Create rotation from Euler angles
        rot = Rot.from_euler("XYZ", euler_angles)
        q = rot.as_quat(scalar_first=True)  # Convert to quaternion (scalar-first format)
        q_ = RotationQuaterion.from_vec(q)  # Create your custom quaternion object

        # Get rotation matrices
        R_scipy = rot.as_matrix()  # Reference matrix from SciPy
        R = q_.R

        # Assert that the matrices are close
        np.testing.assert_allclose(R, R_scipy, rtol=rtol, atol=atol)

def test_something():
    rot1 = Rot.from_euler("XYZ", [0, 0, -np.pi/2])
    rot2 = Rot.from_euler("XYZ", [0, 0, 0])
    qa = rot1.as_quat(scalar_first=True)
    qb = rot2.as_quat(scalar_first=True)
    dr = (rot1.inv() * rot2)
    print(dr.as_euler("XYZ"))
    dq = RotationQuaterion.get_error_quat(qa, qb)

    np.testing.assert_allclose(dq, dr.as_quat(scalar_first=True), rtol=1e-7, atol=1e-9)
    # assert(0 == 1)

    
if __name__ == "__main__":
    import os
    import pytest
    # os.environ["_PYTEST_RAISE"] = "1"
    # pytest.main()
    test_something()

