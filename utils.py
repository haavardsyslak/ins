import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    v_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    return v_skew


def vee(Phi) -> np.ndarray:
    phi = np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])
    return phi
