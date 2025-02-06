import numpy as np
from utils import skew, vee


class SO3:
    TOL = 1e-8

    @classmethod
    def exp(cls, phi):
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            Rot = np.eye(3) + skew(phi)
        else:
            axis = phi / angle
            c = np.cos(angle)
            s = np.sin(angle)
            Rot = c * np.eye(3) + (1 - c) * np.outer(axis,
                                                     axis) + s * skew(axis)
        return Rot

    @classmethod
    def log(cls, Rot):
        cos_angle = 0.5 * np.trace(Rot) - 0.5
        # Clip np.cos(angle) to its proper domain to avoid NaNs from rounding
        # errors
        cos_angle = np.min([np.max([cos_angle, -1]), 1])
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.linalg.norm(angle) < cls.TOL:
            phi = vee(Rot - np.eye(3))
        else:
            # Otherwise take the matrix logarithm and return the rotation vector
            phi = vee((0.5 * angle / np.sin(angle)) * (Rot - Rot.T))
        return phi
