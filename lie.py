import numpy as np
from utils import skew, vee
from scipy.spatial.transform import Rotation as Rot


class SU2:
    TOL = 1e-8

    @classmethod
    def Exp(cls, omega, dt):
        return Rot.from_rotvec(omega*dt).as_quat(scalar_first=True)
        norm = np.linalg.norm(omega)
        angle = norm * dt
        if angle < cls.TOL:
            q = np.array([1.0, *(.5 * omega * dt)])
        else:
            half_angle = angle / 2
            c = np.cos(angle)
            s = omega / norm * np.sin(half_angle)
            q = np.array([c, *s])

        return q

    @classmethod
    def exp(cls, phi):
        angle = np.linalg.norm(phi)
        if angle < 1e-10:
            return np.array([1.0, *(.5 * phi)])

        axis = phi / angle
        half_angle = angle / 2
        c = np.cos(half_angle)
        s = np.sin(half_angle)
        q = np.hstack([c, axis * s])

        return q

    @classmethod
    def log(cls, q):
        qw = q[0]
        qv = q[1:]
        norm = np.linalg.norm(qv)
        
        if norm < cls.TOL:
            # Small-angle approximation
            return 2 * qv / qw
        
        # General case
        theta = 2 * np.arctan2(norm, qw)
        return (theta / norm) * qv

    # @classmethod
    # def log(cls, q):
    #     qw = q[0]
    #     qv = q[1:]
    #     norm = np.linalg.norm(qv)
    #     if norm < cls.TOL:
    #         r = 2 * qv / qw * (1 - norm ** 2 / (3 * qw**2))
    #     else:
    #         theta = 2 * np.atan2(norm, qw)
    #         u = qv / norm
    #         r = theta * u
    #     return r


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
            s = np.sin(anglec, *s)
            Rot = c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * skew(axis)
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

    @classmethod
    def left_jacobian(cls, phi):
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            J = np.eye(3) - .5 * skew(phi)
        else:
            axis = phi / angle
            s = np.sin(angle)
            c = np.cos(angle)
            J = (
                (s / angle) * np.eye(3)
                + (1 - s / angle) * np.outer(axis, axis)
                + ((1 - c) / angle) * skew(phi)
            )

        return J

    @classmethod
    def left_jacobian_inv(cls, phi):
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            J = np.eye(3) - .5 * skew(phi)

        else:
            axis = phi / angle
            half_angle = angle / 2
            cot = 1 / np.tan(half_angle)
            J = half_angle * cot * np.eye(3) + (1 - half_angle * cot) * \
                np.outer(axis, axis) - half_angle * skew(axis)

        return J


class SEK3:
    @classmethod
    def exp(cls, xi):
        k = int(xi.shape[0] / 3 - 1)
        Xi = np.reshape(xi[3:], (3, k), 'F')
        chi = np.eye(3 * k)
        chi[:3, :3] = SO3.exp(xi[:3])
        chi[:3, 3:] = SO3.left_jacobian(xi[:3]).dot(Xi)

        return chi

    @classmethod
    def log(cls, xi):
        phi = SO3.log(xi[:3, :3])
        Xi = SO3.left_jacobian(phi).dot(xi[:3, 3:])
        new_xi = np.hstack([phi, Xi.flatten('F')])

        return new_xi

    @classmethod
    def inv(cls, xi):
        k = xi.shape[0] - 3
        xi_inv = np.eye(3 + k)
        xi_inv[:3, :3] = xi[:3, :3].T
        xi_inv[:3, 3:] = -xi_inv[:3, :3].dot(xi[:3, 3:])

        return xi_inv


