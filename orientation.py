import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rot
from utils import skew
from typing import Union
from abc import ABC, abstractmethod


class Orientation(ABC):
    ori: Union['RotationQuaterion', np.ndarray]

    @abstractmethod
    def R(self):
        pass


class RotationMatrix(Orientation):
    R: np.ndarray


@dataclass
class RotationQuaterion(Orientation):
    eta: float
    epsilon: np.ndarray

    def __post_init__(self):
        if len(self.epsilon) != 3:
            raise ValueError("Vector part must be 3 elements")

        norm = np.linalg.norm(self.as_vec())
        if not np.allclose(norm, 1):
            self.eta /= norm
            self.epsilon /= norm
        if self.eta < 0:
            self.eta *= -1
            self.epsilon *= -1

    @property
    def R(self):
        """As rotation matrix"""
        S = skew(self.epsilon)
        R = np.eye(3) + 2 * self.eta * S + 2 * S @ S

        return R

    def multiply(self, other):
        eta_a = self.eta
        epsilon_a = self.epsilon

        if isinstance(other, RotationQuaterion):
            eta_b = other.eta
            epsilon_b = other.epsilon

        elif isinstance(other, np.ndarray):
            eta_b = other[0]
            epsilon_b = other[1:4]

        else:
            raise ValueError("Quaternion must be numpy array or RotationQuaternion")

        eta, epsilon = self._multiply(eta_a, epsilon_a, eta_b, epsilon_b)

        return RotationQuaterion(eta, epsilon)

    @staticmethod
    def _multiply(eta_a, epsilon_a, eta_b, epsilon_b):
        eta = eta_a * eta_b - np.dot(epsilon_a, epsilon_b)
        epsilon = eta_b * epsilon_a + eta_a * epsilon_b + np.cross(epsilon_a, epsilon_b)
        return eta, epsilon

    def rotate_vec(self, vec):
        t = 2 * np.cross(self.epsilon, vec)
        v_rot = vec + self.eta * t + np.cross(self.epsilon, t)
        return v_rot

    def conj(self):
        return RotationQuaterion(self.eta, -self.epsilon)

    def as_vec(self):
        return np.array([self.eta, *self.epsilon])

    def as_euler(self):
        n = self.eta
        e1, e2, e3 = self.epsilon
        phi = np.atan2(2 * (e3 * e2 + n * e1), n**2 - e1**2 - e2**2 + e3**2)
        theta = np.asin(2 * (n * e2 - e1 * e3))
        psi = np.atan2(2 * (e1 * e2 + n * e3), n**2 + e1**2 - e2**2 - e3**2)

        return np.array([phi, theta, psi])

    @staticmethod
    def get_error_quat(qa, qb):
        eta_a = qa[0]
        epsilon_a = qa[1:]
        eta_b = qb[0]
        epsilon_b = -qb[1:]
        eta, epsilon = RotationQuaterion._multiply(eta_a, epsilon_a, eta_b, epsilon_b)
        q = np.array((eta, *epsilon))
        q /= np.linalg.norm(q)
        return q

    @classmethod
    def from_vec(cls, arr):
        return cls(arr[0], arr[1:])

    def __matmul__(self, other):
        return self.multiply(other)


class AttitudeError:
    """Translates error quaterion into minmal error represntation and vice versa"""

    @staticmethod
    def to_rodrigues_param(quaterion):
        if isinstance(quaterion, RotationQuaterion):
            quaterion = quaterion.to_vec()
        if len(quaterion) != 4:
            raise ValueError("Quaternion must be of length 4")

        return 2 * (quaterion[1:] / quaterion[0])

    @staticmethod
    def from_rodrigues_param(arr: np.ndarray):
        if len(arr) != 3:
            raise ValueError("Rodrigues parameter must be of length 3")

        norm_sq = np.linalg.norm(arr) ** 2
        return (np.sqrt(4 + norm_sq)) * np.hstack((2, arr))


def quaternion_weighted_average(quats, weights):
    C = np.zeros((4, 4))
    for i, q in enumerate(quats):
        C += weights[i] * np.outer(q, q)

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    mean_q = eigenvectors[:, np.argmax(eigenvalues)]
    # Todo should check if eta is negative and change the sign?

    # Ensure unit constraint
    mean_q /= np.linalg.norm(mean_q)
    return mean_q


def average(quats):
    # Should concider changing this to use the faster iterative procedure referenced in Kraft paper
    """Computes the average of a set of quaternions"""

    # Compute the covariance matrix
    C = np.zeros((4, 4))
    for q in quats:
        C += np.outer(q, q)
    C /= len(quats)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # The eigenvector with the largest eigenvalue is the mean quaternion
    mean_q = eigenvectors[:, np.argmax(eigenvalues)]

    # Normalize to ensure unit constraint
    mean_q /= np.linalg.norm(mean_q)
    return mean_q
