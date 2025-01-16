import numpy as np
from dataclasses import dataclass


@dataclass
class RotationQuaterion:
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

        eta = eta_a * eta_b - np.dot(epsilon_a, epsilon_b)
        epsilon = eta_b * epsilon_a + eta_a * epsilon_b + np.cross(eta_a, eta_b)

        return RotationQuaterion(eta, epsilon)

    def conj(self):
        return RotationQuaterion(self.eta, -self.epsilon)

    def as_vec(self):
        return np.hstack((self.eta, self.epsilon))

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
