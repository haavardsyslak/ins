import numpy as np
from abc import ABC, abstractmethod


class IState(ABC):
    """Interface for state representation in a filter."""
    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        """Position in NED frame"""
        pass

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """Velocity in body or world frame"""
        pass

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Rotation matrix from body to world frame"""
        pass

    @property
    @abstractmethod
    def q(self) -> np.ndarray:
        """Quaternion [w, x, y, z]^T"""
        pass

    @property
    @abstractmethod
    def euler(self) -> tuple:
        """(roll, pitch, yaw) in degrees"""
        pass

    @property
    @abstractmethod
    def gyroscope_bias(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def accelerometer_bias(self) -> np.ndarray:
        pass
