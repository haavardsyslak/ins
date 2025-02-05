from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from state import LieState


@dataclass
class ImuModel:
    """The IMU is considered a dynamic model instead of a sensar.
    This works as an IMU measures the change between two states,
    and not the state itself.."""

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    # accm_correction: 'np.ndarray[3, 3]'
    # gyro_correction: 'np.ndarray[3, 3]'

    # g: 'np.ndarray[3]' = field(default=np.array([0, 0, 1]))

    Q_c: 'np.ndarray[6, 6]' = field(init=False, repr=False)

    def __post_init__(self):
        self.g = np.array((0, 0, 1.04))

    def f(self, state: LieState, u: np.ndarray, dt: float, w: np.ndarray):
        acc = u[:3] #+ w[:3]
        omega = u[3:6] #+ w[3:6]
        R = state.R @ Rot.from_rotvec(omega * dt).as_matrix()
        p = state.pos + state.vel * dt + 1/2 * ((state.R @ acc) - self.g) * dt**2
        v = state.vel + ((state.R @ acc) - self.g) * dt
        # print(state.R @ acc)
        return LieState(R, p, v)

    def h(self, state: LieState):
        return state.R.T @ state.vel

    @classmethod
    def phi(cls, state, xi):
        """Takes points in the euclidean space onto the manifold (Exp map)"""
        R = state.R @ Rot.from_rotvec(xi[:3]).as_matrix()
        p = state.pos + xi[3:6]
        v = state.vel + xi[6:9]
        return LieState(R, p, v)

    @classmethod
    def phi_inv(cls, state, state_hat):
        """Takes points from the manifold onto the Lie algebra (Log map)"""
        rot = Rot.from_matrix(state.R @ state_hat.R.T).as_rotvec()
        p = state.pos - state_hat.pos
        v = state.vel - state_hat.vel

        return np.hstack([rot, p, v])

    # def __post_init__(self):
        # def diag3(x): return np.diag([x]*3)

        # accm_corr = self.accm_correction
        # gyro_corr = self.gyro_correction

      #   self.Q_c = scipy.linalg.block_diag(
      #       accm_corr @ diag3(self.accm_std**2) @ accm_corr.T,
      #       gyro_corr @ diag3(self.gyro_std**2) @ gyro_corr.T,
      #       diag3(self.accm_bias_std**2),
      #       diag3(self.gyro_bias_std**2)
      #   )
