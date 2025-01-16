import numpy as np
import scipy
from quaternion import RotationQuaterion, AttitudeError, average
from sigma_points import SigmaPoints
from dataclasses import dataclass


class QUKF:
    """Quaternion UKF based on a paper by Edgar Kraft """

    def __init__(self, dim_x, dim_z, x0, P0, f, h, points: SigmaPoints):
        self.points = points
        self.x = State.from_vec(x0)
        self.P = P0.copy()
        self.P_prior = np.zeros_like(P0)
        self.x_err = np.array(dim_x - 1)
        self.f = f
        self.h = h

        # Delta sigma points for the error state distrubuted about 0
        self.delta_sigma = np.zeros((dim_x - 1, self.points.num_sigmas))
        self.propagated_delta_sigmas = np.zeros((dim_x, self.points.num_sigmas))
        # Simgas to be propagated through
        # Propagated simga p

    def generate_sigmas(self, P):
        self.points.compute_sigma_points(0, P)

    def predict(self):
        # 1. generate sigma poinst about zero for the error state
        self.P += self.Q
        self.delta_sigma = self.generate_sigmas(self.P)

        # 2. Add the current state estimate to the propagations
        # 3. Propagate the sigma points through the state transition function
        propagated_sigmas = np.zeros((self.dim_x, self.points.num_sigmas))
        x_prior = np.zeros(self.x.dim_euclidean())
        x_prior_q = np.zeros(self.x.dim_non_euclidean())
        for i, dx in enumerate(self.delta_sigma):
            x_prior[i], x_prior_q[i] = self.f(
                self.x.add_error_state(ErrorState.from_vec(dx))).split()

        # 4. Cmpute priori state estimate as the mean of these sigma points
        q_mean = RotationQuaterion(average(x_prior_q))
        x_mean = np.dot(self.Wc[:6], x_prior)
        self.propagated_delta_sigmas = np.zeros_like(self.delta_sigma)

        for i in range(x_mean):
            self.propagated_delta_sigmas[:6] = x_prior[i] - x_mean
            self.propagated_delta_sigmas[6:9] = x_prior_q[i] @ q_mean.conj()

        self.x = State.from_vec(np.hstack(x_mean, q_mean))
        self.P_prior = 1 / (2 * self.dim_x) * propagated_sigmas.T.dot(propagated_sigmas).T

        # Compute new weights with the new cov
        self.points.compute_weights(self.P_prior)

    def update(self, z):
        # Take the propagated sigma poinst and project them onto the measurement space
        sigmas_z = np.zeros((len(z), self.points.num_sigmas))
        for i in self.propagated_delta_sigmas:
            sigmas_z = self.h(i)

        # The predicted measurement, assumed to be euclidean
        z_pred = np.dot(self.points.Wm, sigmas_z)
        innov = z - z_pred
        innov_sigmas = np.atleast_2d(sigmas_z) - z_pred[np.newaxis, :]
        Wc_diag = np.diag(self.points.Wc)
        # Innovation covariance
        S = np.dot(innov_sigmas.T, np.dot(Wc_diag, innov_sigmas)) + self.R

        # Cross covariance
        Pxz = np.zeros((self._dim_x, self._dim_z))
        for i in range(self.points.num_sigmas):
            dz = innov_sigmas[i] - z_pred
            Pxz += self.points.Wc[i] * np.outer(self.propagated_delta_sigmas[i], dz)

        # Kalman gain 
        Sinv = np.linalg.solve(S, Pxz).T
        # GPT: K = np.linalg.solve(S.T, Pxz.T).T

        K = Pxz @ Sinv

        self.x.add_error_state(K @ innov)

        self.P -= K @ S @ K.T


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    ori: RotationQuaterion

    def as_vec(self) -> np.ndarray:
        return np.hstack(self.pos, self.vel, self.ori.as_vec())

    @staticmethod
    def from_vec(vec: np.ndarray):
        pos = vec[0:3]
        vel = vec[3:6]
        ori = RotationQuaterion.from_vec(vec[6:10])

        return State(pos, vel, ori)

    def add_error_state(self, error_state: ErrorState):
        self.pos + error_state.pos
        self.vel + error_state.vel
        self.ori @ AttitudeError.from_rodrigues_param(error_state.ori)

    def split(self):
        euclidean = np.hstack(self.pos, self.vel)
        return (euclidean, self.ori.as_vec())

    def dim_euclidean(self):
        return 9

    def dim_non_euclidean(self):
        return 4


@dataclass
class ErrorState:
    pos: np.ndarray
    vel: np.ndarray
    ori: np.ndarray

    def as_vec(self):
        return np.hstack(self.pos, self.vel, self.ori)

    @staticmethod
    def from_vec(vec):
        pos = vec[0:3]
        vel = vec[3:6]
        ori = vec[6:9]
        return ErrorState(pos, vel, ori)
