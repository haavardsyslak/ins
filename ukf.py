import numpy as np
import scipy
from quaternion import RotationQuaterion, AttitudeError, average
from sigma_points import SigmaPoints
from dataclasses import dataclass
from typing import Callable
from models import ImuModel
from state import LieState


class QUKF:
    """Quaternion UKF based on a paper by Edgar Kraft"""

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
                self.x.add_error_state(ErrorState.from_vec(dx))
            ).split()

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


class UKFM:
    def __init__(
        self,
        dim_x: int,
        dim_q: int,
        points: SigmaPoints,
        noise_points: SigmaPoints,
        model: ImuModel,
        x0: LieState,
        P0: np.ndarray,
        Q,
        R=None,
    ):
        self.points = points
        self.noise_points = noise_points
        self.dim_q = dim_q
        self.dim_x = dim_x
        # self.dim_z
        self.x = x0
        self.P = P0
        self.model = model
        self.Q = Q
        self.R = R

    def propagate(self, u, dt):
        # Q = self.model.Q_c
        # self.P += 1e-9 * np.eye(self.P.shape[0])
        Q = self.Q  # TOOO: need to get the discritized Q mat
        # Predict the nominal state
        w_q = np.zeros(self.dim_q)
        x_pred = self.model.f(self.x, u, dt, w_q)
        # Points in the Lie algebra
        xis = self.points.compute_sigma_points(np.zeros(self.dim_x), self.P)
        # Points in t he manifold
        new_xis = np.zeros_like(xis)
        # Retract the sigma points onto the manifold
        w = np.zeros(6)
        for i, point in enumerate(xis):
            s = self.model.phi(self.x, point)
            new_s = self.model.f(s, u, dt, w)
            new_xis[i] = self.model.phi_inv(new_s, self.x)

        new_xi = self.points.Wm_i * np.sum(new_xis, 0) + self.points.Wm_0 * self.model.phi_inv(x_pred, self.x)
        # new_xi = (1 / self.points.num_sigmas) * np.sum(new_xis, 0)
        # new_xi = (1 / self.points.num_sigmas) * np.sum(new_xis, 0)
        new_xis = new_xis - new_xi

        P = self.points.Wc_i * new_xis.T.dot(new_xis) + self.points.Wc_0 * np.outer(new_xi, new_xi)

        # Now do the same for the noise term
        # Compute noise sigma poinst
        noise_sigmas = self.noise_points.compute_sigma_points(w_q, Q)

        new_xis = np.zeros((self.points.num_sigmas, self.dim_x))
        # Propagation of uncertainty
        for i, point in enumerate(noise_sigmas):
            s = self.model.f(self.x, u, dt, point)
            new_xis[i] = self.model.phi_inv(x_pred, s)

        # Compute the covariance
        xi_bar = self.noise_points.Wm_i * np.sum(new_xis, 0)
        # xi_bar = (1 / self.noise_points.num_sigmas) * np.sum(new_xis, 0)
        new_xis = new_xis - xi_bar

        Q = self.points.Wm_i * new_xis.T.dot(new_xis) + self.points.Wc_0 * np.outer(
            xi_bar, xi_bar
        )
        self.P = P + Q
        self.P = (self.P + self.P.T) / 2
        print("P-", self.P)
        self.x = x_pred

    def update(self, z, dt, h=None, R=None):
        # self.P += 1e-9 * np.ones_like(self.P)
        if h is None:
            h = self.model.h

        if R is None:
            R = self.R

        xis = self.points.compute_sigma_points(np.zeros((self.dim_x)), self.P)

        new_xis = np.zeros((self.points.num_sigmas, len(z)))
        y0 = h(self.model.phi(self.x, np.zeros(self.dim_x)))

        for i, point in enumerate(xis):
            new_xis[i] = h(self.model.phi(self.x, point))

        z_pred_bar = self.points.Wm_0 * y0 + self.points.Wm_i * np.sum(new_xis, 0)

        print(f"z:{z}")
        print(f"z pred: {z_pred_bar}")
        dz = y0 - z_pred_bar

        S = self.points.Wc_i * new_xis.T.dot(new_xis) + self.points.Wc_0 * np.outer(dz, dz) + R
        Pxz = self.points.Wc_i * np.hstack([xis[:9].T, xis[9:].T]).dot(new_xis)

        K = np.linalg.solve(S, Pxz.T).T
        innov = z - z_pred_bar
        xi_plus = K @ innov

        print("x_pre: ", self.x)
        self.x = self.model.phi(self.x, xi_plus)
        print("x_post: ", self.x)
        self.P -= K @ S @ K.T

        # Avoid non sysmetric matrices
        self.P = (self.P + self.P.T) / 2
