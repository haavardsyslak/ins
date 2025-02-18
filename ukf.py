import numpy as np
import scipy
from orientation import RotationQuaterion, AttitudeError, average, quaternion_weighted_average
from sigma_points import SigmaPoints
from dataclasses import dataclass
from typing import Callable
from models import ImuModelLie, ImuModelQuat, Measurement
from state import LieState, State
from scipy.spatial.transform import Rotation as Rot
from sigma_points import MwereSigmaPoints


class QUKF:
    def __init__(
        self,
        model,
        dim_x,
        dim_q,
        x0,
        P0,
        sigma_points,
        Q,
        R=None,
    ):
        self.model = model
        self.dim_x = dim_x
        self.dim_q = dim_q
        self.w = np.zeros(dim_q)  # Noise mean
        self.x = x0
        self.P = P0
        n = P0.shape[0] + Q.shape[0]
        self.P_aug = np.zeros((n, n))
        self.sigma_points = sigma_points
        self.Q = Q
        self.R = R
        self.propgated_sigmas = np.zeros((self.sigma_points.num_sigmas, self.dim_x + 1))
        self.delta_sigmas = np.zeros((self.sigma_points.num_sigmas, self.dim_x))

    def propagate(self, u, dt):
        self.P += np.eye(self.dim_x) * 1e-9
        if self.Q is not None:
            Q = self.Q
        else:
            Q = self.model.Q * dt

        Q = self.model.Q * dt

        self.P_aug[:self.dim_x, :self.dim_x] = self.P
        self.P_aug[self.dim_x:, self.dim_x:] = Q

        delta_x_aug_sigmas = self.sigma_points.compute_sigma_points(self.P_aug)
        sigmas = self.propgated_sigmas
        # sigmas[0] = self.model.f(self.x, dt, np.zeros(self._dim_q)).as_vec()
        for i, delta_x_aug in enumerate(delta_x_aug_sigmas):
            delta_x = delta_x_aug[:self.dim_x]
            w = delta_x_aug[self.dim_x:]
            xi = State.add_error_state(self.x, delta_x)
            sigmas[i] = self.model.f(xi, u, dt, w).as_vec()


        # q_bar = average(sigmas[:, :4])
        # print("q_bar: ", q_bar)
        q_bar = quaternion_weighted_average(sigmas[:, :4], self.sigma_points.Wm)
        # print("q_bar", q_bar)

        x_bar_ = np.dot(self.sigma_points.Wm.flatten(), sigmas[:, 4:])
        x_bar = np.concatenate([q_bar, x_bar_])

        P = np.zeros_like(self.P)
        for i, sigma in enumerate(sigmas):
            # x = State.from_vec(sigma)
            # dx = x.to_error_state(x_bar)
            dx = State.to_error_state(sigma, x_bar)
            P += self.sigma_points.Wc[i] * np.outer(dx, dx)
            self.delta_sigmas[i] = dx

        # TODO: when biases are added, we need to add the noise for the biases to the biases
        self.P = (P + P.T) / 2
        self.x = State.from_vec(x_bar)
        self.propgated_sigmas = sigmas

    def update(self, measurement: Measurement, dt: float, R=None):
        self.P += np.eye(self.dim_x) * 1e-9
        if R is None:
            R = measurement.R

        z = measurement.z
        sigmas_z = np.zeros((self.sigma_points.num_sigmas, measurement.dim))
        for i, sigma in enumerate(self.propgated_sigmas):
            # x = State.add_error_state(self.x, sigma)
            x = State.from_vec(sigma)
            sigmas_z[i] = measurement.h(x)


        z_hat = np.dot(self.sigma_points.Wm, sigmas_z)
        
        y = sigmas_z - z_hat[np.newaxis, :]
        Wc_diag = np.diag(self.sigma_points.Wc)
        # S = np.dot(y.T, np.dot(Wc_diag, y)) + measurement.R
        # S_inv = np.linalg.inv(S)
        # print(S_inv)
        Pxz = np.zeros((self.delta_sigmas.shape[1], sigmas_z.shape[1]))

        S = np.zeros((measurement.dim, measurement.dim))
        x = self.x.as_vec()
        sigmas = self.propgated_sigmas
        # print(self.sigma_points.Wc)
        for i in range(self.sigma_points.num_sigmas):
            dx = State.to_error_state(sigmas[i], x)
            dz = sigmas_z[i] - z_hat
            Pxz += self.sigma_points.Wc[i] * np.outer(dx, dz)
            S += self.sigma_points.Wc[i] * np.outer(dz, dz)

        S = S + measurement.R
        # dz = sigmas_z - z_hat[np.newaxis, :]
        # Pxz = (self.sigma_points.Wc * self.delta_sigmas) @ dz
        # S_inv = np.linalg.inv(S)
        S_inv = S
        innov = z - z_hat

        K = Pxz @ S_inv
        print(K)

        dx = K @ innov
        self.x = State.add_error_state(self.x, dx)
        P = self.P - K @ S @ K.T
        self.P = (P + P.T) / 2


class UKFM:
    def __init__(
        self,
        dim_x: int,
        dim_q: int,
        points: SigmaPoints,
        noise_points: SigmaPoints,
        model: ImuModelLie,
        x0: LieState,
        P0: np.ndarray,
        Q=None,
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
        self.phi = self.model.phi
        self.phi_inv = self.model.phi_inv
        self.Q = Q
        self.R = R

    def propagate(self, u, dt):
        # Q = self.model.Q_c
        self.P += 1e-9 * np.eye(self.P.shape[0])

        if self.Q is not None:
            Q = self.Q * dt  # TOOO: need to get the discritized Q mat
        else:
            Q = self.model.Q * dt

        w_q = np.zeros(self.dim_q)
        # Predict the nominal state
        x_pred = self.model.f(self.x, u, dt, w_q)
        # Points in the Lie algebra
        xis = self.points.compute_sigma_points(np.zeros(self.dim_x), self.P)
        # Points in the manifold
        new_xis = np.zeros_like(xis)
        # Retract the sigma points onto the manifold
        for i, point in enumerate(xis):
            s = self.phi(self.x, point)
            new_s = self.model.f(s, u, dt, w_q)
            new_xis[i] = self.phi_inv(x_pred, new_s)

        new_xi = self.points.Wm_i * np.sum(
            new_xis, 0
        )  # + self.points.Wm_0 * self.model.phi_inv(self.x, x_pred)
        new_xis = new_xis - new_xi

        P = self.points.Wc_i * new_xis.T.dot(new_xis) + self.points.Wc_0 * np.outer(new_xi, new_xi)

        # Now do the same for the noise term
        # Compute noise sigma poinst
        noise_sigmas = self.noise_points.compute_sigma_points(w_q, Q)

        new_xis = np.zeros((self.points.num_sigmas, self.dim_x))
        # Propagation of uncertainty
        for i, point in enumerate(noise_sigmas):
            s = self.model.f(self.x, u, dt, point)
            new_xis[i] = self.phi_inv(x_pred, s)

        # Compute the covariance
        xi_bar = self.noise_points.Wm_i * np.sum(new_xis, 0)
        # xi_bar ="" (1 / self.noise_points.num_sigmas) * np.sum(new_xis, 0)
        new_xis = new_xis - xi_bar

        Q = self.points.Wc_i * new_xis.T.dot(new_xis) + self.points.Wc_0 * np.outer(xi_bar, xi_bar)
        self.P = P + Q
        # self.P = (self.P + self.P.T) / 2
        self.x = x_pred
        # self.x = self.model.phi(x_pred, new_xi)

    def update(self, measurement, dt, h=None, R=None):
        self.P += 1e-8 * np.zeros_like(self.P)
        h = measurement.h

        if R is None:
            R = measurement.R

        z = measurement.z

        xis = self.points.compute_sigma_points(np.zeros((self.dim_x)), self.P)

        new_xis = np.zeros((self.points.num_sigmas, measurement.dim))
        y0 = h(self.x)

        for i, point in enumerate(xis):
            new_xis[i] = h(self.phi(self.x, point))

        z_pred_bar = self.points.Wm_i * np.sum(new_xis, 0) + self.points.Wm_0 * y0
        # z_pred_bar = 1 / len(new_xis) * np.sum(new_xis, 0)

        dz = y0 - z_pred_bar
        new_xis = new_xis - z_pred_bar

        S = self.points.Wc_i * new_xis.T.dot(new_xis) + self.points.Wc_0 * np.outer(dz, dz) + R
        Pxz = self.points.Wc_i * np.hstack([xis[:9].T, xis[9:].T]).dot(new_xis)
        S_inv = np.linalg.inv(S)

        K = Pxz @ S_inv
        # K = np.linalg.solve(S, Pxz.T).T
        innov = z - z_pred_bar
        xi_plus = K @ innov

        self.x = self.phi(self.x, xi_plus)

        self.P -= K @ S @ K.T

        # Avoid non sysmetric matrices
        self.P = (self.P + self.P.T) / 2
