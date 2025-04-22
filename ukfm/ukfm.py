import numpy as np
from sigma_points import SigmaPoints
from .models import ImuModel, DvlMeasurement
from .state import LieState


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
            Q = self.Q  # * dt  # TOOO: need to get the discritized Q mat
        else:
            Q = self.model.Q  # * dt**2

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
        self.P = (self.P + self.P.T) / 2
        self.x = x_pred
        # self.x = self.model.phi(x_pred, new_xi)

    def update(self, measurement, dt, h=None, R=None):
        self.P += 1e-8 * np.eye(self.P.shape[0])
        h = measurement.h

        if R is None:
            R = measurement.R

        z = measurement.z
        if type(measurement) is DvlMeasurement:
            z = self.x.extended_pose.rotation().T @ measurement.z

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
        Pxz = self.points.Wc_i * np.hstack([xis[:self.dim_x].T, xis[self.dim_x:].T]).dot(new_xis)
        S_inv = np.linalg.inv(S)

        K = Pxz @ S_inv
        # K = np.linalg.solve(S, Pxz.T).T
        innov = z - z_pred_bar
        xi_plus = K @ innov
        if type(measurement) is DvlMeasurement:
            print("innov: ", innov)
            print("xi_plus: ", xi_plus[-3:])

        self.x = self.phi(self.x, xi_plus)

        self.P -= K @ S @ K.T

        # Avoid non sysmetric matrices
        self.P = (self.P + self.P.T) / 2


def make_ukf(x0: LieState, P0: np.ndarray):
    # Define the parameters for the UKF
    dim_x = x0.dof()  # State dimension
    dim_q = 6  # Process noise dimension
    points = SigmaPoints(dim_x, alpha=1e-4, beta=2, kappa=3 - dim_x)
    noise_points = SigmaPoints(dim_q, alpha=1e-4, beta=2, kappa=3 - dim_q)
    model = ImuModel(
        gyro_std=1e-3,
        gyro_bias_std=0.01,
        gyro_bias_p=0.0001,
        accel_std=10,
        accel_bias_std=0.01,
        accel_bias_p=0.0001,
    )
    # Create an instance of the UKFM class
    ukf = UKFM(dim_x, dim_q, points, noise_points, model, x0, P0)
    return ukf
