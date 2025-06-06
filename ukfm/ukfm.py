import numpy as np
from sigma_points import SigmaPoints
from .models import ImuModel
from .state import LieState
from scipy.spatial.transform import Rotation as Rot
from messages_pb2 import UkfState


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
        self.x = x0
        self.P = P0
        self.model = model
        self.phi = self.model.phi
        self.phi_inv = self.model.phi_inv
        self.Q = Q
        self.R = R
        # Holds the latest innvoation
        self.innovation = None
        # Holds the lates innovation covariance
        self.S_inv = None

    def propagate(self, u, dt):
        # Q = self.model.Q_c
        # self.P += 1e-9 * np.eye(self.P.shape[0])

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
        # self.P[9:12, 9:12] += self.model.Q[6:9, 6:9]
        self.P = (self.P + self.P.T) / 2
        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals[eigvals < 0] = 0
        P_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.P = P_psd#(P_psd + P_psd.T) / 2      
        self.x = x_pred
        # self.x = self.model.phi(x_pred, new_xi)

    def update(self, measurement, dt, h=None, R=None):
        # self.P += 1e-8 * np.eye(self.P.shape[0])
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
        Pxz = self.points.Wc_i * np.hstack([xis[:self.dim_x].T, xis[self.dim_x:].T]).dot(new_xis)
        self.S_inv = np.linalg.inv(S)

        K = Pxz @ self.S_inv
        # K = np.linalg.solve(S, Pxz.T).T
        self.innovation = z - z_pred_bar
        xi_plus = K @ self.innovation
        self.x = self.model.phi_up(self.x, xi_plus)
        # self.x = self.phi(self.x, xi_plus)

        self.P -= K @ S @ K.T

        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals[eigvals < 0] = 0
        P_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.P = P_psd
        # self.P = (P_psd + P_psd.T) / 2      
        # Avoid non sysmetric matrices
        # self.P = (self.P + self.P.T) / 2

    def nees_pos(self, true_pos):
        idx = 2
        x_hat = self.x.position[:idx]
        x = true_pos[:idx]
        return (x_hat - x).T @ np.linalg.inv(self.P[:idx, :idx]) @ (x_hat - x)

    def nis(self):
        # Compute the NIS
        return self.innovation.T @ self.S_inv @ self.innovation

    def to_proto_msg(self):
        extended_pose = self.x.extended_pose
        quat = extended_pose.coeffs()[3:7]

        vel = extended_pose.rotation().T @ extended_pose.linearVelocity()
        rot = Rot.from_matrix(extended_pose.rotation())
        roll, pitch, yaw = rot.as_euler("xyz", degrees=True)

        return UkfState(
            position_x=extended_pose.x(),
            position_y=extended_pose.y(),
            position_z=extended_pose.z(),
            quaternion_w=quat[0],
            quaternion_x=quat[1],
            quaternion_y=quat[2],
            quaternion_z=quat[3],
            velocity_x=vel[0],
            velocity_y=vel[1],
            velocity_z=vel[2],
            heading=yaw,
            roll=roll,
            pitch=pitch,
            gyro_bias_x=self.x.gyro_bias[0],
            gyro_bias_y=self.x.gyro_bias[1],
            gyro_bias_z=self.x.gyro_bias[2],
            accel_bias_x=self.x.acc_bias[0],
            accel_bias_y=self.x.acc_bias[1],
            accel_bias_z=self.x.acc_bias[2],
            covariance=np.diag(self.P)
        )

