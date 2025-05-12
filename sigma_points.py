import numpy as np
import scipy
from dataclasses import dataclass


class JulierSigmaPoints:
    def __init__(self, n, alpha):
        # propagation w.r.t. state
        self.n = n
        self.lambda_ = (alpha**2 - 1) * n
        self.num_sigmas = 2 * n
        self.Wm_i = 1 / (2 * (self.lambda_ + n))  # = wj
        self.Wm_0 = self.lambda_ / (self.lambda_ + n)  # wm
        self.Wc_0 = self.lambda_ / (self.lambda_ + n) + 3 - alpha**2  # w0
        self.Wc_i = 1 / (2 * (self.lambda_ + n))

    def compute_sigma_points(self, x, P) -> np.ndarray:
        L = np.linalg.cholesky((self.n + self.lambda_) * P).T
        points = np.zeros((self.num_sigmas, self.n))
        for i in range(self.n):
            points[i] = x + L[i]
            points[i + self.n] = x - L[i]

        return points


class MwereSigmaPoints:
    def __init__(self, n, alpha=1e-1, beta=2, kappa=0):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.num_sigmas = 2 * n + 1
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        self._compute_weights()

    def _compute_weights(self):
        lambda_ = self.lambda_

        c = .5 / (self.n + lambda_)
        self.Wc = np.full(2 * self.n + 1, c)
        self.Wm = np.full(2 * self.n + 1, c)
        self.Wc[0] = lambda_ / (self.n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (self.n + lambda_)

    def compute_sigma_points(self, P):
        L = np.linalg.cholesky((self.n + self.lambda_) * P, upper=False).T

        points = np.zeros((self.num_sigmas, self.n))

        points[0] = np.zeros(P.shape[0])

        for i in range(self.n):
            # points[i + 1] = L[:, i]
            # points[i + 1 + self.n] = -L[:, i]
            points[i + 1] = L[i]
            points[i + self.n + 1] = -L[i]

        return points

class SigmaPoints:
    """
    Computes sigma points and weights using the scheme by Wan & van der Mwere
    [1]: https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    """

    def __init__(self, n, alpha=1e-1, beta=2, kappa=0):
        self.lamb = alpha**2 * (n + kappa) - n
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.num_sigmas = 2 * n
        self.Wm_0 = 0
        self.Wm_i = 0
        self.Wc_0 = 0
        self.Wc_i = 0
        self._compute_weights()

    def _compute_weights(self):
        lambda_ = self.lamb
        n = self.n
        alpha = self.alpha
        beta = self.beta
        self.Wm_0 = lambda_ / (n + lambda_)
        self.Wc_0 = self.Wc_0 + (1 - alpha**2 + beta)
        self.Wm_i = 1 / (2 * (n + lambda_))
        self.Wc_i = self.Wm_i

    def compute_sigma_points(self, x, P):

        L = np.linalg.cholesky((self.n + self.lamb) * P).T

        points = np.zeros((self.num_sigmas, self.n))
        points[0] = x

        for i in range(self.n):
            points[i] = x + L[i]
            points[i + self.n] = x - L[i]

        return points


class SimplexSigmaPoints(object):
    def __init__(self, n, alpha=1.0):
        self.n = n
        self.alpha = alpha
        self.num_sigmas = self.n + 1

        self._compute_weights()




    def compute_sigma_points(self, P):
        n = self.n

        x = np.zeros(P.shape[0])
        if np.isscalar(x):
            x = np.asarray([x])
        x = x.reshape(-1, 1)

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        U = scipy.linalg.cholesky(P)

        lambda_ = n / (n + 1)
        Istar = np.array([[-1/np.sqrt(2*lambda_), 1/np.sqrt(2*lambda_)]])

        for d in range(2, n+1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1. / np.sqrt(lambda_*d*(d + 1)) # pylint: disable=unsubscriptable-object
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row] # pylint: disable=unsubscriptable-object

        I = np.sqrt(n)*Istar
        scaled_unitary = (U.T).dot(I)

        sigmas = np.subtract(x, -scaled_unitary)
        return sigmas.T


    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter. """

        n = self.n
        c = 1. / (n + 1)
        self.Wm = np.full(n + 1, c)
        self.Wc = self.Wm
