import numpy as np
import scipy
from dataclasses import dataclass


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
        # self.points = np.zeros(n + 1, dtype=float)

    def compute_weights(self):
        lambda_ = self.lamb
        n = self.n
        alpha = self.alpha
        beta = self.beta
        self.Wm_0 = lambda_ / (n + lambda_)
        self.Wc_0 = self.Wc_0 + (1 - alpha**2 + beta)
        self.Wm_i = 1 / (2 * (n + lambda_))
        self.Wc_i = self.Wm_i

    def compute_sigma_points(self, x, P):
        self.compute_weights()

        L = (self.n + self.lamb) * np.linalg.cholesky(P)

        points = np.zeros((self.num_sigmas, self.n))
        points[0] = x

        for i in range(self.n):
            points[i] = x + L[i]
            points[i + self.n] = x - L[i]

        return points
