import numpy as np
import scipy
from dataclasses import dataclass

class JulierSigmaPoints:
        def __init__(self, n, alpha):
            # propagation w.r.t. state
            self.n = n
            self.lambda_ = (alpha**2 - 1) * n
            self.num_sigmas = 2 * n
            self.Wm_i = 1 / (2 * (self.lambda_ + n)) # = wj
            self.Wm_0 = self.lambda_ / (self.lambda_ + n) # wm
            self.Wc_i = self.lambda_ / (self.lambda_ + n) + 3 - alpha**2 # w0
            self.Wc_0 = self.lambda_ / (self.lambda_ + n) 

        def compute_sigma_points(self, x, P) -> np.ndarray:
            L = np.linalg.cholesky((self.n + self.lambda_) * P).T
            points = np.zeros((self.num_sigmas, self.n))
            for i in range(self.n):
                points[i] = x + L[i]
                points[i + self.n] = x - L[i]

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

        L = np.linalg.cholesky((self.n + self.lamb) * P).T

        points = np.zeros((self.num_sigmas, self.n))
        points[0] = x

        for i in range(self.n):
            points[i] = x + L[i]
            points[i + self.n] = x - L[i]

        return points
