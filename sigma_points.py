import numpy as np
import scipy
from dataclasses import dataclass


class SigmaPoints:
    """
    Computes sigma points and weights using the scheme by Wan & van der Mwere
    [1]: https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    """

    def __init__(self, n, alpha=1e-1, beta=2, kappa=0):
        self.lamb = self.alpha**2 / (self.n + self.kappa) - self.n
        self.n = n
        self.num_sigmas = 2 * n + 1
        self.Wm = np.zeros(n)
        self.Wc = np.zeros(n)
        # self.points = np.zeros(n + 1, dtype=float)

    def compute_weights(self):
        a = self.lamb / (self.n + self.lamb)
        self.Wm[0] = a
        self.Wc[0] = a - (1 - self.alpha**2 + self.beta)
        self.Wm[1:] = np.full(self.n, 1 / (2 * (self.n + self.lamb)))
        self.Wc[1:] = self.Wm[1:]

    def compute_sigma_points(self, x, P):
        self.compute_weights()

        L = scipy.linalg.cholesky((self.n + self.lamb) @ P)

        points = np.zeros((self.num_sigmas, self.n))
        points[0] = x

        for i in range(self.n):
            points[i + 1] = x + L
            points[i + 1 + self.n] = x - L

        return points
