import numpy as np
from dataclasses import dataclass


class UKFMTuning:

    def R_dvl(self, fom=None):
        return np.eye(3) * fom**2 
        return np.diag([0.5**2, 0.5**2, 0.5**2]) * 2

    def R_gnss(self, std=2.5):
        return np.eye(2) * std**2

    def R_depth(self):
        return 0.001**2


class ESEKFTuning:

    def R_dvl(self, fom=None):
        # return np.eye(3) * fom**2
        return np.diag([0.25**2, 0.25**2, 0.1**2])

    def R_gnss(self, std=2.5):
        return np.eye(2) * 0.5**2

    def R_depth(self):
        return np.eye(1) * 0.01**2


class QUKFTuning:

    def R_dvl(self, fom=None):
        return np.eye(3) * fom**2 
        # return np.diag([0.25**2, 0.25**2, 0.1**2])

    def R_gnss(self, std=2.5):
        return np.eye(2) * std**2

    def R_depth(self):
        return np.eye(1) * 0.01**2
