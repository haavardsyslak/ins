import numpy as np
import manifpy as manif
from dataclasses import dataclass, field
import pymap3d as pm
from datetime import datetime
from pygeomag import geomag
from istate import IState
from scipy.spatial.transform import Rotation as Rot


@dataclass
class LieState(IState):
    extended_pose: manif.SE_2_3
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    g: np.ndarray = field(default_factory=lambda: np.array([0, 0, 9.819]))

    def dof(self):
        return 9 + len(self.gyro_bias) + len(self.acc_bias)

    @property
    def position(self) -> np.ndarray:
        return self.extended_pose.translation()

    @property
    def velocity(self) -> np.ndarray:
        return self.extended_pose.linearVelocity()

    @property
    def R(self) -> np.ndarray:
        return self.extended_pose.rotation()

    @property
    def q(self) -> np.ndarray:
        # print(self.extended_pose.coeffs()[3:7])
        # input()
        return Rot.from_matrix(self.R).as_quat()
        # return self.extended_pose.coeffs()[3:7]

    @property
    def euler(self) -> np.ndarray:
        return Rot.from_matrix(self.R).as_euler("xyz", degress=True)

    @property
    def gyroscope_bias(self) -> np.ndarray:
        return self.gyro_bias

    @property
    def accelerometer_bias(self) -> np.ndarray:
        return self.acc_bias

    #
    # def to_global_position(self):
    #     lat0 = self.initial_global_pos[0]
    #     long0 = self.initial_global_pos[1]
    #     alt0 = 0.0
    #     if len(self.initial_global_pos) == 3:
    #         alt0 = self.initial_global_pos[2]
    #
    #     pos = self.extended_pose.translation()
    #     lat, lon, alt = pm.ned2geodetic(pos[0], pos[1], pos[2], lat0, long0, alt0)
    #
    #     return lat, lon, alt
    #
    # def from_global_position(self, lat, lon):
    #     lat0 = self.initial_global_pos[0]
    #     long0 = self.initial_global_pos[1]
    #     alt0 = 0.0
    #     if len(self.initial_global_pos) == 3:
    #         alt0 = self.initial_global_pos[2]
    #
    #     pos = pm.geodetic2ned(lat, lon, 0.0, lat0, long0, alt0)
    #
    #     return np.array([pos[0], pos[1]])
    #
    #
    # def _datetime_to_decimal_year(self, dt: datetime) -> float:
    #     year_start = datetime(dt.year, 1, 1)
    #     next_year = datetime(dt.year + 1, 1, 1)
    #     year_length = (next_year - year_start).total_seconds()
    #     elapsed = (dt - year_start).total_seconds()
    #     return dt.year + (elapsed / year_length)
    #
    # def get_mag_field(self):
    #     lat, long, alt = self.to_global_position(self.initial_global_pos)
    #     date = self._datetime_to_decimal_year(datetime.today())
    #
    #     # Get magnetic field result
    #     result = geomag.GeoMag().calculate(lat, long, alt, date)
    #
    #     # Extract NED components (in nanoTesla)
    #     B_n = result.x * 1e-3  # North component
    #     B_e = result.y * 1e-3  # East component
    #     B_d = result.z * 1e-3  # Down component
    #
    #     # Combine into a vector
    #     b_n = np.array([B_n, B_e, B_d])
    #     print(b_n)
    #     input()
    #
    #     return b_n

    def nis(self):
        pass

    def nees(self):
        pass




