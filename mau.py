from pygeomag import geomag
from datetime import datetime
import numpy as np

from datetime import datetime

def datetime_to_decimal_year(dt: datetime) -> float:
    year_start = datetime(dt.year, 1, 1)
    next_year = datetime(dt.year + 1, 1, 1)
    year_length = (next_year - year_start).total_seconds()
    elapsed = (dt - year_start).total_seconds()
    return dt.year + (elapsed / year_length)

# Your location and date
latitude = 63.43
longitude = 10.39
altitude = 0  # meters above sea level
date = datetime_to_decimal_year(datetime.today())

# Get magnetic field result
result = geomag.GeoMag().calculate(latitude, longitude, altitude, date)

# Extract NED components (in nanoTesla)
B_n = result.x * 1e-9  # North component
B_e = result.y * 1e-9  # East component
B_d = result.z * 1e-9  # Down component

# Combine into a vector
b_n = np.array([B_n, B_e, B_d])

print("Magnetic field (b_n) in Tesla:", b_n)

