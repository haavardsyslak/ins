

# 2min DR UKF
- Good values for dataset "adis_mcap/log_dive_2min_2025-05-07 15:06:13.mcap"
- Initial conditions:     heading = np.deg2rad(-110)

```py
    P0[0:3, 0:3] = 2.5 * np.eye(3)
    P0[3:6, 3:6] = 1e-6 * np.eye(3)
    P0[6:9, 6:9] = 1e-3 * np.eye(3)
    P0[9:12, 9:12] = 1e-1 * np.eye(3)
    P0[12:15, 12:15] = 1e-1 * np.eye(3)

    model = ukfm.ImuModel(
        gyro_std=8.73e-2,          # Gyroscope output noise ≈ 0.05 deg/s → 8.73e-4 rad/s
        gyro_bias_std=9.7e-3,      # In-run bias stability ≈ 2 deg/hr → 9.7e-6 rad/s
        gyro_bias_p=0.00001,         # Gauss-Markov decay rate (correlation time ~1000 s)
        accel_std=5.88e-3,         # Accelerometer output noise ≈ 0.6 mg → 5.88e-3 m/s²
        accel_bias_std=3.5e-2,     # Accelerometer in-run bias ≈ 3.6 µg → 3.5e-5 m/s²
        # Gauss-Markov decay rate (correlation time ~1000 s)       # accel_bias_p=0.0000001,
        accel_bias_p=0.00001,
    )

    dim_x = x0.dof()  # State dimension
    dim_q = model.Q.shape[0]  # Process noise dimension

    points = SigmaPoints(dim_x, alpha=5e-2, beta=2, kappa=3 - dim_x)
    noise_points = SigmaPoints(dim_q, alpha=5e-5, beta=2, kappa=3 - dim_q)
```

# DR UKFM square


```py
P0 = np.eye(x0.dof())  
P0[0:3, 0:3] = 5 * np.eye(3)
# P0[2, 2] = 0.2**2
P0[3:6, 3:6] = 1e-2* np.eye(3)
P0[6:9, 6:9] = 5 * np.eye(3)
P0[9:12, 9:12] = 1e-6 * np.eye(3)
P0[12:15, 12:15] = 1e-6 * np.eye(3)
model = ImuModel(
        gyro_std=8.73e-2,          # Gyroscope output noise ≈ 0.05 deg/s → 8.73e-4 rad/s
        gyro_bias_std=9.7e-2,      # In-run bias stability ≈ 2 deg/hr → 9.7e-6 rad/s
        gyro_bias_p=0.0001,         # Gauss-Markov decay rate (correlation time ~1000 s)
        accel_std=5.88e-1,         # Accelerometer output noise ≈ 0.6 mg → 5.88e-3 m/s²
        accel_bias_std=3.5e-3,     # Accelerometer in-run bias ≈ 3.6 µg → 3.5e-5 m/s²
# Gauss-Markov decay rate (correlation time ~1000 s)       # accel_bias_p=0.0000001,
        accel_bias_p=0.0001,
        )

dim_x = x0.dof()  # State dimension
dim_q = model.Q.shape[0]  # Process noise dimension

points = SigmaPoints(dim_x, alpha=8e-2, beta=2.0, kappa=3-dim_x)
noise_points = SigmaPoints(dim_q, alpha=8e-5, beta=2.0, kappa=3-dim_q)


R_dvl = np.diag([0.25**2, 0.25**2, 0.3**2]) * 0.1
R_gnss = np.diag([2.5**2, 2.5**2])
R_depth = 0.01**2


```
 Good DR UKFM square
```py 
P0 = np.eye(x0.dof())  
P0[0:3, 0:3] = 5 * np.eye(3)
# P0[2, 2] = 0.2**2
P0[3:6, 3:6] = 1e-2* np.eye(3)
P0[6:9, 6:9] = 1e-2 * np.eye(3)
P0[9:12, 9:12] = 1e-6 * np.eye(3)
P0[12:15, 12:15] = 1e-6 * np.eye(3)
model = ukfm.ImuModel(
        gyro_std=8.73e-3,          # Gyroscope output noise ≈ 0.05 deg/s → 8.73e-4 rad/s
        gyro_bias_std=9.7e-6,      # In-run bias stability ≈ 2 deg/hr → 9.7e-6 rad/s
        gyro_bias_p=1e-3, #0.0001,         # Gauss-Markov decay rate (correlation time ~1000 s)
        accel_std=5.88e-3,         # (might be 5.88e-3) Accelerometer output noise ≈ 0.6 mg → 5.88e-3 m/s² 
        accel_bias_std=3.5e-5,     # Accelerometer in-run bias ≈ 3.6 µg → 3.5e-5 m/s²
# Gauss-Markov decay rate (correlation time ~1000 s)       # accel_bias_p=0.0000001,
        accel_bias_p=1e-1,
        )

dim_x = x0.dof()  # State dimension
dim_q = model.Q.shape[0]  # Process noise dimension

points = SigmaPoints(dim_x, alpha=1e-3, beta=2.0, kappa=3-dim_x)
noise_points = SigmaPoints(dim_q, alpha=6e-4, beta=2.0, kappa=0)

```


