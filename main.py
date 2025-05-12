import blueye.protocol as bp
import numpy as np
import ukfm
import esekf
from orientation import RotationQuaterion
from scipy.spatial.transform import Rotation as Rot
import manifpy as manif
from google.protobuf.timestamp_pb2 import Timestamp
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_wrapper import FoxgloveLogger
import time
import pymap3d as pm
from utils import wrap_plus_minis_pi
from messages_pb2 import UkfState
from datetime import datetime
from sigma_points import SigmaPoints, SimplexSigmaPoints
from foxglove_wrapper import McapProtobufReader
from filter_interface import KFRunner, GlobalPos

global global_initial_pos


def get_gnss_sensor(msg: bp.PositionEstimateTel):
    GNSS_SENSOR_ID = 4
    for sensor in msg.position_estimate.navigation_sensors:
        if sensor.sensor_id == GNSS_SENSOR_ID:
            return sensor
    return None


def get_initial_state(reader):
    """Get the first measurements and use them as the initial state of the filter"""
    first_dvl_msg_received = False
    first_gnss_msg_received = False
    first_dpeth_msg_received = False

    velocity = np.zeros(3)
    pos = np.zeros(3)
    heading = 0.0
    global_pos = GlobalPos(0.0, 0.0)

    for message in reader:
        match message.topic:
            case "blueye.protocol.DvlVelocityTel":
                if not first_dvl_msg_received:
                    velocity_msg = message.proto_msg.dvl_velocity.velocity
                    velocity = np.array([velocity_msg.x, velocity_msg.y, velocity_msg.z])
                    first_dvl_msg_received = True

            case "blueye.protocol.PositionEstimateTel":
                heading = message.proto_msg.position_estimate.heading
                if not first_gnss_msg_received:
                    gnss_sensor = get_gnss_sensor(message.proto_msg)
                    if gnss_sensor.is_valid:
                        first_gnss_msg_received = True
                        pos[0] = gnss_sensor.northing
                        pos[1] = gnss_sensor.easting
                        global_pos = GlobalPos(gnss_sensor.global_position.latitude,
                                               gnss_sensor.global_position.longitude)

            case "blueye.protocol.DepthTel":
                if not first_dpeth_msg_received:
                    first_dpeth_msg_received = True
                    depth = message.proto_msg.depth.value
                    pos[2] = depth
                    global_pos.alt = depth

        if first_dvl_msg_received and first_gnss_msg_received and first_dpeth_msg_received:
            return pos, velocity, heading, global_pos


def run_esekf():
    np.set_printoptions(precision=4, linewidth=999)
    global initial_global_pos
    # Create an instance of the reader
    reader = McapProtobufReader("adis_mcap/log_auto_square_2025-05-07 14:55:13.mcap")
    # reader = McapProtobufReader("adis_mcap/gnss_challenging_env_2025-05-07 15:32:53.mcap")
    # reader = McapProtobufReader("adis_mcap/log_dive_2min_2025-05-07 15:06:13.mcap")

    pos, vel, heading, initial_global_position = get_initial_state(reader)
    pos[0] = 0.0
    pos[1] = 0.0

    # heading = np.deg2rad(50)
    heading = wrap_plus_minis_pi(heading)
    rot = Rot.from_euler("xyz", [0, 0, heading])
    q_scipy = rot.as_quat()
    q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    vel = rot.as_matrix() @ vel
    # extended_pose = manif.SE_2_3(np.concatenate([pos, q_scipy, vel]))
    print(Rot.from_quat(q, scalar_first=True).as_euler("xyz", degrees=True))
    g = np.array([0.0, 0.0, -9.822])
    gyro_bias = np.array([0.0, 0.0, 0.004])
    accel_bias = np.array([0.0004, 0.001, -0.004])
    q = rot.as_quat()
    ori = manif.SO3(q)
    print("q as euler: ", Rot.from_matrix(ori.rotation()).as_euler("xyz", degrees=True))
    x0 = esekf.NominalState(pos=pos, vel=vel, ori=ori, gyro_bias=gyro_bias, acc_bias=accel_bias)
    P0 = np.eye(x0.dof())
    P0[0:3, 0:3] = 5 * np.eye(3)
    # P0[2, 2] = 0.2**2
    P0[3:6, 3:6] = 1e-2 * np.eye(3)
    P0[6:9, 6:9] = 1e-2 * np.eye(3)
    P0[9:12, 9:12] = 1e-4 * np.eye(3)
    P0[12:15, 12:15] = 1e-4 * np.eye(3)

    model = esekf.models.ImuModel(
        gyro_std=8.73e-3,          # Gyroscope output noise ≈ 0.05 deg/s → 8.73e-4 rad/s
        gyro_bias_std=9.7e-9,      # In-run bias stability ≈ 2 deg/hr → 9.7e-6 rad/s
        gyro_bias_p=0.001,         # Gauss-Markov decay rate (correlation time ~1000 s)
        accel_std=5.88e-1,         # Accelerometer output noise ≈ 0.6 mg → 5.88e-3 m/s²term
        accel_bias_std=3.5e-9,     # Accelerometer in-run bias ≈ 3.6 µg → 3.5e-5 m/s²
        # Gauss-Markov decay9rate (correlation time ~1000 s)       # accel_bias_p=0.0000001,
        accel_bias_p=0.001,
    )

    # model = esekf.models.ImuModel(
    #     gyro_std=8e-2,
    #     gyro_bias_std=4e-4,
    #     gyro_bias_p=0.000001,
    #     accel_std=,
    #     accel_bias_std=0.0001,
    #     accel_bias_p=0.00000001,
    # )
    kf = esekf.ESEFK(x0, P0, model)

    logger = FoxgloveLogger("01testing_esekf.mcap", stream=False)
    runner = KFRunner(kf, logger, initial_global_position)
    runner.run(reader)


def run_ukfm():
    np.set_printoptions(precision=4, linewidth=999)
    global initial_global_pos
    # Create an instance of the reader
    reader = McapProtobufReader("adis_mcap/log_auto_square_2025-05-07 14:55:13.mcap")
    # reader = McapProtobufReader("adis_mcap/gnss_challenging_env_2025-05-07 15:32:53.mcap")
    # reader = McapProtobufReader("adis_mcap/log_dive_2min_2025-05-07 15:06:13.mcap")

    pos, vel, heading, initial_global_position = get_initial_state(reader)
    pos[0] = 0.0
    pos[1] = 0.0

    # heading = np.deg2rad(-110)
    heading = wrap_plus_minis_pi(heading)
    rot = Rot.from_euler("xyz", [0, 0, heading])
    q_scipy = rot.as_quat()
    q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    vel = rot.as_matrix() @ vel
    extended_pose = manif.SE_2_3(np.concatenate([pos, q_scipy, vel]))
    print(Rot.from_quat(q, scalar_first=True).as_euler("xyz", degrees=True))
    g = np.array([0.0, 0.0, -9.822])
    gyro_bias = np.array([0.0, 0.0, 0.004])
    accel_bias = np.array([0.0004, 0.001, -0.004])
    x0 = ukfm.LieState(extended_pose, gyro_bias=gyro_bias, acc_bias=accel_bias)
    P0 = np.eye(x0.dof())
    P0[0:3, 0:3] = 5 * np.eye(3)
    # P0[2, 2] = 0.2**2
    P0[3:6, 3:6] = 1e-2 * np.eye(3)
    P0[6:9, 6:9] = 5 * np.eye(3)
    P0[9:12, 9:12] = 1e-6 * np.eye(3)
    P0[12:15, 12:15] = 1e-6 * np.eye(3)

    model = ukfm.models.ImuModel(
        gyro_std=8.73e-2,          # Gyroscope output noise ≈ 0.05 deg/s → 8.73e-4 rad/s
        gyro_bias_std=9.7e-2,      # In-run bias stability ≈ 2 deg/hr → 9.7e-6 rad/s
        gyro_bias_p=0.0001,         # Gauss-Markov decay rate (correlation time ~1000 s)
        accel_std=5.88,         # Accelerometer output noise ≈ 0.6 mg → 5.88e-3 m/s²
        accel_bias_std=3.5e-3,     # Accelerometer in-run bias ≈ 3.6 µg → 3.5e-5 m/s²
        # Gauss-Markov decay rate (correlation time ~1000 s)       # accel_bias_p=0.0000001,
        accel_bias_p=0.0001,
    )

    dim_x = x0.dof()  # State dimension
    dim_q = model.Q.shape[0]  # Process noise dimension

    points = SigmaPoints(dim_x, alpha=8e-2, beta=2.0, kappa=3 - dim_x)
    noise_points = SigmaPoints(dim_q, alpha=8e-5, beta=2.0, kappa=3 - dim_q)
    # points = SigmaPoints(dim_x, alpha=1e-2, beta=2, kappa=3-dim_x)
    # noise_points = SigmaPoints(dim_q, alpha=1e-4, beta=2, kappa=3-dim_q)

    ukf = ukfm.UKFM(dim_x, dim_q, points, noise_points, model, x0, P0)

    logger = FoxgloveLogger("01testing.mcap", stream=False)
    runner = KFRunner(ukf, logger, initial_global_position)
    runner.run(reader)

    # run_ukf(ukf, reader)


if __name__ == "__main__":
    run_esekf()

