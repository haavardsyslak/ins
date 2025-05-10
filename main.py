import blueye.protocol as bp
import numpy as np
import ukfm
from ukfm.models import ImuModel
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




def make_proto_ukf_state(state):

    extended_pose = state.extended_pose
    g = state.g[-1]
    quat = extended_pose.coeffs()[3:7]

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
        velocity_x=extended_pose.vx(),
        velocity_y=extended_pose.vy(),
        velocity_z=extended_pose.vz(),
        heading=yaw,
        g=g,
        roll=roll,
        pitch=pitch,
        gyro_bias_x=state.gyro_bias[0],
        gyro_bias_y=state.gyro_bias[1],
        gyro_bias_z=state.gyro_bias[2],
    )


def make_location_fix(message: bp.PositionEstimateTel):
    proto_timestamp = make_proto_timestamp(message.log_time_ns)
    global_position = message.proto_msg.position_estimate.global_position

    return LocationFix(
        timestamp=proto_timestamp,
        latitude=global_position.latitude,
        longitude=global_position.longitude,
        altitude=0.0,
    )


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=999)
    global initial_global_pos
    # Create an instance of the reader
    reader = McapProtobufReader("adis_mcap/log_auto_square_2025-05-07 14:55:13.mcap")
    # reader = McapProtobufReader("adis_mcap/gnss_challenging_env_2025-05-07 15:32:53.mcap")
    # reader = McapProtobufReader("adis_mcap/log_dive_2min_2025-05-07 15:06:13.mcap")


    # target_start = datetime.strptime("2025-04-05 11:51:15.782911", "%Y-%m-%d %H:%M:%S.%f")
    # while reader.get_next_message().log_time < target_start:
    #     continue
    # Read messages one by one
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
    P0[2, 2] = 0.2**2
    P0[3:6, 3:6] = 1e-2* np.eye(3)
    P0[6:9, 6:9] = 0.15**2 * np.eye(3)
    P0[9:12, 9:12] = 1e-6 * np.eye(3)
    P0[12:15, 12:15] = 1e-6 * np.eye(3)

    model = ImuModel(
        gyro_std=8.73e-4,          # Gyroscope output noise ≈ 0.05 deg/s → 8.73e-4 rad/s
        gyro_bias_std=9.7e-2,      # In-run bias stability ≈ 2 deg/hr → 9.7e-6 rad/s
        gyro_bias_p=0.0001,         # Gauss-Markov decay rate (correlation time ~1000 s)
        accel_std=5.88e-5,         # Accelerometer output noise ≈ 0.6 mg → 5.88e-3 m/s²
        accel_bias_std=3.5e-3,     # Accelerometer in-run bias ≈ 3.6 µg → 3.5e-5 m/s²
        # Gauss-Markov decay rate (correlation time ~1000 s)       # accel_bias_p=0.0000001,
        accel_bias_p=0.0001,
    )

    dim_x = x0.dof()  # State dimension
    dim_q = model.Q.shape[0]  # Process noise dimension

    points = SigmaPoints(dim_x, alpha=8e-1, beta=2, kappa=3-dim_x)
    noise_points = SigmaPoints(dim_q, alpha=5e-5, beta=2, kappa=3-dim_q)
    # points = SigmaPoints(dim_x, alpha=1e-2, beta=2, kappa=3-dim_x)
    # noise_points = SigmaPoints(dim_q, alpha=1e-4, beta=2, kappa=3-dim_q)

    ukf = ukfm.UKFM(dim_x, dim_q, points, noise_points, model, x0, P0)

    logger = FoxgloveLogger("01testing.mcap", stream=False)
    runner = KFRunner(ukf, logger, initial_global_position)
    runner.run(reader)


    # run_ukf(ukf, reader)

