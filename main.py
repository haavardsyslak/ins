from mcap_protobuf.reader import read_protobuf_messages
import blueye.protocol as bp
import numpy as np
import ukfm
from ukfm.models import DvlMeasurement, DepthMeasurement, GnssMeasurement, ImuModel, Magnetometer
from scipy.spatial.transform import Rotation as Rot
import manifpy as manif
from google.protobuf.timestamp_pb2 import Timestamp
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove import FoxgloveLogger
import time
import pymap3d as pm
from utils import wrap_plus_minis_pi
from messages_pb2 import UkfState
from datetime import datetime
from sigma_points import SigmaPoints, SimplexSigmaPoints

global global_initial_pos


class McapProtobufReader:
    def __init__(self, filename):
        self.filename = filename
        self.msg_iter = read_protobuf_messages(filename)

    def get_next_message(self):
        # Use self.msg_iter to get the next message
        try:
            message = next(self.msg_iter)
            return message
        except StopIteration:
            return None

    def __iter__(self):
        return self.msg_iter


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
    global_pos = np.zeros(3)

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
                        global_pos[0] = gnss_sensor.global_position.latitude
                        global_pos[1] = gnss_sensor.global_position.longitude

            case "blueye.protocol.DepthTel":
                if not first_dpeth_msg_received:
                    first_dpeth_msg_received = True
                    depth = message.proto_msg.depth.value
                    pos[2] = depth
                    global_pos[2] = depth

        if first_dvl_msg_received and first_gnss_msg_received and first_dpeth_msg_received:
            return pos, velocity, heading, global_pos


def run_ukf(ukf, reader):
    logger = FoxgloveLogger("testing.mcap")
    t = reader.get_next_message().log_time
    n = 0
    while True:
        # time.sleep(0.001)
        message = reader.get_next_message()
        if message is None:
            break

        match message.topic:
            case "blueye.protocol.CalibratedImuTel":
                # messge.log_time
                gyro = message.proto_msg.imu.gyroscope
                accel = message.proto_msg.imu.accelerometer
                mag = message.proto_msg.imu.magnetometer
                u = np.array([gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z])
                dt = (message.log_time - t).total_seconds()
                t = message.log_time
                ukf.propagate(u, dt)
                topic = f"blueye.protocol.{message.proto_msg.__name__}"
                logger.publish(topic, message.proto_msg, message.log_time_ns)

                # n += 1
                # if n == 10:
                #     R_mag = np.eye(3) *  1e-1
                #     z = np.array([mag.x, mag.y, mag.z])
                #     measurement = Magnetometer(R_mag, z)
                #     ukf.update(measurement, dt)
                #     n = 0

            case "blueye.protocol.DvlVelocityTel":
                vel = message.proto_msg.dvl_velocity.velocity
                vel = np.array([vel.x, vel.y, vel.z])
                R_dvl = np.eye(3) * 2e-5
                # R_dvl[2] = 2e-3
                # R_dvl[2] = 2.5e-3
                measurement = DvlMeasurement(R_dvl, vel)
                ukf.update(measurement, dt)
                lat, lon, alt = ukf.x.to_global_position(initial_global_position)
                est_vel = ukf.x.extended_pose.linearVelocity()

                rot = Rot.from_matrix(ukf.x.extended_pose.rotation())
                location_fix = LocationFix(
                    timestamp=make_proto_timestamp(message.log_time_ns),
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    position_covariance=ukf.P[:3, :3].flatten().tolist(),
                    position_covariance_type=3,
                )
                topic = f"ukf.foxglove{location_fix.__name__}"
                logger.publish(topic, location_fix, message.log_time_ns, "foxglove.LocationFix")

                msg = make_proto_ukf_state(ukf.x)
                topic = f"custom.{msg.__name__}"
                logger.publish(topic, msg, message.log_time_ns)

                topic = f"blueye.protocol.{message.proto_msg.__name__}"
                logger.publish(topic, message.proto_msg, message.log_time_ns)

            case "blueye.protocol.PositionEstimateTel":
                msg = make_location_fix(message)
                topic = f"gnss.foxglove.{msg.__name__}"
                logger.publish(topic, msg, message.log_time_ns, "foxglove.LocationFix")
                heading = wrap_plus_minis_pi(message.proto_msg.position_estimate.heading)
                message.proto_msg.position_estimate.heading = np.rad2deg(heading)
                logger.publish(
                    f"blueye.protocol.{message.proto_msg.__name__}", message.proto_msg, message.log_time_ns)

                gnss_sensor = get_gnss_sensor(message.proto_msg)

            case "blueye.protocol.DepthTel":
                R_depth = 1e-6
                depth_meas = DepthMeasurement(R_depth)
                depth_meas.z = message.proto_msg.depth.value
                ukf.update(depth_meas, dt)
                topic = f"blueye.protocol.{message.proto_msg.__name__}"
                logger.publish(topic, message.proto_msg, message.log_time_ns)

    input("Enter to close logger")
    logger.close()


def make_proto_ukf_state(state):

    extended_pose = state.extended_pose
    g = state.g[-1]
    quat = extended_pose.coeffs()[3:7]

    rot = Rot.from_matrix(extended_pose.rotation())
    roll, pitch, yaw = rot.as_euler("XYZ", degrees=True)

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


def make_proto_timestamp(log_time_ns):
    seconds = log_time_ns // 1_000_000_000
    nanos = log_time_ns % 1_000_000_000
    return Timestamp(seconds=seconds, nanos=nanos)


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
    np.set_printoptions(precision=2, linewidth=999)
    global initial_global_pos
    # Create an instance of the reader
    reader = McapProtobufReader("mcap/log_square_05.04.mcap")
    target_start = datetime.strptime("2025-04-05 11:51:15.782911", "%Y-%m-%d %H:%M:%S.%f")
    while reader.get_next_message().log_time < target_start:
        continue
    # Read messages one by one
    pos, vel, heading, initial_global_position = get_initial_state(reader)
    pos[0] = 0.0
    pos[1] = 0.0

    heading = wrap_plus_minis_pi(heading)
    rot = Rot.from_euler("XYZ", [0, 0, heading])
    q = rot.as_quat(scalar_first=False)
    vel = rot.as_matrix() @ vel
    extended_pose = manif.SE_2_3(np.concatenate([pos, q, vel]))
    print(Rot.from_quat(q, scalar_first=True).as_euler("XYZ"))
    g = np.array([0.0, 0.0, -9.822])
    gyro_bias = np.array([0.0, 0.0, 0.00])
    x0 = ukfm.LieState(extended_pose, gyro_bias=gyro_bias, initial_global_pos=initial_global_position)
    P0 = np.eye(x0.dof())
    P0[0:3, 0:3] = 2.5 * np.eye(3)
    P0[3:6, 3:6] = 1e-6 * np.eye(3)
    P0[6:9, 6:9] = 1e-3 * np.eye(3)
    P0[9:12, 9:12] = 1e-4 * np.eye(3)
    P0[12:15, 12:15] = 1e-4 * np.eye(3)

    model = ImuModel(
        gyro_std=8e-2,
        gyro_bias_std=4e-9,
        gyro_bias_p=0.001,
        accel_std=1,
        accel_bias_std=0.0001,
        accel_bias_p=0.0001,
    )

    dim_x = x0.dof()  # State dimension
    dim_q = model.Q.shape[0]  # Process noise dimension

    points = SigmaPoints(dim_x, alpha=1e-3, beta=2, kappa=3 - dim_x)
    noise_points = SigmaPoints(dim_q, alpha=4e-3, beta=2, kappa=3 - dim_q)
    # points = SigmaPoints(dim_x, alpha=1e-2, beta=2, kappa=3-dim_x)
    # noise_points = SigmaPoints(dim_q, alpha=1e-4, beta=2, kappa=3-dim_q)

    ukf = ukfm.UKFM(dim_x, dim_q, points, noise_points, model, x0, P0)

    run_ukf(ukf, reader)

