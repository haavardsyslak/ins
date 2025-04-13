from mcap_protobuf.reader import read_protobuf_messages
import blueye.protocol as bp
import numpy as np
import ukfm
from ukfm.models import DvlMeasurement, DepthMeasurement
from scipy.spatial.transform import Rotation as Rot
import manifpy as manif
from google.protobuf.timestamp_pb2 import Timestamp
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix

from foxglove import FoxgloveLogger
import time

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


def initial_state(reader):
    first_dvl_msg_received = False
    first_gnss_msg_received = False
    first_dpeth_msg_received = False

    velocity = np.zeros(3)
    pos = np.zeros(3)
    heading = 0.0

    for message in reader:
        match message.topic:
            case "blueye.protocol.DvlVelocityTel":
                if not first_dvl_msg_received:
                    velocity = message.proto_msg.dvl_velocity.velocity
                    vel = np.array([velocity.x, velocity.y, velocity.z])
                    first_dvl_msg_received = True

            case "blueye.protocol.PositionEstimateTel":
                heading = message.proto_msg.position_estimate.heading
                if not first_gnss_msg_received:
                    gnss_sensor = get_gnss_sensor(message.proto_msg)
                    if gnss_sensor.is_valid:
                        first_gnss_msg_received = True
                        pos[0] = gnss_sensor.northing
                        pos[1] = gnss_sensor.easting

            case "blueye.protocol.DepthTel":
                if not first_dpeth_msg_received:
                    first_dpeth_msg_received = True
                    pos[1] = message.proto_msg.depth.value

        if first_dvl_msg_received and first_gnss_msg_received and first_dpeth_msg_received:
            return pos, vel, heading


def run_ukf(ukf, reader):
    print("asdfsdf")
    try: 
        logger = FoxgloveLogger("testing.mcap")
        print("asdfsdf")
        t = reader.get_next_message().log_time
        while True:
            time.sleep(0.001)
            message = reader.get_next_message()

            dt = 1 / 80
            match message.topic:
                case "blueye.protocol.CalibratedImuTel":
                    # messge.log_time
                    gyro = message.proto_msg.imu.gyroscope
                    accel = message.proto_msg.imu.accelerometer
                    u = np.array([gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z])
                    # ukf.propagate(u, dt)

                case "blueye.protocol.DvlVelocityTel":
                    vel = message.proto_msg.dvl_velocity.velocity
                    vel = np.array([vel.x, vel.y, vel.z])
                    measurement = DvlMeasurement(vel)

                case "blueye.protocol.PositionEstimateTel":
                    msg = make_location_fix(message)
                    print("publishing")
                    topic = f"gnss.foxglove.{msg.__name__}"
                    logger.publish(topic, msg, message.log_time_ns, "foxglove.LocationFix")

                case "blueye.protocol.DepthTel":
                    pass
    finally:
        logger.close()


def make_location_fix(message: bp.PositionEstimateTel):
    log_time_ns = message.log_time_ns
    seconds = log_time_ns // 1_000_000_000
    nanos = log_time_ns % 1_000_000_000
    proto_timestamp = Timestamp(seconds=seconds, nanos=nanos)
    global_position = message.proto_msg.position_estimate.global_position

    return LocationFix(
        timestamp=proto_timestamp,
        latitude=global_position.latitude,
        longitude=global_position.longitude,
        altitude=0.0,
    )


if __name__ == "__main__":
    # Create an instance of the reader
    reader = McapProtobufReader("mcap/log_square_05.04.mcap")
    # Read messages one by one
    pos, vel, heading = initial_state(reader)
    q = Rot.from_euler("XYZ", [0, 0, heading]).as_quat(scalar_first=True)
    extended_pose = manif.SE_2_3(np.concatenate([pos, q, vel]))
    g = np.array([0.0, 0.0, 9.822])
    initial_state = ukfm.LieState(extended_pose, g=g)
    P0 = np.eye(initial_state.dof()) * 1e-3

    ukf = ukfm.make_ukf(initial_state, P0)
    run_ukf(ukf, reader)


