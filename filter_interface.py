from foxglove_wrapper import FoxgloveLogger, McapProtobufReader
import numpy as np
from dataclasses import dataclass, field
import pymap3d as pm
import utils
from typing import Self
import math
from tqdm import tqdm

# from models import DvlMeasurement, DepthMeasurement, GnssMeasurement
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from measurement_models import DvlMeasurement, GnssMeasurement, DepthMeasurement
from google.protobuf.timestamp_pb2 import Timestamp


@dataclass
class GlobalPos:
    lat: float
    lon: float
    alt: float = field(default_factory=lambda: 0.0)
    ned: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_ned(self, initial_global_pos: Self):
        pos = pm.geodetic2ned(
            self.lat,
            self.lon,
            self.alt,
            initial_global_pos.lat,
            initial_global_pos.lon,
            initial_global_pos.alt,
        )
        return np.array([pos[0], pos[1], pos[2]])

    def from_ned(self, ned: np.ndarray):
        lat, long, alt = pm.ned2geodetic(ned[0], ned[1], ned[2], self.lat, self.lon, self.alt)
        return GlobalPos(lat=lat, lon=long, alt=alt)


class KFRunner:
    def __init__(self, kf, logger, initial_global_pos: GlobalPos):
        self.kf = kf
        self.logger = logger
        self.last_timestamp = None
        self.initial_global_pos = initial_global_pos
        self.omega = np.zeros(3)
        self.elapsed_time = 0.0

    def handle_message(self, message):
        topic = message.topic
        match topic:
            case "blueye.protocol.Imu2Tel":
                dt = self._compute_dt(message.log_time)
                u = self._parse_imu(message)
                self.elapsed_time += dt
                self.omega = u[:3]
                self.kf.propagate(u, dt)
                self._publish_pose(message.log_time_ns)
                self._publish_raw(message)

            case "blueye.protocol.DvlVelocityTel":
                vel = self._parse_dvl(message)
                fom = message.proto_msg.dvl_velocity.fom
                if fom > 1:
                    fom = 0.02
                R_dvl = np.eye(3) * fom
                measurement = DvlMeasurement(R_dvl, vel)
                self.kf.update(measurement, 0.0)
                self._publish_state(message.log_time_ns)
                self._publish_raw(message)

            case "blueye.protocol.PositionEstimateTel":
                message = self._parse_drone_pos_estimate(message)
                gnss_sensor = self._parse_gnss(message)
                self._publish_raw(message)

                if gnss_sensor.is_valid:

                    if gnss_sensor is not None:
                        gnss_pos = GlobalPos(
                            gnss_sensor.global_position.latitude, gnss_sensor.global_position.longitude
                        )
                        self._publish_position(gnss_pos, "gnss.LocationFix", message.log_time_ns)
                        z = gnss_pos.to_ned(self.initial_global_pos)[:2]
                        R_gnss = np.diag([gnss_sensor.std, gnss_sensor.std])
                        measurement = GnssMeasurement(R_gnss, z)
                        if self.elapsed_time < 100:
                            self.kf.update(measurement, 0.0)

            case "blueye.protocol.DepthTel":
                self._publish_raw(message)
                depth = self._parse_depth(message)
                z = DepthMeasurement(0.05**2, depth)
                self.kf.update(z, 0.0)

    def _publish_pose(self, log_time):
        q = self.kf.x.q
        pos = self.kf.x.position

        tf = FrameTransform(
            timestamp=utils.make_proto_timestamp(log_time),
            parent_frame_id="ned",
            child_frame_id="rov",
            translation=Vector3(x=pos[0], y=pos[1], z=pos[2]),
            rotation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        )

        self.logger.publish("ukf.Pose", tf, log_time, "foxglove.FrameTransform")

    def _publish_state(self, log_time):
        msg = self.kf.to_proto_msg()
        self.logger.publish("blueye.protocol.UKFState", msg, log_time)

        global_pos = self.initial_global_pos.from_ned(self.kf.x.position)
        self._publish_position(global_pos, "ufk.locatinFix", log_time, cov=self.kf.P)

    def _publish_position(self, pos: GlobalPos, topic: str, log_time, cov=None):
        location_fix_kwargs = dict(
            timestamp=utils.make_proto_timestamp(log_time),
            latitude=pos.lat,
            longitude=pos.lon,
            altitude=pos.alt,
        )

        if cov is not None:
            location_fix_kwargs.update(
                position_covariance=cov[:3, :3].flatten().tolist(),
                position_covariance_type=3,
            )

        location_fix = LocationFix(**location_fix_kwargs)

        self.logger.publish(topic, location_fix, log_time, "foxglove.LocationFix")

    def _parse_imu(self, message):
        imu = message.proto_msg.imu
        gyro = imu.gyroscope
        accel = imu.accelerometer
        u = np.array([gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z])
        return u

    def _parse_dvl(self, message):
        vel = message.proto_msg.dvl_velocity.velocity
        return np.array([vel.x, vel.y, vel.z])

    def _parse_gnss(self, message):
        GNSS_SENSOR_ID = 4
        gnss_sensor = None
        for sensor in message.proto_msg.position_estimate.navigation_sensors:
            if sensor.sensor_id == GNSS_SENSOR_ID:
                gnss_sensor = sensor
        return gnss_sensor

    def _parse_drone_pos_estimate(self, message):
        heading = utils.wrap_plus_minis_pi(message.proto_msg.position_estimate.heading)
        cog = utils.wrap_plus_minis_pi(message.proto_msg.position_estimate.course_over_ground)
        message.proto_msg.position_estimate.heading = np.rad2deg(heading)
        message.proto_msg.position_estimate.course_over_ground = np.rad2deg(cog)
        return message

    def _parse_depth(self, message):
        return message.proto_msg.depth.value

    def _publish_raw(self, message):
        topic = f"blueye.protocol.{message.proto_msg.__name__}"
        self.logger.publish(topic, message.proto_msg, message.log_time_ns)

    def _compute_dt(self, timestamp):
        if self.last_timestamp is None:
            dt = 1.0 / 50.0
        else:
            dt = (timestamp - self.last_timestamp).total_seconds()
        self.last_timestamp = timestamp
        return dt

    def _publish_reference_frame(self, log_time):

        tf = FrameTransform(
            timestamp=utils.make_proto_timestamp(log_time),
            parent_frame_id="world",
            child_frame_id="ned",
            translation=Vector3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(x=1 / math.sqrt(2), y=1 / math.sqrt(2), z=0.0, w=0.0),
        )

        self.logger.publish("ned", tf, log_time, "foxglove.FrameTransform")

    def run(self, reader: McapProtobufReader):
        msg = reader.get_next_message()
        self._publish_reference_frame(msg.log_time_ns)

        for message in tqdm(reader):
            self.handle_message(message)

        self.logger.close()
