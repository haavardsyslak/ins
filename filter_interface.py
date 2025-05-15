from foxglove_wrapper import FoxgloveLogger, McapProtobufReader
import numpy as np
from dataclasses import dataclass, field
import pymap3d as pm
import utils
from typing import Self
import math
from tqdm import tqdm
import time
from tuning import ESEKFTuning, UKFMTuning, QUKFTuning
import statistics

# from models import DvlMeasurement, DepthMeasurement, GnssMeasurement
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from measurement_models import DvlMeasurement, GnssMeasurement, DepthMeasurement
from google.protobuf.timestamp_pb2 import Timestamp
from messages_pb2 import NIS, NEES, PositionError


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
            -self.alt,
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
        self.last_gnss_time = 0.0
        self.last_depth = 0.0
        if "esekf" in str(kf.__class__):
            self.tuning = ESEKFTuning()
            self.prefix = "esekf"
        elif "ukfm" in str(kf.__class__):
            self.tuning = UKFMTuning()
            self.prefix = "ukfm"
        else:
            self.tuning = QUKFTuning()
            self.prefix = "qukf"
        self.propagation_times = []
        self.update_times = []
        self.total_messages = 0
        self.propagation_count = 0
        self.update_count = 0

    def handle_message(self, message):
        topic = message.topic
        match topic:
            case "blueye.protocol.Imu2Tel":
                dt = self._compute_dt(message.log_time)
                u = self._parse_imu(message)
                self.elapsed_time += dt
                self.last_gnss_time += dt
                self.omega = u[:3]
                start = time.perf_counter()
                self.kf.propagate(u, dt)
                duration = time.perf_counter() - start
                self.propagation_times.append(duration)
                self.propagation_count += 1
                self._publish_pose(
                    message.log_time_ns, f"{self.prefix}.Pose", self.kf.x.position, ""
                )
                # self._publish_raw(message)
                self._publish_state(message.log_time_ns)

            case "blueye.protocol.DvlVelocityTel":
                vel = self._parse_dvl(message)
                fom = message.proto_msg.dvl_velocity.fom
                R_dvl = self.tuning.R_dvl(fom)
                measurement = DvlMeasurement(R_dvl, vel)
                start = time.perf_counter()
                self.kf.update(measurement, 0.0)
                duration = time.perf_counter() - start
                self.update_times.append(duration)
                self.update_count += 1
                self._publish_state(message.log_time_ns)
                # self._publish_raw(message)

                nis = self.kf.nis()
                self._publish_nis(f"{self.prefix}.Dvl.Nis", nis, message.log_time_ns)

            case "blueye.protocol.PositionEstimateTel":
                message = self._parse_drone_pos_estimate(message)
                gnss_sensor = self._parse_gnss(message)
                # self._publish_raw(message)

                if gnss_sensor.is_valid:

                    if gnss_sensor is not None:
                        gnss_pos = GlobalPos(
                            gnss_sensor.global_position.latitude,
                            gnss_sensor.global_position.longitude,
                            self.last_depth,
                        )
                        ned_pos = gnss_pos.to_ned(self.initial_global_pos)
                        self._publish_pose(
                            message.log_time_ns, f"{self.prefix}.gnss.Pose", ned_pos, "gnss"
                        )
                        # self._publ.......ish_position(gnss_pos, "gnss.LocationFix", message.log_time_ns)
                        nees = self.kf.nees_pos(ned_pos)
                        self._publish_nees(f"{self.prefix}.NEES pos", nees, message.log_time_ns)
                        error = self.kf.x.position - ned_pos
                        z = ned_pos[:2]
                        if self.elapsed_time < 100:
                            R_gnss = self.tuning.R_gnss()
                            measurement = GnssMeasurement(R_gnss, z)
                            start = time.perf_counter()
                            self.kf.update(measurement, 0.0)
                            duration = time.perf_counter() - start
                            self.update_times.append(duration)
                            self.update_count += 1
                        elif self.last_gnss_time > 1:
                            self.last_gnss_time = 0
                            R_gnss = np.diag([500, 500])
                            measurement = GnssMeasurement(R_gnss, z)
                            # self.kf.update(measurement, 0.0)

            case "blueye.protocol.DepthTel":
                # self._publish_raw(message)
                depth = self._parse_depth(message)
                self.last_depth = depth
                R_depth = self.tuning.R_depth()
                # R_depth = 0.01**2 * np.eye(1)
                # R_depth = 1e-9 * np.eye(1)
                measurement = DepthMeasurement(R_depth, depth)
                start = time.perf_counter()
                self.kf.update(measurement, 0.0)
                duration = time.perf_counter() - start
                self.update_times.append(duration)
                self.update_count += 1
                nis = self.kf.nis()
                self._publish_nis(f"{self.prefix}.Depth.Nis", nis, message.log_time_ns)

    def _publish_pos_error(self, log_time, error):
        msg = PositionError(x=error[0], y=error[1], z=0)
        self.logger.publish("ukf.PosError", msg, log_time)

    def _publish_pose(self, log_time, topic, pos, frame):
        q = self.kf.x.q
        if frame == "":
            frame = self.prefix

        tf = FrameTransform(
            timestamp=utils.make_proto_timestamp(log_time),
            parent_frame_id="ned",
            child_frame_id=frame,
            translation=Vector3(x=pos[0], y=pos[1], z=pos[2]),
            rotation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        )

        self.logger.publish(topic, tf, log_time, "foxglove.FrameTransform")

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

    def _publish_state(self, log_time):
        msg = self.kf.to_proto_msg()
        self.logger.publish(f"{self.prefix}.State", msg, log_time)

        global_pos = self.initial_global_pos.from_ned(self.kf.x.position)
        self._publish_position(global_pos, f"{self.prefix}.locatinFix", log_time, cov=self.kf.P)

    def _publish_position(self, pos: GlobalPos, topic: str, log_time, cov=None):
        location_fix_kwargs = dict(
            timestamp=utils.make_proto_timestamp(log_time),
            latitude=pos.lat,
            longitude=pos.lon,
            altitude=pos.alt,
        )

        if self.prefix == "qukf":
            s = 3
            e = 6
        else:
            s = 0
            e = 3
        if cov is not None:
            location_fix_kwargs.update(
                position_covariance=cov[s:e, s:e].flatten().tolist(),
                position_covariance_type=3,
            )

        location_fix = LocationFix(**location_fix_kwargs)

        self.logger.publish(topic, location_fix, log_time, "foxglove.LocationFix")

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

    def _publish_nis(self, topic, nis, timestamp):
        msg = NIS(value=nis)
        self.logger.publish(topic, msg, timestamp)

    def _publish_nees(self, topic, nees, timestamp):
        msg = NEES(value=nees)
        self.logger.publish(topic, msg, timestamp)

    def _print_performance_summary(self):
        import statistics

        def summary(label, times):
            if not times:
                return f"{label}: No data"
            mean = statistics.mean(times)
            stddev = statistics.stdev(times) if len(times) > 1 else 0.0
            mean_ms = mean * 1e3
            stddev_ms = stddev * 1e3
            return (
                f"{label}:\n"
                f"  mean     = {mean_ms:.3f} ms\n"
                f"  max      = {1e3 * max(times):.3f} ms\n"
                f"  min      = {1e3 * min(times):.3f} ms\n"
                f"  stddev   = {stddev_ms:.3f} ms\n"
                f"  95% CI   ≈ {mean_ms:.3f} ± {2 * stddev_ms:.3f} ms\n"
                f"  count    = {len(times)}"
            )

        print(f"\n=== {self.prefix.upper()} Performance Metrics ===")
        print(f"Total messages handled: {self.total_messages}")
        print(f"Propagations: {self.propagation_count}")
        print(f"Updates: {self.update_count}")
        print(summary("Propagation time", self.propagation_times))
        print(summary("Update time", self.update_times))
        print("=========================================")

    def run(self, reader: McapProtobufReader):
        msg = reader.get_next_message()
        # msg = reader.get_next_message()
        self._publish_state(msg.log_time_ns)
        self._publish_reference_frame(msg.log_time_ns)

        for message in tqdm(reader):
            self.handle_message(message)

        self._print_performance_summary()
