import time
from typing import List
from blueye.sdk import Drone
import blueye.protocol as bp
import json

# from mcap_protobuf.writer import Writer
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
import threading
from queue import Queue

from schemas import imu_schema, dvl_schema, pos_estimate_schema, depth_schema

class McapLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, "wb")
        self.writer = Writer(self.file)
        self.writer.start()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._log_worker)
        self.queue = Queue()
        self.channels = {}
        self.schemas = {}

        self._initialize_mcap()

    def _initialize_mcap(self):
        """Register JSON schemas and channels for telemetry types."""
        # Define the JSON schema for the IMU message

        # Register the IMU schema
        imu_schema_id = self.writer.register_schema(
            name="Imu",
            encoding="jsonschema",
            data=json.dumps(imu_schema).encode(),
        )

        # Register the IMU channel
        imu_channel_id = self.writer.register_channel(
            schema_id=imu_schema_id, topic="imu", message_encoding="json"
        )

        dvl_schema_id = self.writer.register_schema(
            name="Dvl", encoding="jsonschema", data=json.dumps(dvl_schema).encode()
        )

        dvl_channel_id = self.writer.register_channel(
            schema_id=dvl_schema_id, topic="dvl", message_encoding="json"
        )

        pos_est_schema_id = self.writer.register_schema(
            name="Pos_estimate",
            encoding="jsonschema",
            data=json.dumps(pos_estimate_schema).encode(),
        )

        pos_est_channel_id = self.writer.register_channel(
            schema_id=pos_est_schema_id, topic="pos", message_encoding="json"
        )

        depth_schema_id = self.writer.register_schema(
            name="Depth", encoding="jsonschema", data=json.dumps(depth_schema).encode()
        )

        depth_channel_id = self.writer.register_channel(
            schema_id=depth_schema_id, topic="depth", message_encoding="json"
        )

        self.schemas["imu"] = imu_schema_id
        self.channels["imu"] = imu_channel_id
        self.schemas["dvl"] = dvl_schema_id
        self.channels["dvl"] = dvl_channel_id
        self.schemas["pos"] = pos_est_schema_id
        self.channels["pos"] = pos_est_channel_id
        self.schemas["depth"] = depth_schema_id
        self.channels["depth"] = depth_channel_id

    def log_message(self, topic: str, message: bytes):
        """Add a telemetry message to the queue."""
        self.queue.put((topic, message, int(time.time_ns())))

    def _log_worker(self):
        """Worker thread for writing messages to the MCAP file."""
        while self.running or not self.queue.empty():
            # try:
            topic, message, timestamp = self.queue.get()
            with self.lock:
                self.writer.add_message(
                    channel_id=self.channels[topic],
                    log_time=timestamp,
                    publish_time=timestamp,
                    data=message.encode(),
                    # data=json.dumps(message).encode("utf-8"),
                )
            # except Exception:
            # continue

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        with self.lock:
            self.writer.finish()
            self.file.close()


class DroneTelemetry:
    def __init__(self, filename: str):
        self.filename = filename
        self.logger = McapLogger(filename)
        self.drone = None
        self.callbacks = []
        self._setup()

    def start(self):
        self.logger.start()

    def stop(self):
        self.logger.stop()

    def _setup(self):
        self.drone = Drone()

        self.drone.telemetry.set_msg_publish_frequency(bp.CalibratedImuTel, 100)
        self.drone.telemetry.set_msg_publish_frequency(bp.DepthTel, 100)
        self.drone.telemetry.set_msg_publish_frequency(bp.DvlVelocityTel, 100)
        self.drone.telemetry.set_msg_publish_frequency(bp.PositionEstimateTel, 100)

        cb_imu = self.drone.telemetry.add_msg_callback([bp.CalibratedImuTel], self._callback_imu)
        self.callbacks.append(cb_imu)

        cb_depth = self.drone.telemetry.add_msg_callback([bp.DepthTel], self._callback_depth)
        self.callbacks.append(cb_depth)

        cb_dvl = self.drone.telemetry.add_msg_callback([bp.DvlVelocityTel], self._callback_dvl)
        self.callbacks.append(cb_dvl)

        cb_pos_estimate = self.drone.telemetry.add_msg_callback(
            [bp.PositionEstimateTel], self._callback_pos_estimate
        )
        self.callbacks.append(cb_pos_estimate)

    def _callback_imu(self, msg_type: str, msg: bp.CalibratedImuTel):
        imu = msg.imu
        imu_data = {
            "accelerometer": {
                "x": imu.accelerometer.x,
                "y": imu.accelerometer.y,
                "z": imu.accelerometer.z,
            },
            "gyroscope": {
                "x": imu.gyroscope.x,
                "y": imu.gyroscope.y,
                "z": imu.gyroscope.z,
            },
            "magnetometer": {
                "x": imu.magnetometer.x,
                "y": imu.magnetometer.y,
                "z": imu.magnetometer.z,
            },
            "temperature": imu.temperature,
        }
        self.logger.log_message("imu", bp.Imu.to_json(imu))

    def _callback_dvl(self, msg_type: str, msg: bp.DvlVelocityTel):
        dvl = msg.dvl_velocity
        dvl_data = {
                "status": dvl.status,
                "delta_time": dvl.delta_time,
                "fom": dvl.fom,
                "velocity": {
                    "x": dvl.velocity.x,
                    "y": dvl.velocity.y,
                    "z": dvl.velocity.z,
                },
                "is_water_tracking": dvl.is_water_tracking,
            }
        print(json.dumps(dvl_data))

        self.logger.log_message("dvl", json.dumps(dvl_data))

    def _callback_depth(self, msg_type: str, msg: bp.DepthTel):
        self.logger.log_message("depth", bp.Depth.to_json(msg.depth))

    def _callback_pos_estimate(self, msg_type: str, msg: bp.PositionEstimateTel):
        pos = msg.position_estimate
        pos_estimate = {
                    "northing": pos.northing,
                    "easting": pos.easting,
                    "heading": pos.heading,
                    "surge_rate": pos.surge_rate,
                    "sway_rate": pos.sway_rate,
                    "yaw_rate": pos.yaw_rate,
                    "ocean_current": pos.ocean_current,
                    "odometer": pos.odometer,
                    "is_valid": pos.is_valid,
                    "global_position": {
                        "latitude": pos.global_position.latitude,
                        "longitude": pos.global_position.longitude,
                        },
                    "speed_over_ground": pos.speed_over_ground,
                    "course_over_ground": pos.course_over_ground,
                    }
        self.logger.log_message("pos", json.dumps(pos_estimate))


if __name__ == "__main__":
    telem = DroneTelemetry("mcap_test.mcap")
    telem.start()

    input("Enter to stop")
    telem.stop()
