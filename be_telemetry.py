import time
import logging
from foxglove import FoxgloveLogger
import blueye
from blueye.sdk import Drone
import blueye.protocol as bp


class DroneTelemetry:
    def __init__(self, filename: str):
        self.filename = filename
        self.drone = None
        self.telemetry_messages = []
        self._start_bridge()
        self.foxglove_logger = FoxgloveLogger(self.filename)

        # Setup the drone (but do not add callbacks yet)
        self._setup_drone()

        # Register topics now that server is up
        # self.foxglove_logger.register_blueye_descriptors()

        # Only NOW add telemetry callbacks
        self._setup_callbacks()

    def _start_bridge(self):
        logger = logging.getLogger("FoxgloveBridge")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s: [%(levelname)s] <%(name)s> %(message)s"))
        logger.addHandler(handler)
        logger.info("Starting Foxglove bridge")

        logger_sdk = logging.getLogger(blueye.sdk.__name__)
        logger_sdk.setLevel(logging.DEBUG)
        logger_sdk.addHandler(handler)

    def _setup_drone(self):
        self.drone = Drone(connect_as_observer=True)

    def _setup_callbacks(self):
        msgs = []
        self.drone.telemetry.set_msg_publish_frequency(bp.CalibratedImuTel, 100)
        msgs.append(bp.CalibratedImuTel)
        self.drone.telemetry.set_msg_publish_frequency(bp.Imu1Tel, 100)
        msgs.append(bp.Imu1Tel)
        self.drone.telemetry.set_msg_publish_frequency(bp.Imu2Tel, 50)
        msgs.append(bp.Imu2Tel)
        self.drone.telemetry.set_msg_publish_frequency(bp.DepthTel, 10)
        msgs.append(bp.DepthTel)
        self.drone.telemetry.set_msg_publish_frequency(bp.DvlVelocityTel, 25)
        msgs.append(bp.DvlVelocityTel)
        self.drone.telemetry.set_msg_publish_frequency(bp.PositionEstimateTel, 10)
        msgs.append(bp.PositionEstimateTel)

        self.drone.telemetry.add_msg_callback(msgs, self.parse_be_message, raw=True)
        self.telemetry_messages = [i.__name__ for i in msgs]

    def parse_be_message(self, payload_msg_name, data):
        proto_cls = getattr(blueye.protocol, payload_msg_name)._meta.pb
        msg = proto_cls()
        msg.ParseFromString(data)
        timestamp = time.time_ns()
        topic = f"blueye.protocol.{payload_msg_name}"

        self.foxglove_logger.publish(topic, msg, timestamp, topic)


# Example usage
if __name__ == "__main__":
    telem = DroneTelemetry("mau_asdf.mcap")

    while True:
        inn = input("Stop? [y/N]")
        if inn == "y":
            break
    telem.foxglove_logger.close()
