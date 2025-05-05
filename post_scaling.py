from mcap_protobuf.reader import read_protobuf_messages
from foxglove import FoxgloveLogger
from tqdm import tqdm
import numpy as np


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
        return self

    def __next__(self):
        return next(self.msg_iter)


accel_scale = 1#1e9 / 262_144_000.0
gyro_scale = np.deg2rad(1.0e9 * (1.0 / 10_485_760.0))

if __name__ == "__main__":
    mcap_reader = McapProtobufReader("./adis_mcap/log_grid_01_05.mcap")
    logger = FoxgloveLogger("./adis_mcap/log_grid_01_05_scaled.mcap")

    for message in tqdm(mcap_reader):

        if message.topic == "blueye.protocol.Imu2Tel":
            message.proto_msg.imu.accelerometer.x *= accel_scale
            message.proto_msg.imu.accelerometer.y *= accel_scale
            message.proto_msg.imu.accelerometer.z *= accel_scale
            message.proto_msg.imu.gyroscope.x *= gyro_scale
            message.proto_msg.imu.gyroscope.y *= gyro_scale
            message.proto_msg.imu.gyroscope.z *= gyro_scale
            

        logger.publish(message.topic, message.proto_msg, message.log_time_ns)

    logger.close()


