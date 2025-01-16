import time
from blueye.sdk import Drone
import blueye.protocol as bp


def callback_imu():
    pass


def callback_imu():
    pass


def callback_gnss():
    pass


def add_callbacks():
    pass


def setup_telemetry():
    drone = Drone()
    drone.connect()
    drone.telemetry.set_msg_publish_frequency(bp.CalibratedImuTel, 100)
    drone.telemetry.set_msg_publish_frequency(bp.DepthTel, 100)
    drone.telemetry.set_msg_publish_frequency(bp.DvlVelocityTel, 100)
    drone.telemetry.set_msg_publish_frequency(bp.DepthTel, 100)
    # drone.telemetry.set_msg_publish_frequency(bp.)

    # cb_raw = my_drone.telemetry.add_msg_callback([bp.Imu1Tel, bp.Imu2Tel], callback_imu_raw)
    # cb_calibrated = my_drone.telemetry.add_msg_callback()
