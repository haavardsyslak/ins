from quaternion import RotationQuaterion
import numpy as np
from mcap.reader import make_reader
from state import LieState
from sigma_points import SigmaPoints
from scipy.spatial.transform import Rotation as Rot
from ukf import UKFM
from be_telemetry import McapLogger
from models import ImuModel
import json
import traceback


def a():
    with open("./log_car_left_back.mcap", "rb") as f:
        reader = make_reader(f)
        print(reader)
        t0 = None
        for schema, channel, message in reader.iter_messages():
            print(channel.topic)
            print(schema.name)
            print(message.log_time - t0)
            input()


def make_ukf(model):
    Q = np.eye(6) * 1e-3
    R_v = np.eye(3) * 1e-3

    dim_x = 9
    dim_z = 3
    noise_points = SigmaPoints(6, alpha=1e-5, kappa=0)
    sigma_points = SigmaPoints(9, alpha=1e-5, kappa=0)
    P0 = np.eye(9) * 1e-5

    R = Rot.from_euler("XYZ", [0, 0, -2.14])
    x0 = LieState(R=R.as_matrix(), pos=np.array([1.8, -4.3, 0]), vel=np.array([0, 0, 0]))

    ukf = UKFM(
        dim_x=dim_x,
        dim_q=6,
        points=sigma_points,
        noise_points=noise_points,
        model=model,
        x0=x0,
        P0=P0,
        Q=Q,
        R=R_v,
    )

    return ukf


def main():
    np.set_printoptions(precision=2, linewidth=99)
    logger = McapLogger("ufk_test.mcap")
    logger.start()

    model = ImuModel(0, 0, 0, 0, 0, 0)

    ukf = make_ukf(model)
    has_pred = False

    with open("./log_car_left_back.mcap", "rb") as f:
        reader = make_reader(f)
        t = 0
        t_tot = 0
        # iter = reader.iter_messages()
        # schema, channel, message = next(iter)
        t0 = 0
        for schema, channel, message in reader.iter_messages():
            # TOOD: run the ukf here, and log the raw messages back to a mcap file. alternatively,
            # we can log the ukf results first then the raw messages if this takes up too much cpu
            # if not has_pred and channel.topic != "imu":
            #     t0 = message.log_time
            #     t = message.log_time
            #     continue
            if not has_pred:
                t0 = message.log_time
                t = message.log_time
            try:
                match channel.topic:
                    case "imu":
                        if not has_pred:
                            dt = 0.05
                            t = message.log_time
                            has_pred = True
                        else:
                            dt = float(message.log_time - t) * 1e-9
                            t = message.log_time
                        msg = json.loads(message.data.decode())
                        acc = msg["accelerometer"]
                        gyro = msg["gyroscope"]
                        u = np.array([acc["x"], acc["y"], acc["z"],
                                     gyro["x"], gyro["y"], gyro["z"]])

                        ukf.propagate(u, dt)
                        logger.log_message("state", ukf.x.to_json().encode(),
                                           timestamp=message.log_time)

                    case "dvl":
                        msg = json.loads(message.data.decode())
                        vel = msg["velocity"]
                        z = np.array([vel["x"], vel["y"], vel["z"]])
                        ukf.update(z, 0)

                        logger.log_message("dvl", message.data, message.log_time)
                        logger.log_message("state", ukf.x.to_json().encode(), message.log_time)

                    case "pos":
                        logger.log_message("pos", message.data, timestamp=message.log_time)

                    case "depth":
                        logger.log_message("depth", message.data, timestamp=message.log_time)

            except Exception:
                print(f"died at: {(t - t0) * 1e-9}")
                logger.stop()
                print(traceback.format_exc())
                break

    if logger.running:
        logger.stop()


if __name__ == "__main__":
    main()
