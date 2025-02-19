from orientation import RotationQuaterion
import numpy as np
import scipy
from mcap.reader import make_reader
from state import LieState, State
from sigma_points import SigmaPoints, JulierSigmaPoints, MwereSigmaPoints, SimplexSigmaPoints
from scipy.spatial.transform import Rotation as Rot
from ukf import UKFM, QUKF
from be_telemetry import McapLogger
from models import ImuModelQuat, DvlMeasurement, DepthMeasurement
import json
import traceback
from tqdm import tqdm


def make_ukf():
    model = ImuModelQuat(
        gyro_std=0.001,
        gyro_bias_std=3.5e-4,
        gyro_bias_p=1e-16,

        acc_std=0.001,
        acc_bias_std=5e-4,
        acc_bias_p=1e-16
    )

    dim_x = 9 
    dim_q = model.Q.shape[0]
    dim_z = 3
    # sigma_points = MwereSigmaPoints(dim_x + dim_q, alpha=1.e-3, kappa=3. - (dim_x + dim_q))
    sigma_points = SimplexSigmaPoints(dim_x + dim_q)
    P0 = np.eye(dim_x)

    R = Rot.from_euler("XYZ", [0, 0, -2.14])
    q0 = Rot.from_euler("XYZ", [0, 0, -2.14]).as_quat(scalar_first=True)
    q0 = RotationQuaterion.from_vec(q0)
    v0 = q0.rotate_vec(np.array([-0.04, 0.024, -0.023]))
    p0 = np.array([1.8, -4.3, 0.22])

    # x0 = LieState(R=R, pos=p0, vel=v0)
    x0 = State(ori=q0, vel=v0, pos=p0)

    # ukf = UKFM(
    #     dim_x=dim_x,
    #     dim_q=dim_q,
    #     points=sigma_points,
    #     noise_points=noise_points,
    #     model=model,
    #     x0=x0,
    #     P0=P0,
    # )

    ukf = QUKF(model=model, dim_x=dim_x, dim_q=dim_q, x0=x0, P0=P0, sigma_points=sigma_points, Q=model.Q)

    return ukf


def main():
    np.set_printoptions(precision=2, linewidth=999)
    logger = McapLogger("ufk_test.mcap")
    logger.start()

    ukf = make_ukf()
    has_pred = False
    R_dvl = np.eye(3) * 2e-9
    # R_dvl[-1] = 4e-2
    R_depth = 1e-3
    dvl_meas = DvlMeasurement(R_dvl)
    depth_meas = DepthMeasurement(R_depth)

    with open("./log_car_left_back.mcap", "rb") as f:
        reader = make_reader(f)
        t = 0
        t_tot = 0
        t0 = 0
        for schema, channel, message in tqdm(reader.iter_messages()):
            if not has_pred and channel.topic != "imu":
                t0 = message.log_time
                t = message.log_time
                continue
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
                        u = np.array(
                            [gyro["x"], gyro["y"], gyro["z"], acc["x"], acc["y"], acc["z"]]
                        )

                        ukf.propagate(u, dt)
                        logger.log_message(
                            "state", ukf.x.to_json().encode(), timestamp=message.log_time
                        )

                    case "dvl":
                        msg = json.loads(message.data.decode())
                        vel = msg["velocity"]
                        dvl_meas.z = np.array([vel["x"], vel["y"], vel["z"]])
                        # ukf.update(dvl_meas, 0)

                        logger.log_message("dvl", message.data, message.log_time)
                        logger.log_message("state", ukf.x.to_json().encode(), message.log_time)

                    case "pos":
                        msg = log_pos(message.data)
                        logger.log_message("pos", msg, timestamp=message.log_time)

                    case "depth":
                        logger.log_message("depth", message.data, timestamp=message.log_time)
                        msg = json.loads(message.data.decode())
                        depth_meas.z = msg["value"]
                        ukf.update(depth_meas, 0)

            except Exception:
                print(f"died at: {(t - t0) * 1e-9}")
                logger.stop()
                print(traceback.format_exc())
                break

    if logger.running:
        logger.stop()


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def log_pos(msg):
    pos = json.loads(msg.decode())
    pos_estimate = {
        "northing": pos["northing"],
        "easting": pos["easting"],
        "heading": wrap_angle(pos["heading"]),
        "surge_rate": pos["surge_rate"],
        "sway_rate": pos["sway_rate"],
        "yaw_rate": pos["yaw_rate"],
        "ocean_current": pos["ocean_current"],
        "odometer": pos["odometer"],
        "is_valid": pos["is_valid"],
        "global_position": {
            "latitude": pos["global_position"]["latitude"],
            "longitude": pos["global_position"]["longitude"],
        },
        "speed_over_ground": pos["speed_over_ground"],
        "course_over_ground": pos["course_over_ground"],
    }
    return json.dumps(pos_estimate).encode()


if __name__ == "__main__":
    main()
