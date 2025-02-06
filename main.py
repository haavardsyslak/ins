from quaternion import RotationQuaterion
import numpy as np
import scipy
from mcap.reader import make_reader
from state import LieState
from sigma_points import SigmaPoints, JulierSigmaPoints
from scipy.spatial.transform import Rotation as Rot
from ukf import UKFM
from be_telemetry import McapLogger
from models import ImuModel
import json
import traceback
from tqdm import tqdm



def make_ukf(model):
    imu_std = np.array(
        [
            0.01,  # gyro
            0.01,  # accel
            0.0000001,  # gyro bias
            # 0.000000000001,  # acc bias
        ]
    )
    I3 = np.eye(3)
    Q = scipy.linalg.block_diag(imu_std[0]**2 * I3, imu_std[1]**2 * I3)
    # Q = imu_noise_to_Q(imu_std)
    R_v = np.eye(3) * 1e-9

    dim_x = 9
    dim_q = Q.shape[0]
    dim_z = 3
    noise_points = SigmaPoints(dim_q, alpha=1.5e-1, kappa=0)
    sigma_points = SigmaPoints(dim_x, alpha=1.5e-1, kappa=0)
    # noise_points = JulierSigmaPoints(Q.shape[0], alpha=1e-1)
    # sigma_points = JulierSigmaPoints(dim_x, alpha=1e-1)
    P0 = np.eye(dim_x) * 1e-6

    R = Rot.from_euler("XYZ", [0, 0, -2.14]).as_matrix()
    v0 = R @ np.array([-0.04, 0.024, -0.023])
    p0 = np.array([1.8, -4.3, 0])
    bg0 = np.array([0, 0, 0])
    ba0 = np.array([0, 0, 0])

    x0 = LieState(R=R, pos=np.array([1.8, -4.3, 0]), vel=v0)# , gyro_bias=bg0)# , acc_bias=bg0)

    ukf = UKFM(
        dim_x=dim_x,
        dim_q=dim_q,
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
        for schema, channel, message in tqdm(reader.iter_messages()):
            # TOOD: run the ukf here, and log the raw messages back to a mcap file. alternatively,
            # we can log the ukf results first then the raw messages if this takes up too much cpu
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
                        z = np.array([vel["x"], vel["y"], vel["z"]])
                        ukf.update(z, 0)

                        logger.log_message("dvl", message.data, message.log_time)
                        logger.log_message("state", ukf.x.to_json().encode(), message.log_time)

                    case "pos":
                        msg = log_pos(message.data)
                        logger.log_message("pos", msg, timestamp=message.log_time)

                    case "depth":
                        logger.log_message("depth", message.data, timestamp=message.log_time)

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

def imu_noise_to_Q(arr):
    n = len(arr)
    # Initialize an empty block diagonal matrix of size (3n, 3n)
    block_diag = np.zeros((3 * n, 3 * n))

    for i, element in enumerate(arr):
        # Compute the block: element^2 * I_3
        block = (element**2) * np.eye(3)
        # Place the block in the appropriate position in the block diagonal matrix
        block_diag[3 * i: 3 * (i + 1), 3 * i: 3 * (i + 1)] = block

    return block_diag


if __name__ == "__main__":
    main()
