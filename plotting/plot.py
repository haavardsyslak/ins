import matplotlib.pyplot as plt
import sys
import os
from collections import defaultdict
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from foxglove_wrapper import McapProtobufReader
# Topics

# ned
# UKFState
# ufk.locatinFix
# blueye.protocol.DvlVelocityTel
# Dvl.Nis
# ukf.Pose
# blueye.protocol.Imu2Tel
# blueye.protocol.PositionEstimateTel
# gnss.Pose
# gnss.LocationFix
# NEES pos
# blueye.protocol.DepthTel

def plot_3d_position(data, topic, label=None):
    """
    Plots 3D position from the specified topic using matplotlib,
    without explicit mpl_toolkits import.

    Args:
        data (dict): Dictionary with numpy arrays per topic.
        topic (str): Topic name in the data dictionary.
        label (str): Optional label for the plotted trajectory.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # This triggers mpl_toolkits internally

    print(data[topic].keys())
    if all(k in data[topic] for k in ("position_x", "position_y", "position_z")):
        # Adjust as per your convention — velocity_x here used as a dummy axis
        x = data[topic]["position_x"]
        y = data[topic]["position_y"]
        z = data[topic]["position_z"]
    else:
        raise ValueError(f"Cannot find suitable 3D data in topic '{topic}'.")

    ax.plot(x, y, z, label=label or topic)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"3D Position Trajectory: {label or topic}")
    ax.legend()
    plt.tight_layout()


def plot_xyz_with_std_bounds(data, topic, cov_indices, label_prefix=""):
    """
    Plots position (x, y, z) with ±1σ bounds and overlays GNSS (x, y) and depth (z).

    Args:
        data (dict): Extracted data dictionary.
        topic (str): Topic name (e.g., "UKFState").
        cov_indices (tuple): Indices in flattened covariance for (x, y, z).
        label_prefix (str): Label prefix for titles/legend.
    """
    t = data[topic]["timestamp"]
    pos = np.array(data[topic]["pos"])
    cov_flat = np.array(data[topic]["covariance"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r"$x$", r"$y$", r"$z$"]

    for i, (ax, cov_idx, label) in enumerate(zip(axes, cov_indices, labels)):
        pos_i = pos[:, i]
        std = np.sqrt(cov_flat[:, cov_idx])

        # UKF with bounds
        ax.plot(t, pos_i, label=fr"{label_prefix}{label}")
        ax.fill_between(t, pos_i - std, pos_i + std, alpha=0.3, label=r"$\pm1\sigma$")

        # GNSS overlay for x and y
        if i in (0, 1) and "gnss.Pose" in data:
            gnss_t = data["gnss.Pose"]["timestamp"]
            gnss_pos = np.array(data["gnss.Pose"]["pos"])
            ax.plot(gnss_t, gnss_pos[:, i], 'k--', label="GNSS")

        # Depth overlay for z
        if i == 2 and "Depth" in data:
            depth_t = data["Depth"]["timestamp"]
            depth_z = data["Depth"]["depth"]
            ax.plot(depth_t, depth_z, 'g--', label="Depth Sensor")

        ax.set_ylabel(f"{label} [m]")
        ax.grid(True)
        ax.legend()
        ax.set_title(f"{label_prefix}{label} position with $\pm1\sigma$ bounds")

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()


def plot_position_error_with_std_bounds(data, estimate_topic, truth_topic, cov_indices, label_prefix=""):
    """
    Plots X, Y, Z position errors with ±1σ bounds using diagonal covariance.

    Args:
        data (dict): Parsed data dictionary.
        estimate_topic (str): Topic name for estimate (e.g., "UKFState").
        truth_topic (str): Topic name for truth (e.g., "gnss.Pose").
        cov_indices (tuple): Indices in flattened covariance for (x, y, z).
        label_prefix (str): Prefix for labeling the estimate.
    """
    # Interpolate GNSS truth onto estimator timestamps
    t_est = data[estimate_topic]["timestamp"]
    pos_est = np.array(data[estimate_topic]["pos"])
    cov_flat = np.array(data[estimate_topic]["covariance"])

    t_truth = data[truth_topic]["timestamp"]
    pos_truth = np.array(data[truth_topic]["pos"])

    # Interpolate ground truth position onto estimator timestamps (per axis)
    pos_truth_interp = np.stack([
        np.interp(t_est, t_truth, pos_truth[:, i]) for i in range(3)
    ], axis=1)

    error = pos_est - pos_truth_interp  # shape (N, 3)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r"$x$", r"$y$", r"$z$"]

    for i, (ax, cov_idx, label) in enumerate(zip(axes, cov_indices, labels)):
        err_i = error[:, i]
        std_i = np.sqrt(cov_flat[:, cov_idx])

        ax.plot(t_est, err_i, label=f"{label_prefix}error {label}")
        ax.fill_between(t_est, -std_i, std_i, alpha=0.3, label=r"$\pm1\sigma$")
        ax.set_ylabel(f"{label} error [m]")
        ax.set_title(f"{label} position error with $\pm1\sigma$ bounds")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

msgs = McapProtobufReader("mcap_plotting/square_dead_rekkoning.mcap")
data = defaultdict(lambda: defaultdict(list))

for message in msgs:
    timestamp = message.log_time_ns * 1e-9  # convert ns to seconds

    match message.topic:
        case "UKFState":
            msg = message.proto_msg
            data["UKFState"]["timestamp"].append(timestamp)
            data["UKFState"]["pos"].append(np.array([msg.position_x, msg.position_y, msg.position_z]))
            data["UKFState"]["vel"].append(np.array([msg.velocity_x, msg.velocity_y, msg.velocity_z]))
            data["UKFState"]["heading"].append(msg.heading)
            data["UKFState"]["roll"].append(msg.roll)
            data["UKFState"]["pitch"].append(msg.pitch)
            data["UKFState"]["gyro_bias"].append(np.array([msg.gyro_bias_x, msg.gyro_bias_y, msg.gyro_bias_z]))
            data["UKFState"]["accel_bias"].append(np.array([msg.accel_bias_x, msg.accel_bias_y, msg.accel_bias_z]))
            data["UKFState"]["covariance"].append(msg.covariance)

        case "gnss.Pose":
            msg = message.proto_msg
            data["gnss.Pose"]["timestamp"].append(timestamp)
            data["gnss.Pose"]["pos"].append([msg.translation.x, msg.translation.y, msg.translation.z])

        case "blueye.protocol.DvlVelocityTel":
            vel = message.proto_msg.dvl_velocity.velocity
            data["DVL"]["timestamp"].append(timestamp)
            data["DVL"]["vel"].append([vel.x, vel.y, vel.z])

        case "blueye.protocol.DepthTel":
            depth = message.proto_msg.depth.value
            data["Depth"]["timestamp"].append(timestamp)
            data["Depth"]["depth"].append(depth)

        case "Dvl.Nis":
            data["Dvl.NIS"]["timestamp"].append(timestamp)
            data["Dvl.NIS"]["nis"].append(message.proto_msg.value)

        case "Nees pos":
            data["NEES"]["timestamp"].append(timestamp)
            data["NEES"]["pos"].append(message.proto_msg.value)

# Convert all lists to numpy arrays
for topic in data:
    for key in data[topic]:
        data[topic][key] = np.array(data[topic][key])

# plot_3d_position(data, "UKFState")
plot_xyz_with_std_bounds(
    data,
    topic="UKFState",
    cov_indices=(0, 1, 2),
    label_prefix="UKFM "
)

plot_position_error_with_std_bounds(
    data,
    estimate_topic="UKFState",
    truth_topic="gnss.Pose",
    cov_indices=(0, 1, 2),  # if state starts with position
    label_prefix="UKF "
)
plt.show()

