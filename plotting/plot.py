import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from collections import defaultdict
import numpy as np
import scipy.stats as stats
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
    t = get_relative_time(data[topic]["timestamp"])
    pos = np.array(data[topic]["pos"])
    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    padding = 0.5 * (pos_max - pos_min)
    y_lims = [(pos_min[i] - padding[i], pos_max[i] + padding[i]) for i in range(3)]
    cov_flat = np.array(data[topic]["covariance"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r"$x$", r"$y$", r"$z$"]

    for i, (ax, cov_idx, label) in enumerate(zip(axes, cov_indices, labels)):
        pos_i = pos[:, i]
        std = np.sqrt(cov_flat[:, cov_idx])

        # UKF with bounds
        ax.plot(t, pos_i, label=fr"{label_prefix}{label}")
        ax.fill_between(t, pos_i - std, pos_i + std, alpha=0.3, label=r"$\pm3\sigma$")

        # GNSS overlay for x and y
        if i in (0, 1) and "gnss.Pose" in data:
            gnss_t = get_relative_time(data["gnss.Pose"]["timestamp"])
            gnss_pos = np.array(data["gnss.Pose"]["pos"])
            ax.plot(gnss_t, gnss_pos[:, i], 'k--', label="GNSS")

        # Depth overlay for z
        if i == 2 and "Depth" in data:
            depth_t = get_relative_time(data["Depth"]["timestamp"])
            depth_z = data["Depth"]["depth"]
            ax.plot(depth_t, depth_z, 'g--', label="Depth Sensor")

        ax.set_ylim(*y_lims[i])
        ax.set_ylabel(f"{label} [m]")
        ax.grid(True)
        ax.legend()
        ax.set_title(f"{label_prefix}{label} position with $\pm3\sigma$ bounds")

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
    t_est = get_relative_time(data[estimate_topic]["timestamp"])
    pos_est = np.array(data[estimate_topic]["pos"])[:, :2]
    cov_flat = np.array(data[estimate_topic]["covariance"])

    t_truth = get_relative_time(data[truth_topic]["timestamp"])
    pos_truth = np.array(data[truth_topic]["pos"])

    # Interpolate ground truth position onto estimator timestamps (per axis)
    pos_truth_interp = np.stack([
        np.interp(t_est, t_truth, pos_truth[:, i]) for i in range(2)
    ], axis=1)

    error = pos_est - pos_truth_interp  # shape (N, 3)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    labels = [r"$x$", r"$y$", r"$z$"]

    for i, (ax, cov_idx, label) in enumerate(zip(axes, cov_indices, labels)):
        err_i = error[:, i]
        std_i = np.sqrt(cov_flat[:, cov_idx])

        ax.plot(t_est, err_i, label=f"{label_prefix}error {label}")
        ax.fill_between(t_est, -std_i, std_i, alpha=0.3, label=r"$\pm 3\sigma$")
        ax.set_ylabel(f"{label} error [m]")
        ax.set_title(f"{label} position error with $\pm3\sigma$ bounds")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()


def plot_velocity_with_std_bounds(data, estimate_topic, dvl_topic, cov_indices, label_prefix=""):
    """
    Plots estimated velocity (x, y, z) with ±1σ bounds and overlays DVL velocities.

    Args:
        data (dict): Extracted data dictionary.
        estimate_topic (str): Topic with UKF velocity estimates.
        dvl_topic (str): Topic with DVL velocity measurements.
        cov_indices (tuple): Indices in flattened covariance for velocity (vx, vy, vz).
        label_prefix (str): Optional label prefix for plot titles/legends.
    """
    t_est = get_relative_time(data[estimate_topic]["timestamp"])
    vel_est = np.array(data[estimate_topic]["vel"])
    cov_flat = np.array(data[estimate_topic]["covariance"])
    vel_min = vel_est.min(axis=0)
    vel_max = vel_est.max(axis=0)
    padding = 0.2 * (vel_max - vel_min)
    y_lims = [(vel_min[i] - padding[i], vel_max[i] + padding[i]) for i in range(3)]

    t_dvl = get_relative_time(data[dvl_topic]["timestamp"])
    vel_dvl = np.array(data[dvl_topic]["vel"])

    # Interpolate DVL velocities to estimator timestamps for alignment
    vel_dvl_interp = np.stack([
        np.interp(t_est, t_dvl, vel_dvl[:, i]) for i in range(3)
    ], axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r"$v_x$", r"$v_y$", r"$v_z$"]

    for i, (ax, cov_idx, label) in enumerate(zip(axes, cov_indices, labels)):
        vel_i = vel_est[:, i]
        std_i = np.sqrt(cov_flat[:, cov_idx]) * 3
        dvl_i = vel_dvl_interp[:, i]

        ax.plot(t_est, vel_i, label=f"{label_prefix}{label} (UKF)")
        ax.fill_between(t_est, vel_i - std_i, vel_i + std_i, alpha=0.3, label=r"$\pm3\sigma$")
        ax.plot(t_est, dvl_i, 'k--', label="DVL")

        ax.set_ylabel(f"{label} [m/s]")
        ax.set_title(f"{label} velocity with $\pm3\sigma$ bounds")
        ax.grid(True)
        ax.legend()
        ax.set_ylim(*y_lims[i])

    axes[-1].set_xlabel("Time [s]")
    # plt.tight_layout()


def plot_nis_nees_with_chi2_bounds(data, nis_topic, nees_topic, dof_nis, dof_nees, alpha=0.05):
    """
    Plots NIS and NEES with Chi-squared confidence bounds and shows
    percentage of samples inside the bounds in the legend.

    Args:
        data (dict): Parsed data dictionary.
        nis_topic (str): Topic name for NIS data.
        nees_topic (str): Topic name for NEES data.
        dof_nis (int): Degrees of freedom for NIS (typically measurement dimension).
        dof_nees (int): Degrees of freedom for NEES (typically state dimension).
        alpha (float): Significance level for confidence bounds (default is 0.05 for 95%).
    """
    def chi2_bounds(dof):
        lower = stats.chi2.ppf(alpha / 2, dof)
        upper = stats.chi2.ppf(1 - alpha / 2, dof)
        return lower, upper

    nis_t = get_relative_time(data[nis_topic]["timestamp"])
    nis_vals = data[nis_topic]["nis"]
    nis_lower, nis_upper = chi2_bounds(dof_nis)
    nis_inside = np.sum((nis_vals >= nis_lower) & (nis_vals <= nis_upper)) / len(nis_vals) * 100

    nees_t = get_relative_time(data[nees_topic]["timestamp"])
    nees_vals = data[nees_topic]["pos"]
    nees_lower, nees_upper = chi2_bounds(dof_nees)
    nees_inside = np.sum((nees_vals >= nees_lower) & (nees_vals <= nees_upper)) / len(nees_vals) * 100

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # NIS Plot
    ax = axes[0]
    ax.plot(nis_t, nis_vals, label="NIS")
    ax.axhline(nis_lower, color="r", linestyle="--", label=f"Chi² {1-alpha:.0%} bounds")
    ax.axhline(nis_upper, color="r", linestyle="--")
    ax.set_title(f"NIS (inside bounds: {nis_inside:.1f}%)")
    ax.set_ylabel("NIS")
    ax.grid(True)
    ax.legend()

    # NEES Plot
    ax = axes[1]
    ax.plot(nees_t, nees_vals, label="NEES")
    ax.axhline(nees_lower, color="r", linestyle="--", label=f"Chi² {1-alpha:.0%} bounds")
    ax.axhline(nees_upper, color="r", linestyle="--")
    ax.set_title(f"NEES (inside bounds: {nees_inside:.1f}%)")
    ax.set_ylabel("NEES")
    ax.set_xlabel("Time [s]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

def plot_orientation(data, topic, label_prefix=""):
    """
    Plots roll, pitch, and heading over time.

    Args:
        data (dict): Data dictionary.
        topic (str): Topic name (e.g., "UKFState").
        label_prefix (str): Optional label prefix for legends/titles.
    """
    t = get_relative_time(data[topic]["timestamp"])
    roll = np.array(data[topic]["roll"])
    pitch = np.array(data[topic]["pitch"])
    heading = np.array(data[topic]["heading"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    angles = [roll, pitch, heading]
    labels = [r"Roll [$^\circ$]", r"Pitch [$^\circ$]", r"Heading [$^\circ$]"]

    for ax, angle, label in zip(axes, angles, labels):
        ax.plot(t, angle, label=f"{label_prefix}{label}")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()
        ax.set_title(f"{label_prefix}{label}")

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()

def plot_gyro_bias_with_std_bounds(data, topic, cov_indices, label_prefix=""):
    """
    Plots gyroscope bias estimates (x, y, z) with ±1σ bounds.

    Args:
        data (dict): Data dictionary.
        topic (str): Topic containing UKF state (e.g., "UKFState").
        cov_indices (tuple): Indices in the flattened covariance for (bias_x, bias_y, bias_z).
        label_prefix (str): Optional label prefix for titles and legends.
    """
    t = data[topic]["timestamp"]
    t = t - t[0]  # Relative time starting at t = 0

    gyro_bias = np.array(data[topic]["gyro_bias"])
    cov_flat = np.array(data[topic]["covariance"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r"$b_{\omega, x}$", r"$b_{\omega, y}$", r"$b_{omega, z}$"]

    for i, (ax, cov_idx, label) in enumerate(zip(axes, cov_indices, labels)):
        bias_i = gyro_bias[:, i]
        std_i = np.sqrt(cov_flat[:, cov_idx])

        ax.plot(t, bias_i, label=f"{label_prefix}{label} bias")
        ax.fill_between(t, bias_i - std_i, bias_i + std_i, alpha=0.3, label=r"$\pm3\sigma$")
        ax.set_ylabel(f"{label} [rad/s]")
        ax.set_title(f"{label_prefix}{label} gyroscope bias with $\pm3\sigma$ bounds")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()


def plot_accel_bias_with_std_bounds(data, topic, cov_indices, label_prefix=""):
    """
    Plots accelerometer bias estimates (x, y, z) with ±1σ bounds.

    Args:
        data (dict): Data dictionary.
        topic (str): Topic containing UKF state (e.g., "UKFState").
        cov_indices (tuple): Indices in the flattened covariance for (bias_x, bias_y, bias_z).
        label_prefix (str): Optional label prefix for titles and legends.
    """
    t = data[topic]["timestamp"]
    t = t - t[0]  # Relative time

    accel_bias = np.array(data[topic]["accel_bias"])
    cov_flat = np.array(data[topic]["covariance"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r"$b_{a, x}$", r"$b_{a, y}$", r"$b_{a, z}$"]

    for i, (ax, cov_idx, label) in enumerate(zip(axes, cov_indices, labels)):
        bias_i = accel_bias[:, i]
        std_i = np.sqrt(cov_flat[:, cov_idx])

        ax.plot(t, bias_i, label=f"{label_prefix}{label} bias")
        ax.fill_between(t, bias_i - std_i, bias_i + std_i, alpha=0.3, label=r"$\pm3\sigma$")
        ax.set_ylabel(f"{label} [m/s²]")
        ax.set_title(f"{label_prefix}{label} accelerometer bias with $\pm3\sigma$ bounds")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def get_relative_time(timestamps):
    return timestamps - timestamps[0]

# msgs = McapProtobufReader("mcap_plotting/square_dead_rekkoning_final.mcap")
msgs = McapProtobufReader("mcap_plotting/square_dead_rekkoning_final_ferdiferdi.mcap")
# msgs = McapProtobufReader("01testing.mcap")
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

        case "NEES pos":
            data["NEES"]["timestamp"].append(timestamp)
            data["NEES"]["pos"].append(message.proto_msg.value)

# Convert all lists to numpy arrays
for topic in data:
    for key in data[topic]:
        data[topic][key] = np.array(data[topic][key])

mpl.rcParams.update({
    "font.size": 12,              # Base font size
    "axes.titlesize": 14,         # Title size
    "axes.labelsize": 12,         # Axis label size
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


def ukfm_square_plots():
# plot_3d_position(data, "UKFState")
    plot_xyz_with_std_bounds(
        data,
        topic="UKFState",
        cov_indices=(0, 1, 2),
        label_prefix="UKFM "
    )
    plt.savefig("plotting/ukfm_position.pdf", bbox_inches='tight')

    plot_position_error_with_std_bounds(
        data,
        estimate_topic="UKFState",
        truth_topic="gnss.Pose",
        cov_indices=(0, 1, 2),  # if state starts with position
        label_prefix="UKF "
    )
    plt.savefig("plotting/ukfm_position_error.pdf", bbox_inches='tight')

    plot_velocity_with_std_bounds(
        data,
        estimate_topic="UKFState",
        dvl_topic="DVL",
        cov_indices=(6,7,8),
        label_prefix="UKFM"
    )
    plt.savefig("plotting/ukfm_velocity.pdf", bbox_inches='tight')

    plot_nis_nees_with_chi2_bounds(
        data,
        nis_topic="Dvl.NIS",
        nees_topic="NEES",
        dof_nis=3,
        dof_nees=2,
        alpha=0.05
    )
    plt.savefig("plotting/ukfm_NIS_NEES.pdf", bbox_inches='tight')


    plot_orientation(data, topic="UKFState", label_prefix="UKF ")

    plot_gyro_bias_with_std_bounds(
        data,
        topic="UKFState",
        cov_indices=(9, 10, 11),
        label_prefix="UKFM "
    )
    plt.savefig("plotting/ukf_gyro_bias.pdf", bbox_inches='tight')

    plot_accel_bias_with_std_bounds(
        data,
        topic="UKFState",
        cov_indices=(12, 13, 14),
        label_prefix="UKFM"
    )
    plt.savefig("plotting/ukf_accel_bias.pdf", bbox_inches='tight')

    plt.show()


ukfm_square_plots()
