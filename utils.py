import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp


def skew(v: np.ndarray) -> np.ndarray:
    v_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    return v_skew


def vee(Phi) -> np.ndarray:
    phi = np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])
    return phi


def wrap_plus_minis_pi(heading):
    """Wraps the angle to the range [-pi, pi]"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


def make_proto_timestamp(log_time_ns):
    seconds = log_time_ns // 1_000_000_000
    nanos = log_time_ns % 1_000_000_000
    return Timestamp(seconds=seconds, nanos=nanos)


