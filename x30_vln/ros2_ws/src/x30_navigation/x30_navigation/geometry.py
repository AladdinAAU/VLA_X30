"""Small SE(3) / quaternion helpers (numpy only, no tf_transformations dependency)."""
from __future__ import annotations

import math

import numpy as np


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix from a quaternion (x, y, z, w); a zero quaternion yields identity."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) from a rotation matrix."""
    t = np.trace(r)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(r: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 0.999999:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return roll, pitch, yaw


def yaw_of(r: np.ndarray) -> float:
    return math.atan2(r[1, 0], r[0, 0])


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def make_transform(r: np.ndarray, t) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def invert(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def pose_msg_to_matrix(pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4."""
    q = pose.orientation
    return make_transform(quat_to_matrix(q.x, q.y, q.z, q.w), [pose.position.x, pose.position.y, pose.position.z])


def transform_msg_to_matrix(tf) -> np.ndarray:
    """geometry_msgs/Transform -> 4x4."""
    q = tf.rotation
    return make_transform(quat_to_matrix(q.x, q.y, q.z, q.w), [tf.translation.x, tf.translation.y, tf.translation.z])


def fill_pose(msg, m: np.ndarray) -> None:
    """Write a 4x4 into a geometry_msgs/Pose."""
    msg.position.x, msg.position.y, msg.position.z = (float(v) for v in m[:3, 3])
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = matrix_to_quat(m[:3, :3])


def fill_transform(msg, m: np.ndarray) -> None:
    """Write a 4x4 into a geometry_msgs/Transform."""
    msg.translation.x, msg.translation.y, msg.translation.z = (float(v) for v in m[:3, 3])
    msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w = matrix_to_quat(m[:3, :3])
