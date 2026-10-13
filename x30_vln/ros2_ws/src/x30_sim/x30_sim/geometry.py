"""Small rigid-transform helpers shared by the robot description and the scene parser (numpy only)."""
from __future__ import annotations

import math

import numpy as np


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    sp = -R[2, 0]
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    if abs(sp) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def axis_angle_matrix(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def pose_matrix(x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(roll, pitch, yaw)
    T[:3, 3] = (x, y, z)
    return T


def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])
