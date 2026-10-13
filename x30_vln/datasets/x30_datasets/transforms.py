"""Coordinate-frame helpers.

World frame (Gazebo / ROS, REP-103): x forward/east, y left, z up, yaw counter-clockwise about +z.

Habitat frame: y up, the agent looks along -z with identity rotation, quaternions are
JSON ``[x, y, z, w]``. A Habitat scene is the ROS scene rotated by -90 deg about x, hence

    ros = (x_h, -z_h, y_h),  habitat = (x_r, z_r, -y_r)

and the Habitat heading with rotation q is the direction ``R(q) @ (0, 0, -1)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3x3 rotation, R = Rz(yaw) Ry(pitch) Rx(roll) (SDF / ROS convention)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def pose_matrix(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
    T[:3, 3] = (x, y, z)
    return T


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_to_quat_xyzw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


# Rotation taking Habitat (y up) vectors to ROS (z up): ros = M @ hab.
HAB_TO_ROS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@dataclass
class SceneTransform:
    """Rigid transform from the source scene frame to the Gazebo world frame.

    ``p_world = R @ p_src + t``. Stored next to every imported world
    (``scene_transform`` in the registry) so episode importers apply the same transform.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def point(self, p) -> np.ndarray:
        return self.R @ np.asarray(p, dtype=float) + self.t

    def points(self, pts) -> np.ndarray:
        return np.asarray(pts, dtype=float) @ self.R.T + self.t

    def direction(self, v) -> np.ndarray:
        return self.R @ np.asarray(v, dtype=float)

    def to_dict(self) -> dict:
        return {"R": np.asarray(self.R).round(9).tolist(), "t": np.asarray(self.t).round(6).tolist()}

    @staticmethod
    def from_dict(d: dict) -> "SceneTransform":
        return SceneTransform(np.array(d["R"], dtype=float), np.array(d["t"], dtype=float))

    @staticmethod
    def habitat(floor_y: float = 0.0, offset_xy=(0.0, 0.0)) -> "SceneTransform":
        """Habitat scene -> world, with the floor at habitat height ``floor_y`` mapped to z=0."""
        t = -HAB_TO_ROS @ np.array([0.0, floor_y, 0.0]) + np.array([offset_xy[0], offset_xy[1], 0.0])
        return SceneTransform(HAB_TO_ROS.copy(), t)

    @staticmethod
    def from_up_axis(up: str = "z", scale: float = 1.0, floor: float = 0.0) -> "SceneTransform":
        """Generic mesh: up axis 'y' or 'z' (optionally '-y'/'-z'), unit scale, floor height (source units)."""
        mats = {
            "z": np.eye(3),
            "y": HAB_TO_ROS,
            "-y": HAB_TO_ROS.T,
            "-z": np.diag([1.0, -1.0, -1.0]),
        }
        if up not in mats:
            raise ValueError(f"up axis must be one of {sorted(mats)}")
        R = mats[up] * scale
        up_vec = np.array([0.0, 0.0, 1.0])
        # source height of the floor along the source up direction
        src_up = np.linalg.inv(mats[up]) @ up_vec
        t = -(mats[up] @ (src_up * floor)) * scale
        return SceneTransform(R, t)

    def yaw_from_forward(self, forward_src) -> float:
        f = self.direction(forward_src)
        return math.atan2(f[1], f[0])


def habitat_pose_to_world(position, rotation_xyzw, tf: SceneTransform) -> tuple[float, float, float, float]:
    """Habitat agent pose -> (x, y, z, yaw) in the world frame (agent forward = -z_h)."""
    p = tf.point(position)
    R = quat_xyzw_to_matrix(rotation_xyzw)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    yaw = tf.yaw_from_forward(fwd)
    return float(p[0]), float(p[1]), float(p[2]), float(yaw)


def world_pose_to_habitat(x: float, y: float, z: float, yaw: float, tf: SceneTransform) -> tuple[list, list]:
    """Inverse of :func:`habitat_pose_to_world` (rotation about the habitat up axis)."""
    p = np.linalg.inv(tf.R) @ (np.array([x, y, z]) - tf.t)
    fwd_w = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    fwd_h = np.linalg.inv(tf.R) @ fwd_w
    # yaw_h: rotation about +y that maps (0,0,-1) to fwd_h
    ang = math.atan2(-fwd_h[0], -fwd_h[2])
    q = [0.0, math.sin(ang / 2.0), 0.0, math.cos(ang / 2.0)]
    return p.tolist(), q
