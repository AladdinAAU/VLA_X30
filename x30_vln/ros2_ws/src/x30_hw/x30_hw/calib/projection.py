"""Lidar -> camera projection math and extrinsic adjustment (numpy/OpenCV, no ROS)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def rpy_deg_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS convention: R = Rz(yaw) Ry(pitch) Rx(roll) (extrinsic x-y-z rotations), degrees."""
    r, p, y = (math.radians(a) for a in (roll, pitch, yaw))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def pose_matrix(xyz: Sequence[float], rpy_deg: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_deg_to_matrix(*rpy_deg)
    t[:3, 3] = xyz
    return t


def invert(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


class ExtrinsicModel:
    """camera_link pose in base_link (xyz, rpy_deg) + fixed optical rotation + lidar_link pose in base_link.

    `cam_from_lidar()` maps lidar_link points into the camera OPTICAL frame (z forward, x right, y down)."""

    def __init__(self, camera_xyz, camera_rpy_deg, optical_rpy_deg=(-90.0, 0.0, -90.0), lidar_xyz=(0, 0, 0),
                 lidar_rpy_deg=(0, 0, 0)):
        self.xyz = list(map(float, camera_xyz))
        self.rpy = list(map(float, camera_rpy_deg))
        self.optical = list(map(float, optical_rpy_deg))
        self.t_base_lidar = pose_matrix(lidar_xyz, lidar_rpy_deg)
        self.initial = (list(self.xyz), list(self.rpy))

    def cam_from_lidar(self) -> np.ndarray:
        t_base_opt = pose_matrix(self.xyz, self.rpy) @ pose_matrix((0, 0, 0), self.optical)
        return invert(t_base_opt) @ self.t_base_lidar

    def adjust(self, axis: str, delta: float) -> None:
        """axis in x y z (metres, camera_link axes) or roll pitch yaw (degrees)."""
        if axis in ("x", "y", "z"):
            self.xyz["xyz".index(axis)] += delta
        elif axis in ("roll", "pitch", "yaw"):
            self.rpy[("roll", "pitch", "yaw").index(axis)] += delta
        else:
            raise ValueError(axis)

    def reset(self) -> None:
        self.xyz, self.rpy = list(self.initial[0]), list(self.initial[1])

    def overlay(self) -> Dict[str, Dict[str, List[float]]]:
        return {"camera_link": {"xyz": [round(v, 5) for v in self.xyz], "rpy_deg": [round(v, 4) for v in self.rpy]}}


def project_points(points_lidar: np.ndarray, t_cam_lidar: np.ndarray, k: np.ndarray,
                   dist: Optional[np.ndarray] = None, size: Optional[Tuple[int, int]] = None,
                   min_depth: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (uv Nx2 float, depth N, valid mask N). Points behind the camera or outside `size`=(w,h) are invalid."""
    p = points_lidar[:, :3].astype(np.float64)
    pc = p @ t_cam_lidar[:3, :3].T + t_cam_lidar[:3, 3]
    z = pc[:, 2]
    valid = z > min_depth
    zs = np.where(valid, z, 1.0)
    if dist is not None and np.any(np.asarray(dist) != 0):
        import cv2
        uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, np.asarray(dist, np.float64))
        uv = uv.reshape(-1, 2)
    else:
        uv = np.stack([k[0, 0] * pc[:, 0] / zs + k[0, 2], k[1, 1] * pc[:, 1] / zs + k[1, 2]], axis=1)
    if size is not None:
        w, h = size
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, z, valid


def colorize_depth(depth: np.ndarray, max_depth: float = 12.0) -> np.ndarray:
    """BGR uint8 colours from near (red) to far (blue)."""
    t = np.clip(depth / max_depth, 0.0, 1.0)
    b = (255 * t).astype(np.uint8)
    r = (255 * (1 - t)).astype(np.uint8)
    g = (255 * (1 - np.abs(2 * t - 1))).astype(np.uint8)
    return np.stack([b, g, r], axis=1)


def overlay_points(image_bgr: np.ndarray, uv: np.ndarray, depth: np.ndarray, valid: np.ndarray, radius: int = 2,
                   max_depth: float = 12.0) -> np.ndarray:
    import cv2
    out = image_bgr.copy()
    cols = colorize_depth(depth[valid], max_depth)
    for (u, v), c in zip(uv[valid], cols):
        cv2.circle(out, (int(round(u)), int(round(v))), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def load_cloud(path: str) -> np.ndarray:
    """.npy (Nx3+) or whitespace/ASCII PCD text (x y z [..])."""
    if path.endswith(".npy"):
        return np.load(path)[:, :3]
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        in_data = not path.endswith(".pcd")
        for line in f:
            if not in_data:
                in_data = line.startswith("DATA")
                if in_data and "ascii" not in line:
                    raise ValueError("only ASCII PCD supported; convert with pcl_convert_pcd_ascii_binary or save .npy")
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=np.float64)


def parse_adjust_script(script: str) -> List[Tuple[str, float]]:
    """'x+0.05,yaw-2,pitch+1.5' -> [('x', 0.05), ('yaw', -2.0), ('pitch', 1.5)]."""
    out = []
    for item in filter(None, (s.strip() for s in script.split(","))):
        for sign in ("+", "-"):
            if sign in item[1:]:
                i = item.index(sign, 1)
                out.append((item[:i], float(item[i:])))
                break
        else:
            raise ValueError(f"bad adjustment '{item}' (expected e.g. x+0.05 or yaw-2)")
    return out
