"""SE(2) helpers and trajectory alignment (numpy only)."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def wrap(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    w = (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi
    return float(w) if w.ndim == 0 else w


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class SE2:
    """Rigid 2D transform p' = R(theta) p + t."""

    __slots__ = ("x", "y", "yaw")

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)

    def apply(self, pts):
        """Apply to a point (2,) or an array (N, 2)."""
        p = np.asarray(pts, float)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        r = np.array([[c, -s], [s, c]])
        return p @ r.T + np.array([self.x, self.y])

    def compose(self, other: "SE2") -> "SE2":
        """self * other (apply other first)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return SE2(self.x + c * other.x - s * other.y, self.y + s * other.x + c * other.y, wrap(self.yaw + other.yaw))

    def inverse(self) -> "SE2":
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return SE2(-(c * self.x + s * self.y), -(-s * self.x + c * self.y), wrap(-self.yaw))

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}

    def __repr__(self) -> str:
        return f"SE2(x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f})"


def path_length(xy) -> float:
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy[:, :2], axis=0), axis=1)))


def circular_mean(angles: Iterable[float]) -> float:
    a = np.asarray(list(angles), float)
    return float(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def align_yaw(map_xy, map_yaw, gt_xy, gt_yaw) -> SE2:
    """Transform taking map-frame poses to the world frame, from yaw differences plus translation.

    The rotation is the circular mean of (gt_yaw - map_yaw), so it stays valid for short or straight
    trajectories where a position-only fit is ill-conditioned. The translation is the mean residual."""
    map_xy, gt_xy = np.asarray(map_xy, float), np.asarray(gt_xy, float)
    dyaw = circular_mean(wrap(np.asarray(gt_yaw, float) - np.asarray(map_yaw, float)))
    rot = SE2(0.0, 0.0, dyaw)
    t = np.mean(gt_xy - rot.apply(map_xy), axis=0)
    return SE2(float(t[0]), float(t[1]), dyaw)


def align_umeyama_2d(map_xy, gt_xy) -> SE2:
    """Least-squares rigid (no scale) 2D fit of map points onto world points."""
    p, q = np.asarray(map_xy, float), np.asarray(gt_xy, float)
    mp, mq = p.mean(0), q.mean(0)
    h = (p - mp).T @ (q - mq) / len(p)
    u, _, vt = np.linalg.svd(h)
    s = np.eye(2)
    if np.linalg.det(vt.T @ u.T) < 0:
        s[1, 1] = -1
    r = vt.T @ s @ u.T
    t = mq - r @ mp
    return SE2(float(t[0]), float(t[1]), math.atan2(r[1, 0], r[0, 0]))


def resample_path(xy, spacing: float) -> np.ndarray:
    """Linear resampling of a polyline at (about) constant arc-length spacing; keeps first and last point."""
    xy = np.asarray(xy, float)[:, :2]
    if len(xy) < 2:
        return xy.copy()
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return xy[:1].copy()
    n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, xy[:, 0]), np.interp(u, s, xy[:, 1])], axis=1)


def point_to_box_distance(p: Sequence[float], center: Sequence[float], size: Sequence[float]) -> float:
    """Euclidean distance in the xy plane from point p to an axis-aligned box (0 inside)."""
    dx = max(abs(p[0] - center[0]) - size[0] / 2.0, 0.0)
    dy = max(abs(p[1] - center[1]) - size[1] / 2.0, 0.0)
    return math.hypot(dx, dy)


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix of a unit quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
