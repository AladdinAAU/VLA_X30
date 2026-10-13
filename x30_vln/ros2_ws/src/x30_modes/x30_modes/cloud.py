"""Point cloud / laser scan helpers on plain numpy (messages are read duck-typed, no sensor_msgs_py needed)."""
from __future__ import annotations

import math

import numpy as np

_NP = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_xyz(msg, drop_zero_intensity: bool = False) -> np.ndarray:
    """sensor_msgs/PointCloud2 -> (N, 3) float32 of finite points. Requires float32/float64 x, y, z fields.

    With `drop_zero_intensity` points whose `intensity` field is 0 are removed: the Livox cloud (real robot, bag and simulator) reports
    invalid returns as points AT THE SENSOR ORIGIN with intensity 0, which would otherwise look like obstacles on the robot body.
    """
    offs = {}
    inten = None
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            offs[f.name] = (int(f.offset), _NP[int(f.datatype)])
        elif f.name == "intensity" and drop_zero_intensity:
            inten = (int(f.offset), _NP[int(f.datatype)])
    if len(offs) < 3:
        raise ValueError("PointCloud2 has no x, y, z fields")
    n = int(msg.width) * int(msg.height)
    if n == 0:
        return np.zeros((0, 3), np.float32)
    step = int(msg.point_step)
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    cols = []
    for k in ("x", "y", "z"):
        off, dt = offs[k]
        size = np.dtype(dt).itemsize
        cols.append(np.ascontiguousarray(raw[:, off:off + size]).view(dt).reshape(n).astype(np.float32))
    xyz = np.stack(cols, axis=1)
    keep = np.isfinite(xyz).all(axis=1)
    if inten is not None:
        off, dt = inten
        size = np.dtype(dt).itemsize
        keep &= np.ascontiguousarray(raw[:, off:off + size]).view(dt).reshape(n) != 0
    return xyz[keep]


def scan_xy(msg) -> np.ndarray:
    """sensor_msgs/LaserScan -> (N, 2) points in the scan frame (valid ranges only)."""
    r = np.asarray(msg.ranges, dtype=np.float32)
    ang = msg.angle_min + np.arange(r.size) * msg.angle_increment
    ok = np.isfinite(r) & (r >= max(msg.range_min, 0.0)) & (r <= msg.range_max)
    return np.stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])], axis=1)


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform_points(xyz: np.ndarray, rot: np.ndarray, trans) -> np.ndarray:
    return xyz @ rot.T + np.asarray(trans, float)


def band_filter_xy(xyz: np.ndarray, z_min: float, z_max: float, max_points: int = 6000) -> np.ndarray:
    """Keep points with z in [z_min, z_max] and return their (x, y); thin out to at most max_points."""
    if xyz.size == 0:
        return np.zeros((0, 2), np.float32)
    keep = (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    xy = xyz[keep, :2]
    if xy.shape[0] > max_points:
        xy = xy[:: int(math.ceil(xy.shape[0] / max_points))]
    return xy
