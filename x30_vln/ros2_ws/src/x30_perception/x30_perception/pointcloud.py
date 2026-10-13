"""Decode sensor_msgs/PointCloud2 buffers into numpy arrays (no ROS imports; works on a message or a bag record)."""

from __future__ import annotations

import numpy as np

_DTYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def decode_xyz(
    data, fields, point_step: int, is_bigendian: bool = False, time_field: str = 'timestamp', time_unit_s: float = 1e-3
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (xyz float32 (N,3), per-point time in seconds (N,) float64 or None).

    `fields` is a sequence of objects with .name .offset .datatype (PointField). Non-finite points are dropped.
    The Livox `timestamp` field (float64) is in milliseconds on the X30 (checked on the recorded bag), hence the default unit."""
    buf = np.frombuffer(bytes(data) if not isinstance(data, (bytes, bytearray, memoryview, np.ndarray)) else data, dtype=np.uint8)
    n = buf.size // point_step
    buf = buf[: n * point_step].reshape(n, point_step)
    by_name = {f.name: f for f in fields}
    end = '>' if is_bigendian else '<'

    def col(name):
        f = by_name[name]
        dt = np.dtype(end + _DTYPES[f.datatype])
        return np.ascontiguousarray(buf[:, f.offset : f.offset + dt.itemsize]).view(dt).reshape(n)

    xyz = np.stack([col('x'), col('y'), col('z')], axis=1).astype(np.float32, copy=False)
    t = col(time_field).astype(np.float64) * time_unit_s if time_field in by_name else None
    ok = np.isfinite(xyz).all(axis=1)
    if not ok.all():
        xyz = xyz[ok]
        t = t[ok] if t is not None else None
    return xyz, t
