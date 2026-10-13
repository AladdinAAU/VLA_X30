"""PointCloud2 <-> numpy without sensor_msgs_py (fast path for 80k-point clouds at 10 Hz)."""
from __future__ import annotations

import array

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

_NP_TYPES = {
    PointField.INT8: "i1", PointField.UINT8: "u1", PointField.INT16: "i2", PointField.UINT16: "u2",
    PointField.INT32: "i4", PointField.UINT32: "u4", PointField.FLOAT32: "f4", PointField.FLOAT64: "f8",
}
_PF_TYPES = {v: k for k, v in _NP_TYPES.items()}


def cloud_dtype(msg: PointCloud2) -> np.dtype:
    """Structured dtype matching the message layout."""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in _NP_TYPES or f.count != 1:
            continue
        names.append(f.name)
        formats.append(("<" if not msg.is_bigendian else ">") + _NP_TYPES[f.datatype])
        offsets.append(f.offset)
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})


def cloud_to_struct(msg: PointCloud2) -> np.ndarray:
    """Structured view of the points (one record per point, zero-copy on the message buffer)."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros(0, dtype=cloud_dtype(msg))
    return np.frombuffer(msg.data, dtype=cloud_dtype(msg), count=n)


def xyz_of(points: np.ndarray) -> np.ndarray:
    """(N, 3) float64 array from a structured cloud."""
    return np.column_stack((points["x"], points["y"], points["z"])).astype(np.float64)


def make_cloud(header, columns: dict[str, np.ndarray]) -> PointCloud2:
    """Build a PointCloud2 from named 1-D arrays (kept in the given order, tightly packed, no padding)."""
    names = list(columns.keys())
    n = len(columns[names[0]]) if names else 0
    fields, formats, offsets, offset = [], [], [], 0
    for name in names:
        dt = np.dtype(columns[name].dtype).newbyteorder("<")
        code = dt.kind + str(dt.itemsize)
        fields.append(PointField(name=name, offset=offset, datatype=_PF_TYPES[code], count=1))
        formats.append(dt)
        offsets.append(offset)
        offset += dt.itemsize
    rec = np.zeros(n, dtype=np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": offset}))
    for name in names:
        rec[name] = columns[name]
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = offset
    msg.row_step = offset * n
    msg.data = array.array("B", rec.tobytes())  # a bytes assignment is validated per element (20 ms for 350 kB)
    msg.is_dense = True
    return msg


def voxel_downsample(xyz: np.ndarray, voxel: float, *extra: np.ndarray) -> tuple[np.ndarray, ...]:
    """Keep the first point of every voxel. Returns (xyz, *extra) subsets."""
    if len(xyz) == 0 or voxel <= 0.0:
        return (xyz, *extra)
    idx = np.floor(xyz / voxel).astype(np.int64)
    idx -= idx.min(axis=0)
    dims = idx.max(axis=0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, first = np.unique(key, return_index=True)
    return (xyz[first], *(e[first] for e in extra))
