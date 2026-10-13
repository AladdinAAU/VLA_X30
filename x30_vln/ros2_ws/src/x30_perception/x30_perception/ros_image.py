"""sensor_msgs/Image <-> numpy without cv_bridge (the encodings used here are simple). No ROS imports."""

from __future__ import annotations

import numpy as np

_CH = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1, '8UC1': 1, '8UC3': 3, '16UC1': 1, 'mono16': 1, '32FC1': 1}
_DT = {
    'rgb8': 'u1',
    'bgr8': 'u1',
    'rgba8': 'u1',
    'bgra8': 'u1',
    'mono8': 'u1',
    '8UC1': 'u1',
    '8UC3': 'u1',
    '16UC1': '<u2',
    'mono16': '<u2',
    '32FC1': '<f4',
}


def to_array(msg) -> np.ndarray:
    """Decode an Image message to an (h, w) or (h, w, c) array in the message's own channel order."""
    enc = msg.encoding
    if enc not in _DT:
        raise ValueError(f'unsupported image encoding {enc!r}')
    dt, ch = np.dtype(_DT[enc]), _CH[enc]
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = row[:, : msg.width * ch * dt.itemsize].copy().view(dt).reshape(msg.height, msg.width, ch)
    return arr[..., 0] if ch == 1 else arr


def to_bgr(msg) -> np.ndarray:
    """Colour or grey Image message -> uint8 BGR (h, w, 3)."""
    a = to_array(msg)
    enc = msg.encoding
    if enc in ('rgb8',):
        return np.ascontiguousarray(a[..., ::-1])
    if enc in ('rgba8',):
        return np.ascontiguousarray(a[..., 2::-1])
    if enc in ('bgra8',):
        return np.ascontiguousarray(a[..., :3])
    if enc in ('bgr8', '8UC3'):
        return a
    if a.ndim == 2:
        g = a if a.dtype == np.uint8 else (a >> 8).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)
    raise ValueError(f'cannot convert {enc!r} to BGR')


def depth_meters(msg) -> np.ndarray:
    """Depth image -> float32 metres (32FC1 as is; 16UC1 is millimetres)."""
    a = to_array(msg)
    if msg.encoding == '32FC1':
        return a.astype(np.float32, copy=False)
    if msg.encoding in ('16UC1', 'mono16'):
        d = a.astype(np.float32) * 1e-3
        d[a == 0] = np.nan
        return d
    raise ValueError(f'unsupported depth encoding {msg.encoding!r}')


def from_bgr(arr: np.ndarray, msg, stamp, frame_id: str):
    """Fill an Image message (passed in) from a BGR uint8 array."""
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * 3
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg
