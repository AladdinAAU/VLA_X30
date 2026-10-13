"""Pure-numpy model of the merged Livox cloud: contract layout, self-body mask, intensity, tags, per-point time.

Kept free of ROS imports so it can be unit-tested and reused by the comparison script.
"""
from __future__ import annotations

import numpy as np

# INTERFACES section 2: x,y,z,intensity float32 (0,4,8,12), tag u8 (16), line u8 (17), timestamp f64 (18), point_step 26
POINT_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'],
    'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
    'offsets': [0, 4, 8, 12, 16, 17, 18],
    'itemsize': 26,
})


def box_chord(origin: np.ndarray, dirs: np.ndarray, length: np.ndarray, box: tuple) -> np.ndarray:
    """Length of the ray segment [0, length] that lies inside the axis-aligned box (slab method).

    origin: (3,), dirs: (N,3) unit vectors, length: (N,), box: (xmin,xmax,ymin,ymax,zmin,zmax). Returns (N,).
    """
    lo = np.array(box[0::2], dtype=float)
    hi = np.array(box[1::2], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / dirs
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
    # parallel rays: inside the slab -> (-inf, inf), outside -> empty
    par = np.abs(dirs) < 1e-12
    inside = (origin >= lo) & (origin <= hi)
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    t_enter = np.maximum(tmin.max(axis=1), 0.0)
    t_exit = np.minimum(tmax.min(axis=1), length)
    return np.clip(t_exit - t_enter, 0.0, None)


def self_mask(origin: np.ndarray, dirs: np.ndarray, length: np.ndarray, boxes: dict, min_chord: float) -> np.ndarray:
    """True where the ray passes through the robot body: longer than min_chord inside any box, or its hit point lies inside one."""
    hit = np.zeros(len(dirs), dtype=bool)
    end = origin + dirs * length[:, None]
    for box in boxes.values():
        lo, hi = np.array(box[0::2]), np.array(box[1::2])
        hit |= box_chord(origin, dirs, length, tuple(box)) > min_chord
        hit |= np.all((end >= lo) & (end <= hi), axis=1) & (box_chord(origin, dirs, length, tuple(box)) > 0.0)
    return hit


def synth_intensity(rng: np.random.Generator, ranges: np.ndarray, cfg: dict, retro: np.ndarray | None = None) -> np.ndarray:
    """Synthetic reflectivity: lognormal around the real median with a mild range falloff. gz retro > 0 wins."""
    n = len(ranges)
    val = cfg['median'] * np.exp(cfg['sigma'] * rng.standard_normal(n))
    val *= 1.0 / (1.0 + (ranges / cfg['range_falloff_m']) ** 2)
    val = np.clip(val, 0.0, cfg['max'])
    if retro is not None:
        use = retro > 0
        val = np.where(use, np.clip(retro, 0.0, cfg['max']), val)
    return np.floor(val).astype(np.float32)


def sample_tags(rng: np.random.Generator, valid: np.ndarray, cfg: dict) -> np.ndarray:
    """Tag bytes drawn from the empirical distributions of the recorded bag (valid / invalid points)."""
    out = np.zeros(len(valid), dtype=np.uint8)
    for flag, key in ((True, 'valid'), (False, 'invalid')):
        sel = valid == flag
        n = int(sel.sum())
        if n == 0:
            continue
        values = np.array(list(cfg[key].keys()), dtype=np.uint8)
        probs = np.array(list(cfg[key].values()), dtype=float)
        probs = probs / probs.sum()
        out[sel] = rng.choice(values, size=n, p=probs)
    return out


def assemble(points_xyz: np.ndarray, intensity: np.ndarray, tag: np.ndarray, line: np.ndarray, timestamp_ms: np.ndarray) -> np.ndarray:
    arr = np.zeros(len(points_xyz), dtype=POINT_DTYPE)
    arr['x'], arr['y'], arr['z'] = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    arr['intensity'] = intensity
    arr['tag'] = tag
    arr['line'] = line
    arr['timestamp'] = timestamp_ms
    return arr
