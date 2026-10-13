"""Thermal helpers (pure numpy). The X30 has no stock thermal camera, so every function here is optional at run time."""

from __future__ import annotations

import numpy as np

from .camera_model import CameraModel


def mono8_to_celsius(img: np.ndarray, t_min_c: float, t_max_c: float) -> np.ndarray:
    """Approximate temperature from a normalised mono8 display image when only the display range is known.

    Linear mapping 0 -> t_min, 255 -> t_max. Accuracy is limited to the camera's auto-gain behaviour: if the display range
    changes from frame to frame (auto range) the result is only a relative measure."""
    return t_min_c + img.astype(np.float32) * ((t_max_c - t_min_c) / 255.0)


def mask_pixels_to_thermal(
    mask: np.ndarray, cam_rgb: CameraModel, cam_th: CameraModel, T_th_rgb: np.ndarray, depth_m: float, max_samples: int = 1500
) -> np.ndarray:
    """Pixels (u, v) in the thermal image corresponding to the pixels of an RGB mask.

    Each mask pixel is back-projected to the object distance (`depth_m`, the median lidar depth of the object, or an
    assumed default when only a bearing is known), moved into the thermal optical frame by `T_th_rgb` (thermal <- rgb) and
    projected with the thermal intrinsics. The parallax between the two cameras (a few cm) is negligible at >1 m; for
    nearby objects the depth estimate matters."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((0, 2))
    if ys.size > max_samples:
        idx = np.linspace(0, ys.size - 1, max_samples).astype(int)
        ys, xs = ys[idx], xs[idx]
    rays = cam_rgb.rays(np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64))
    pts = rays * (depth_m / rays[:, 2])[:, None]
    pts_th = pts @ T_th_rgb[:3, :3].T + T_th_rgb[:3, 3]
    uv, valid = cam_th.project(pts_th)
    uv = uv[valid & cam_th.in_bounds(uv)]
    return uv


def thermal_stats(temp_c: np.ndarray, uv: np.ndarray, min_samples: int = 10) -> tuple[float, float]:
    """(mean, max) of the temperature image sampled at pixels `uv`; NaN, NaN with too few valid samples."""
    if uv.shape[0] == 0:
        return float('nan'), float('nan')
    u = np.clip(uv[:, 0].astype(int), 0, temp_c.shape[1] - 1)
    v = np.clip(uv[:, 1].astype(int), 0, temp_c.shape[0] - 1)
    vals = temp_c[v, u]
    vals = vals[np.isfinite(vals)]
    if vals.size < min_samples:
        return float('nan'), float('nan')
    return float(vals.mean()), float(vals.max())


def attributes_from_stats(mean_c: float, max_c: float, ambient_c: float, label: str, rules: list[dict]) -> list[str]:
    """Attributes ('warm', 'hot', ...) from thermal statistics according to the rule list in configs/perception.yaml.

    A rule is {attribute, [min_mean_c], [min_max_c], [min_delta_c], [labels]}: all given conditions must hold
    (delta = mean - ambient, ambient = median temperature of the whole thermal image). Nothing is hard-coded here;
    with NaN statistics no attribute is produced."""
    if not np.isfinite(mean_c):
        return []
    out = []
    for r in rules:
        if 'labels' in r and label not in r['labels']:
            continue
        if 'min_mean_c' in r and not mean_c >= r['min_mean_c']:
            continue
        if 'min_max_c' in r and not max_c >= r['min_max_c']:
            continue
        if 'min_delta_c' in r and not (np.isfinite(ambient_c) and mean_c - ambient_c >= r['min_delta_c']):
            continue
        out.append(r['attribute'])
    return list(dict.fromkeys(out))
