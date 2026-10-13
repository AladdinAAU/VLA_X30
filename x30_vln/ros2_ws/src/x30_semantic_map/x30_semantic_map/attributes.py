"""Per-object attributes derived from geometry, thermal statistics and image crops (no ROS imports)."""
from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .config import SizeCfg, ThermalCfg

DEFAULT_PALETTE: dict[str, tuple[int, int, int]] = {
    'black': (20, 20, 20), 'white': (240, 240, 240), 'gray': (128, 128, 128),
    'red': (200, 30, 30), 'orange': (240, 130, 20), 'yellow': (235, 210, 40),
    'green': (40, 150, 60), 'blue': (40, 80, 200), 'purple': (130, 60, 160),
    'pink': (240, 150, 190), 'brown': (110, 70, 40), 'beige': (215, 195, 155),
}


# ----------------------------------------------------------------------------------------------- images
def decode_jpeg(data: bytes) -> np.ndarray | None:
    """Decode to an RGB uint8 array, or None when the bytes are not an image."""
    if not data:
        return None
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return buf.tobytes()


def shrink_jpeg(data: bytes, max_px: int, quality: int = 85) -> bytes:
    """Return a JPEG whose longer side is at most max_px (the original if already smaller or undecodable)."""
    rgb = decode_jpeg(data)
    if rgb is None:
        return data
    h, w = rgb.shape[:2]
    scale = max_px / max(h, w)
    if scale >= 1.0:
        return data
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return encode_jpeg(small, quality)


def ahash(rgb: np.ndarray) -> int:
    """64-bit average hash; near-identical views have a small Hamming distance."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def sharpness(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = 96.0 / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def view_quality(score: float, box_area_px: float, sharp: float, sharpness_weight: float) -> float:
    """Larger, sharper, more confident crops are better views. Sharpness is squashed into [0, 1]."""
    sharp_term = 1.0 - math.exp(-sharp / 150.0)
    size_term = math.sqrt(max(box_area_px, 1.0)) / 100.0
    return float(score * size_term * (1.0 - sharpness_weight + sharpness_weight * sharp_term))


# ----------------------------------------------------------------------------------------------- colour
def _palette_arrays(palette: dict[str, Sequence[int]] | None):
    pal = palette or DEFAULT_PALETTE
    names = list(pal)
    rgb = np.array([pal[n] for n in names], dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return names, lab


def colour_names(jpeg: bytes, palette: dict[str, Sequence[int]] | None = None, inner: float = 0.6
                 ) -> list[tuple[str, float]]:
    """Colour names of the centre of a crop with their pixel fractions, most frequent first.

    The crop is a box around the object, so only its inner `inner` fraction (per side) is used to keep most of the
    background out. Pixels are assigned to the nearest palette colour in CIE Lab.
    """
    rgb = decode_jpeg(jpeg)
    if rgb is None:
        return []
    h, w = rgb.shape[:2]
    mh, mw = int(h * (1 - inner) / 2), int(w * (1 - inner) / 2)
    core = rgb[mh:h - mh or None, mw:w - mw or None]
    if core.size == 0:
        core = rgb
    if max(core.shape[:2]) > 64:
        s = 64.0 / max(core.shape[:2])
        core = cv2.resize(core, (max(1, int(core.shape[1] * s)), max(1, int(core.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    names, ref = _palette_arrays(palette)
    lab = cv2.cvtColor(np.ascontiguousarray(core), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    d = ((lab[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    idx = d.argmin(axis=1)
    counts = np.bincount(idx, minlength=len(names)).astype(float)
    frac = counts / counts.sum()
    order = np.argsort(-frac, kind='stable')
    return [(names[i], float(frac[i])) for i in order if frac[i] > 0.0]


def dominant_colour(jpeg: bytes, palette: dict[str, Sequence[int]] | None = None) -> str:
    res = colour_names(jpeg, palette)
    return res[0][0] if res else ''


# ----------------------------------------------------------------------------------------------- size, thermal
def size_class(extent: np.ndarray, cfg: SizeCfg) -> str:
    largest = float(np.max(extent))
    for thr, name in zip(cfg.thresholds_m, cfg.names):
        if largest < thr:
            return name
    return cfg.names[-1]


def thermal_attributes(mean_c: float, max_c: float, ambient_c: float | None, cfg: ThermalCfg) -> list[str]:
    """'warm' / 'hot' from the object's thermal statistics. Returns [] for NaN or when thermal is disabled."""
    if not cfg.enabled:
        return []
    out: list[str] = []
    has_mean = math.isfinite(mean_c)
    has_max = math.isfinite(max_c)
    if has_max and max_c >= cfg.hot_c:
        out.append('hot')
    if has_mean:
        rel = ambient_c is not None and math.isfinite(ambient_c) and mean_c >= ambient_c + cfg.warm_delta_c
        if rel or mean_c >= cfg.warm_abs_c:
            out.append('warm')
    return out
