"""Appearance signature of a person from the detector's crop (no ROS imports; numpy + opencv).

The signature is two colour histograms of the person's bounding-box crop: torso (upper-middle part) and legs (lower-middle part). Each is a
joint hue x saturation histogram of coloured pixels plus a value histogram of grey/white/black pixels, so 'navy scrubs, brown skin' differs
from 'red scrubs' and from 'white shirt, black trousers'. Distance = mean Bhattacharyya distance of the two parts (0 = same, 1 = disjoint).
This is a plain colour signature: it separates people in different outfits and is useless for two people dressed alike (the lock layer
handles that case by reporting the situation as ambiguous).
"""
from __future__ import annotations

import cv2
import numpy as np

H_BINS, S_BINS, V_BINS = 16, 4, 4
SAT_MIN = 45                                    # below: achromatic (grey / white / black)
PARTS = ((0.18, 0.52), (0.55, 0.92))            # torso and legs as row fractions of the crop; columns 22-78 %
DIM = H_BINS * S_BINS + V_BINS


def _part_hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h = np.zeros(DIM, np.float32)
    if len(hsv) == 0:
        return h
    chroma = hsv[:, 1] >= SAT_MIN
    hb = np.minimum((hsv[chroma, 0] / 180.0 * H_BINS).astype(int), H_BINS - 1)
    sb = np.minimum(((hsv[chroma, 1] - SAT_MIN) / (255.0 - SAT_MIN) * S_BINS).astype(int), S_BINS - 1)
    np.add.at(h, hb * S_BINS + sb, 1.0)
    vb = np.minimum((hsv[~chroma, 2] / 256.0 * V_BINS).astype(int), V_BINS - 1)
    np.add.at(h, H_BINS * S_BINS + vb, 1.0)
    total = h.sum()
    return h / total if total > 0 else h


def signature_from_bgr(img: np.ndarray) -> np.ndarray | None:
    h, w = img.shape[:2]
    if h < 24 or w < 8:
        return None
    c0, c1 = int(0.22 * w), int(0.78 * w)
    parts = [_part_hist(img[int(a * h):int(b * h), c0:c1]) for a, b in PARTS]
    return np.concatenate(parts)


def signature(jpeg: bytes) -> np.ndarray | None:
    if not jpeg:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else signature_from_bgr(img)


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Mean Bhattacharyya distance of torso and legs (0 same .. 1 disjoint); None if a signature is missing."""
    if a is None or b is None:
        return None
    d = []
    for k in range(len(PARTS)):
        pa, pb = a[k * DIM:(k + 1) * DIM], b[k * DIM:(k + 1) * DIM]
        bc = float(np.sqrt(pa * pb).sum())
        d.append(float(np.sqrt(max(0.0, 1.0 - bc))))
    return float(np.mean(d))


def blend(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    out = (1.0 - alpha) * old + alpha * new
    for k in range(len(PARTS)):
        s = out[k * DIM:(k + 1) * DIM].sum()
        if s > 0:
            out[k * DIM:(k + 1) * DIM] /= s
    return out
