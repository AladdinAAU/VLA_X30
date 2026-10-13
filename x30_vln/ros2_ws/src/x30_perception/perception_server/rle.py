"""COCO run-length encoding of binary masks in pure numpy (no pycocotools dependency).

Format = pycocotools compressed RLE: column-major (Fortran) scan, alternating run lengths starting with a run of zeros,
packed into an ASCII string. `encode(mask)` -> {"size": [h, w], "counts": "<str>"}; `decode(rle)` -> bool (h, w).
"""

from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, bool).ravel(order='F')
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs = [0] + runs
    return runs


def _to_string(cnts: list[int]) -> str:
    out = []
    for i, x in enumerate(cnts):
        if i > 2:
            x -= cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return ''.join(out)


def _from_string(s: str) -> list[int]:
    cnts: list[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def encode(mask: np.ndarray) -> dict:
    h, w = mask.shape
    return {'size': [int(h), int(w)], 'counts': _to_string(_runs(mask))}


def decode(rle: dict) -> np.ndarray:
    h, w = rle['size']
    counts = rle['counts']
    cnts = _from_string(counts) if isinstance(counts, str) else list(counts)
    flat = np.zeros(h * w, bool)
    pos = 0
    for i, c in enumerate(cnts):
        if i % 2 == 1:
            flat[pos : pos + c] = True
        pos += c
    return flat.reshape((w, h)).T


def area(rle: dict) -> int:
    counts = rle['counts']
    cnts = _from_string(counts) if isinstance(counts, str) else list(counts)
    return int(sum(cnts[1::2]))
