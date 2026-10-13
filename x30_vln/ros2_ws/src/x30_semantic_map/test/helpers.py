"""Shared builders for the tests."""
from __future__ import annotations

import numpy as np

from x30_semantic_map.config import load_config
from x30_semantic_map.datatypes import Detection, Frame
from x30_semantic_map.semantic_map import SemanticMap
from x30_semantic_map.synthetic import _crop


def make_map(**overrides) -> SemanticMap:
    cfg = load_config()
    cfg.vlm.enabled = False
    for k, v in overrides.items():
        section, key = k.split('__')
        setattr(getattr(cfg, section), key, v)
    return SemanticMap(cfg)


def det(label='chair', pos=(1.0, 1.0, 0.45), ext=(0.45, 0.45, 0.9), score=0.7, points=100, crop=b'', box=(0, 0, 80, 80),
        bearing_from=None, **kw) -> Detection:
    p = np.array(pos, float)
    if bearing_from is not None:
        b = p - np.array(bearing_from, float)
        return Detection(label, score, np.full(3, np.nan), np.zeros(3), b / np.linalg.norm(b), True, 0, crop=crop,
                         box_xyxy=box, **kw)
    return Detection(label, score, p, np.array(ext, float), np.array([1.0, 0.0, 0.0]), False, points, crop=crop,
                     box_xyxy=box, **kw)


def frame(t, dets, cam=(0.0, 0.0, 0.5), view=None) -> Frame:
    return Frame(stamp=float(t), camera_position=np.array(cam, float), detections=list(dets), view=view)


def feed(smap, dets_fn, n=10, t0=0.0, dt=0.3, cam=(0.0, 0.0, 0.5)):
    """Feed n frames; dets_fn(i) returns the detections of frame i."""
    for i in range(n):
        smap.process_frame(frame(t0 + i * dt, dets_fn(i), cam))
    return t0 + n * dt


def red_crop(rng=None) -> bytes:
    return _crop((200, 30, 30), rng or np.random.default_rng(0))
