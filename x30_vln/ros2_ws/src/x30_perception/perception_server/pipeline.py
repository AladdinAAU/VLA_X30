"""Request handling independent of the web framework: decode, full-frame pass, optional enlarged-region passes, merge, RLE."""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import rle
from .backends import Backend, Instance, SamRefiner


@dataclass
class SegmentParams:
    vocab: list = field(default_factory=list)
    conf: float = 0.25
    imgsz: int = 640
    iou: float = 0.7
    max_det: int = 100
    regions: list = field(default_factory=list)  # [[x1, y1, x2, y2], ...] fractions of the image, re-run enlarged
    region_imgsz: int = 0  # 0: use imgsz
    masks: bool = True
    refine: bool = False  # run the SAM refiner on the final boxes
    dedupe_iou: float = 0.5


def decode_image(b64: str) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('image is not a decodable JPEG/PNG')
    return img


def box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_instances(primary: list[Instance], extra: list[Instance], iou_thr: float) -> list[Instance]:
    """Add `extra` instances unless a same-label instance with overlapping box exists (keeps the higher score)."""
    out = list(primary)
    for e in extra:
        dup = next((o for o in out if o.label == e.label and box_iou(o.box, e.box) > iou_thr), None)
        if dup is None:
            out.append(e)
        elif e.score > dup.score:
            out[out.index(dup)] = e
    return out


def _region_pass(backend: Backend, img: np.ndarray, region, phrases, p: SegmentParams, conf: float) -> list[Instance]:
    h, w = img.shape[:2]
    x1, y1 = int(max(region[0], 0) * w), int(max(region[1], 0) * h)
    x2, y2 = int(min(region[2], 1) * w), int(min(region[3], 1) * h)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return []
    out = []
    for inst in backend.predict(
        np.ascontiguousarray(img[y1:y2, x1:x2]), phrases, conf, p.region_imgsz or p.imgsz, p.iou, p.max_det
    ):
        full = None
        if inst.mask is not None:
            full = np.zeros((h, w), bool)
            full[y1:y2, x1:x2] = inst.mask
        b = inst.box
        out.append(Instance(inst.label, inst.score, [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1], full))
    return out


def box_mask(box, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[max(int(box[1]), 0) : int(box[3]) + 1, max(int(box[0]), 0) : int(box[2]) + 1] = True
    return m


class Segmenter:
    """Serialises GPU work behind a lock and fills in timing."""

    def __init__(self, backend: Backend, refiner: SamRefiner | None = None, default_vocab: list[str] | None = None):
        self.backend, self.refiner = backend, refiner
        self.default_vocab = default_vocab or []
        self._lock = threading.Lock()
        self.stats = {'requests': 0, 'last_total_ms': 0.0, 'started': time.time()}

    def segment(self, image_b64: str, p: SegmentParams) -> dict:
        t_req = time.perf_counter()
        t = time.perf_counter()
        img = decode_image(image_b64)
        decode_ms = (time.perf_counter() - t) * 1000.0
        phrases = list(dict.fromkeys(p.vocab)) or self.default_vocab
        if not phrases:
            raise ValueError('empty vocabulary and no server default configured')
        with self._lock:
            wait_ms = (time.perf_counter() - t_req) * 1000.0 - decode_ms
            vocab_ms = self.backend.prepare_vocab(phrases)
            t = time.perf_counter()
            insts = self.backend.predict(img, phrases, p.conf, p.imgsz, p.iou, p.max_det)
            for region in p.regions:
                insts = merge_instances(insts, _region_pass(self.backend, img, region, phrases, p, p.conf), p.dedupe_iou)
            infer_ms = (time.perf_counter() - t) * 1000.0
            refine_ms = 0.0
            if p.refine and self.refiner is not None:
                t = time.perf_counter()
                insts = self.refiner.refine(img, insts)
                refine_ms = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        h, w = img.shape[:2]
        out = []
        for i in insts[: p.max_det]:
            item = {'label': i.label, 'score': round(i.score, 4), 'box': [round(v, 1) for v in i.box]}
            if p.masks:
                m = i.mask if i.mask is not None else box_mask(i.box, (h, w))
                item['mask_rle'] = rle.encode(m)
                item['mask_source'] = 'model' if i.mask is not None else 'box'
            out.append(item)
        rle_ms = (time.perf_counter() - t) * 1000.0
        total = (time.perf_counter() - t_req) * 1000.0
        self.stats['requests'] += 1
        self.stats['last_total_ms'] = total
        return {
            'instances': out,
            'ms': round(total, 2),
            'image': {'width': w, 'height': h},
            'backend': self.backend.name,
            'timing': {
                'decode_ms': round(decode_ms, 2),
                'wait_ms': round(wait_ms, 2),
                'vocab_ms': round(vocab_ms, 2),
                'infer_ms': round(infer_ms, 2),
                'refine_ms': round(refine_ms, 2),
                'rle_ms': round(rle_ms, 2),
            },
        }

    def warm_vocab(self, phrases: list[str]) -> float:
        with self._lock:
            return self.backend.prepare_vocab(list(dict.fromkeys(phrases)))

    def embed_text(self, texts: list[str]):
        with self._lock:
            return self.backend.text_embeddings(texts)
