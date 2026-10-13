"""Segmentation backends behind one interface. Only `yoloe` is the supported default; the others are options.

  yoloe       YOLOE-seg (ultralytics): open-vocabulary boxes + masks in one pass, text prompts, no SAM needed.
  yolo_world  YOLO-World v2 (ultralytics): open-vocabulary boxes only; masks come from a refiner (SAM) or are boxes.
  fake        colour-blob segmenter for tests without a GPU or weights.

A refiner (`SamRefiner`) can turn boxes into masks or sharpen existing masks for any backend.
Heavy imports (torch, ultralytics) happen inside `load()` so that this module imports without them.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger('perception_server')


@dataclass
class Instance:
    label: str
    score: float
    box: list  # [x1, y1, x2, y2] pixels in the image the backend was given
    mask: np.ndarray | None = None  # bool (h, w) of that image


class Backend:
    name = 'base'
    has_masks = True

    def load(self) -> None:
        pass

    def prepare_vocab(self, phrases: list[str]) -> float:
        """Make `phrases` the active class list; returns milliseconds spent (0 when unchanged)."""
        return 0.0

    def predict(self, bgr: np.ndarray, phrases: list[str], conf: float, imgsz: int, iou: float, max_det: int) -> list[Instance]:
        raise NotImplementedError

    def text_embeddings(self, texts: list[str]) -> np.ndarray | None:
        return None

    def info(self) -> dict:
        return {'backend': self.name}


def _resolve_weights(weights: str, cache_dir: str) -> str:
    p = Path(weights)
    if p.is_absolute() and p.exists():
        return str(p)
    cand = Path(cache_dir) / weights
    return str(cand) if cand.exists() else weights  # else let ultralytics try to download it into the cwd


def _to_instances(res, names: list[str], want_masks: bool) -> list[Instance]:
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    sc = res.boxes.conf.cpu().numpy()
    cl = res.boxes.cls.cpu().numpy().astype(int)
    masks = None
    if want_masks and res.masks is not None:
        masks = res.masks.data.cpu().numpy() > 0.5
    return [
        Instance(names[c], float(s), [float(v) for v in b], None if masks is None else masks[i])
        for i, (b, s, c) in enumerate(zip(xyxy, sc, cl))
    ]


class YoloeBackend(Backend):
    """YOLOE-seg with a per-phrase text-embedding cache.

    The text encoder is kept loaded (ultralytics rebuilds it on every `get_text_pe` call by default, ~2 s), embeddings of
    phrases already seen are reused, and the class head is only reset when the active phrase list changes."""

    name = 'yoloe'

    def __init__(self, weights: str = 'yoloe-11l-seg.pt', cache_dir: str = '/cache', device: str = '0', half: bool = True):
        self.weights, self.cache_dir, self.device, self.half = weights, cache_dir, device, half
        self.model = None
        self._pe: dict[str, object] = {}
        self._active: list[str] | None = None

    def load(self) -> None:
        from ultralytics import YOLOE

        if Path(self.cache_dir).is_dir():
            os.chdir(self.cache_dir)  # ultralytics looks for mobileclip_blt.ts in the working directory
        self.model = YOLOE(_resolve_weights(self.weights, self.cache_dir))
        self._predict_kwargs = {'quantize': 16} if self.half else {}
        log.info('loaded %s on device %s', self.weights, self.device)

    def prepare_vocab(self, phrases: list[str]) -> float:
        if phrases == self._active:
            return 0.0
        import torch

        t0 = time.perf_counter()
        missing = [p for p in dict.fromkeys(phrases) if p not in self._pe]
        if missing:
            emb = self.model.model.get_text_pe(missing, cache_clip_model=True)  # (1, M, D)
            for p, e in zip(missing, emb[0]):
                self._pe[p] = e
        pe = torch.stack([self._pe[p] for p in phrases])[None]
        self.model.set_classes(phrases, pe)
        self._active = list(phrases)
        return (time.perf_counter() - t0) * 1000.0

    def predict(self, bgr, phrases, conf, imgsz, iou, max_det):
        res = self.model.predict(
            bgr,
            conf=conf,
            imgsz=imgsz,
            iou=iou,
            max_det=max_det,
            device=self.device,
            verbose=False,
            retina_masks=True,
            **self._predict_kwargs,
        )[0]
        return _to_instances(res, phrases, True)

    def text_embeddings(self, texts):
        import torch

        with torch.no_grad():
            emb = self.model.model.get_text_pe(texts, cache_clip_model=True, without_reprta=True)
        return emb[0].float().cpu().numpy()

    def info(self):
        d = {'backend': self.name, 'weights': self.weights, 'device': self.device, 'cached_phrases': len(self._pe)}
        try:
            import torch

            if torch.cuda.is_available():
                d['gpu'] = torch.cuda.get_device_name(0)
                d['gpu_mem_allocated_mb'] = round(torch.cuda.memory_allocated() / 2**20, 1)
                d['gpu_mem_peak_mb'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        return d


class YoloWorldBackend(Backend):
    """YOLO-World v2: boxes only. Use with the SAM refiner (PERCEPTION_REFINER=sam) to get masks."""

    name = 'yolo_world'
    has_masks = False

    def __init__(self, weights: str = 'yolov8s-worldv2.pt', cache_dir: str = '/cache', device: str = '0'):
        self.weights, self.cache_dir, self.device = weights, cache_dir, device
        self.model = None
        self._active: list[str] | None = None

    def load(self) -> None:
        from ultralytics import YOLOWorld

        if Path(self.cache_dir).is_dir():
            os.chdir(self.cache_dir)
        self.model = YOLOWorld(_resolve_weights(self.weights, self.cache_dir))

    def prepare_vocab(self, phrases):
        if phrases == self._active:
            return 0.0
        t0 = time.perf_counter()
        self.model.set_classes(phrases)
        self._active = list(phrases)
        return (time.perf_counter() - t0) * 1000.0

    def predict(self, bgr, phrases, conf, imgsz, iou, max_det):
        res = self.model.predict(bgr, conf=conf, imgsz=imgsz, iou=iou, max_det=max_det, device=self.device, verbose=False)[0]
        return _to_instances(res, phrases, False)

    def info(self):
        return {'backend': self.name, 'weights': self.weights, 'device': self.device}


class FakeBackend(Backend):
    """Test double: every saturated colour blob larger than `min_area` becomes an instance of the first phrase."""

    name = 'fake'

    def __init__(self, min_area: int = 200, **_):
        self.min_area = min_area

    def predict(self, bgr, phrases, conf, imgsz, iou, max_det):
        import cv2

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        fg = (hsv[..., 1] > 120).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= self.min_area:
                out.append(
                    Instance(phrases[0] if phrases else 'object', 0.9, [float(x), float(y), float(x + w), float(y + h)], lab == i)
                )
        return out[:max_det]


class SamRefiner:
    """Mask refinement with a SAM-class model, prompted by the boxes (ultralytics SAM / MobileSAM / SAM 2)."""

    def __init__(self, weights: str = 'sam2.1_t.pt', cache_dir: str = '/cache', device: str = '0'):
        self.weights, self.cache_dir, self.device = weights, cache_dir, device
        self.model = None

    def load(self) -> None:
        from ultralytics import SAM

        self.model = SAM(_resolve_weights(self.weights, self.cache_dir))

    def refine(self, bgr: np.ndarray, instances: list[Instance]) -> list[Instance]:
        if not instances:
            return instances
        boxes = [i.box for i in instances]
        res = self.model(bgr, bboxes=boxes, device=self.device, verbose=False)[0]
        if res.masks is None:
            return instances
        masks = res.masks.data.cpu().numpy() > 0.5
        for inst, m in zip(instances, masks):
            if m.any():
                inst.mask = m
        return instances


def make_backend(name: str, **kw) -> Backend:
    name = name.lower()
    if name == 'yoloe':
        return YoloeBackend(**{k: v for k, v in kw.items() if k in ('weights', 'cache_dir', 'device', 'half') and v is not None})
    if name == 'yolo_world':
        return YoloWorldBackend(**{k: v for k, v in kw.items() if k in ('weights', 'cache_dir', 'device') and v is not None})
    if name == 'fake':
        return FakeBackend()
    raise ValueError(f'unknown backend {name!r} (yoloe | yolo_world | fake)')
