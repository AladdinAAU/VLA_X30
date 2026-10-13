"""FastAPI application. Run:  uvicorn perception_server.app:app --host 0.0.0.0 --port 8100   (from the directory above this one)

Environment (all optional):
  PERCEPTION_BACKEND   yoloe (default) | yolo_world | fake
  PERCEPTION_WEIGHTS   weights file name (looked up in PERCEPTION_CACHE first); YOLOE_WEIGHTS is accepted too (compose passes it)
  PERCEPTION_CACHE     directory with weights and mobileclip_blt.ts, default /cache
  PERCEPTION_DEVICE    CUDA device index or 'cpu', default 0
  PERCEPTION_HALF      1/0: fp16 inference (yoloe), default 1
  PERCEPTION_REFINER   none (default) | sam: box-prompted mask refinement, requests opt in with refine=true
  PERCEPTION_REFINER_WEIGHTS  default sam2.1_t.pt
  PERCEPTION_CONFIG    path to configs/perception.yaml; its `vocabulary` is the default when a request sends none
  PERCEPTION_WARMUP    1/0: embed the default vocabulary and run one dummy frame at startup, default 1
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .backends import SamRefiner, make_backend
from .pipeline import SegmentParams, Segmenter

log = logging.getLogger('perception_server')
DEFAULT_WEIGHTS = {'yoloe': 'yoloe-11l-seg.pt', 'yolo_world': 'yolov8s-worldv2.pt'}


class SegmentRequest(BaseModel):
    image: str = Field(..., description='base64 of a JPEG or PNG')
    vocab: list[str] = Field(default_factory=list)
    conf: float = 0.25
    imgsz: int = 640
    iou: float = 0.7
    max_det: int = 100
    regions: list[list[float]] = Field(default_factory=list, description='fractions [x1,y1,x2,y2] re-run at region_imgsz')
    region_imgsz: int = 0
    masks: bool = True
    refine: bool = False


class VocabRequest(BaseModel):
    vocab: list[str]


class EmbedRequest(BaseModel):
    texts: list[str] | None = None
    image: str | None = None


def _default_vocab() -> list[str]:
    path = os.environ.get('PERCEPTION_CONFIG')
    if not path or not os.path.exists(path):
        return []
    import yaml

    with open(path, 'r', encoding='utf-8') as f:
        return list((yaml.safe_load(f) or {}).get('vocabulary', []))


def build_segmenter() -> Segmenter:
    name = os.environ.get('PERCEPTION_BACKEND', 'yoloe')
    cache = os.environ.get('PERCEPTION_CACHE', '/cache')
    backend = make_backend(
        name,
        weights=os.environ.get('PERCEPTION_WEIGHTS') or os.environ.get('YOLOE_WEIGHTS') or DEFAULT_WEIGHTS.get(name),
        cache_dir=cache,
        device=os.environ.get('PERCEPTION_DEVICE', '0'),
        half=os.environ.get('PERCEPTION_HALF', '1') == '1',
    )
    backend.load()
    refiner = None
    if os.environ.get('PERCEPTION_REFINER', 'none') == 'sam':
        refiner = SamRefiner(
            os.environ.get('PERCEPTION_REFINER_WEIGHTS', 'sam2.1_t.pt'), cache, os.environ.get('PERCEPTION_DEVICE', '0')
        )
        refiner.load()
    seg = Segmenter(backend, refiner, _default_vocab())
    if os.environ.get('PERCEPTION_WARMUP', '1') == '1' and seg.default_vocab:
        log.info('warm-up: %d default phrases', len(seg.default_vocab))
        seg.warm_vocab(seg.default_vocab)
        backend.predict(np.zeros((480, 640, 3), np.uint8), seg.default_vocab, 0.25, 640, 0.7, 10)
    return seg


def create_app(segmenter: Segmenter | None = None) -> FastAPI:
    """`segmenter` can be injected (tests); otherwise it is built from the environment at startup."""
    state: dict = {'seg': segmenter}

    @asynccontextmanager
    async def lifespan(_app):
        if state['seg'] is None:
            state['seg'] = build_segmenter()
        yield

    app = FastAPI(title='x30 perception server', lifespan=lifespan)

    def seg() -> Segmenter:
        if state['seg'] is None:
            raise HTTPException(503, 'model not loaded yet')
        return state['seg']

    @app.get('/health')
    def health():
        s = seg()
        return {
            'ok': True,
            **s.backend.info(),
            'refiner': s.refiner is not None,
            'default_vocab': len(s.default_vocab),
            'requests': s.stats['requests'],
            'last_total_ms': round(s.stats['last_total_ms'], 2),
        }

    @app.post('/segment')
    def segment(req: SegmentRequest):
        s = seg()
        try:
            return s.segment(
                req.image,
                SegmentParams(
                    req.vocab, req.conf, req.imgsz, req.iou, req.max_det, req.regions, req.region_imgsz, req.masks, req.refine
                ),
            )
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex

    @app.post('/vocab')
    def vocab(req: VocabRequest):
        """Pre-compute text embeddings for a vocabulary so the next /segment with it has no first-use latency."""
        return {'ok': True, 'phrases': len(req.vocab), 'ms': round(seg().warm_vocab(req.vocab), 2)}

    @app.post('/embed')
    def embed(req: EmbedRequest):
        """Text embeddings from the model's text encoder. Image-crop embeddings are not available with the supported
        backends (YOLOE ships only the text tower of its CLIP-like encoder): answer 501 instead of faking it."""
        if req.image is not None:
            raise HTTPException(501, 'image embeddings are not available with backend ' + seg().backend.name)
        emb = seg().embed_text(req.texts or [])
        if emb is None:
            raise HTTPException(501, 'text embeddings are not available with backend ' + seg().backend.name)
        return {'dim': int(emb.shape[1]), 'embeddings': emb.round(5).tolist()}

    return app


app = create_app()
