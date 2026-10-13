"""HTTP client for the perception server (requests; no ROS imports)."""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
import requests


class ServerError(RuntimeError):
    pass


class PerceptionClient:
    def __init__(self, url: str, timeout_s: float = 5.0, jpeg_quality: int = 90):
        self.url = url.rstrip('/')
        self.timeout_s = timeout_s
        self.jpeg_quality = jpeg_quality
        self._s = requests.Session()

    def health(self, timeout_s: float = 2.0) -> dict:
        r = self._s.get(self.url + '/health', timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def segment(
        self,
        bgr: np.ndarray,
        vocab: list[str],
        conf: float,
        imgsz: int,
        regions: list | None = None,
        region_imgsz: int = 0,
        max_det: int = 100,
    ) -> tuple[dict, dict]:
        """Returns (response json, client timing {encode_ms, rtt_ms}). Raises ServerError on any failure."""
        t0 = time.perf_counter()
        ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise ServerError('jpeg encoding failed')
        body = {
            'image': base64.b64encode(jpg.tobytes()).decode('ascii'),
            'vocab': vocab,
            'conf': conf,
            'imgsz': imgsz,
            'max_det': max_det,
        }
        if regions:
            body.update(regions=regions, region_imgsz=region_imgsz)
        t1 = time.perf_counter()
        try:
            r = self._s.post(self.url + '/segment', json=body, timeout=self.timeout_s)
            r.raise_for_status()
            j = r.json()
        except (requests.RequestException, ValueError) as ex:
            raise ServerError(f'{type(ex).__name__}: {ex}') from ex
        return j, {'encode_ms': (t1 - t0) * 1000.0, 'rtt_ms': (time.perf_counter() - t1) * 1000.0}

    def set_vocab(self, vocab: list[str]) -> None:
        try:
            self._s.post(self.url + '/vocab', json={'vocab': vocab}, timeout=30).raise_for_status()
        except requests.RequestException as ex:
            raise ServerError(str(ex)) from ex
