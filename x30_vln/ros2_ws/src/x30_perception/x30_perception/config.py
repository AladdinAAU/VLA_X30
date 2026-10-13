"""Configuration helpers shared by the client node and the tests (no ROS imports): vocabulary, thresholds, calibration."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from .camera_model import CameraModel
from .geometry import transform_from_xyzrpy


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class Vocabulary:
    """Text prompts for the open-vocabulary segmenter.

    base phrases come from configs/perception.yaml; task phrases can be added at run time (optionally with a time to live)
    and removed again. `aliases` map a prompt to the canonical label that is published (e.g. 'office chair' -> 'chair').
    Per-class confidence thresholds are looked up on the canonical label; the server is asked with the minimum of all of
    them and the client filters, because thresholds differ strongly between classes (large structures score low).
    """

    def __init__(
        self,
        base: list[str],
        aliases: dict[str, str] | None = None,
        thresholds: dict[str, float] | None = None,
        max_phrases: int = 200,
    ):
        self.base = list(dict.fromkeys(s.strip() for s in base if s and s.strip()))
        self.aliases = {k.lower(): v for k, v in (aliases or {}).items()}
        self.thresholds = {'default': 0.25, **{k.lower(): float(v) for k, v in (thresholds or {}).items()}}
        self.max_phrases = max_phrases
        self._extra: dict[str, float] = {}  # phrase -> expiry (monotonic seconds, inf = never)

    @classmethod
    def from_config(cls, cfg: dict) -> 'Vocabulary':
        return cls(cfg.get('vocabulary', []), cfg.get('aliases'), cfg.get('thresholds'), int(cfg.get('max_phrases', 200)))

    def add(self, phrases: list[str], ttl_s: float | None = None) -> None:
        exp = time.monotonic() + ttl_s if ttl_s else float('inf')
        for s in phrases:
            s = s.strip()
            if s:
                self._extra[s] = exp

    def remove(self, phrases: list[str]) -> None:
        for s in phrases:
            self._extra.pop(s.strip(), None)
            if s.strip() in self.base:
                self.base.remove(s.strip())

    def set_base(self, phrases: list[str]) -> None:
        self.base = list(dict.fromkeys(s.strip() for s in phrases if s and s.strip()))
        self._extra.clear()

    def reset(self, base: list[str]) -> None:
        """Back to the configured base list, dropping all task phrases."""
        self.set_base(base)

    def phrases(self) -> list[str]:
        now = time.monotonic()
        for k in [k for k, e in self._extra.items() if e < now]:
            del self._extra[k]
        return list(dict.fromkeys(self.base + list(self._extra)))[: self.max_phrases]

    def canonical(self, phrase: str) -> str:
        return self.aliases.get(phrase.lower(), phrase)

    def threshold(self, label: str) -> float:
        return self.thresholds.get(label.lower(), self.thresholds['default'])

    def server_conf(self) -> float:
        """Lowest threshold over the current phrases: what the server is asked for."""
        vals = [self.threshold(self.canonical(p)) for p in self.phrases()] or [self.thresholds['default']]
        return min(vals)


def load_calibration(path: str | Path) -> dict:
    """Read a camera calibration file.

    Accepts the YAML written by the ROS camera_calibration tool (image_width, image_height, camera_matrix.data,
    distortion_model, distortion_coefficients.data) or the same keys flattened as a CameraInfo dict (width, height, k, d,
    distortion_model). Optional key `lidar_to_camera_xyzrpy: [x, y, z, roll, pitch, yaw]` gives the pose of the camera
    optical frame expressed in lidar_link (used only when TF does not provide camera <- lidar).
    Returns {'camera': CameraModel, 'T_lidar_camera': 4x4 | None}."""
    d = load_yaml(path)
    w = d.get('image_width', d.get('width'))
    h = d.get('image_height', d.get('height'))
    k = d['camera_matrix']['data'] if 'camera_matrix' in d else d['k']
    dist = d['distortion_coefficients']['data'] if 'distortion_coefficients' in d else d.get('d', [])
    model = d.get('distortion_model', 'plumb_bob')
    if model == 'fisheye':
        model = 'equidistant'
    cam = CameraModel.from_camera_info(w, h, k, dist, model)
    ext = d.get('lidar_to_camera_xyzrpy')
    return {'camera': cam, 'T_lidar_camera': transform_from_xyzrpy(ext) if ext else None}
