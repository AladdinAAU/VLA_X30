"""Plain data containers shared by the semantic map modules (no ROS imports)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NAN = float('nan')


def v3(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


@dataclass
class Detection:
    """One perception detection already expressed in the map frame (mirrors x30_msgs/Detection3D)."""
    label: str
    score: float
    centroid: np.ndarray = field(default_factory=v3)
    extent: np.ndarray = field(default_factory=v3)
    bearing: np.ndarray = field(default_factory=lambda: v3(1.0, 0.0, 0.0))
    bearing_only: bool = False
    point_count: int = 0
    thermal_mean_c: float = NAN
    thermal_max_c: float = NAN
    crop: bytes = b''                       # JPEG
    box_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    depth_source: str = 'lidar'

    @property
    def has_thermal(self) -> bool:
        return math.isfinite(self.thermal_mean_c) or math.isfinite(self.thermal_max_c)


@dataclass
class Frame:
    """All detections of one processed RGB frame."""
    stamp: float                            # seconds (image stamp)
    camera_position: np.ndarray
    detections: list[Detection]
    # optional view geometry for "should have been visible" evidence
    view: 'ViewGeometry | None' = None


@dataclass
class ViewGeometry:
    position: np.ndarray
    forward: np.ndarray                     # unit vector, map frame
    hfov_rad: float
    vfov_rad: float
    max_range_m: float = 6.0


@dataclass
class ObjectView:
    """A stored image of an object with the geometry it was taken from."""
    jpeg: bytes
    quality: float
    camera_position: np.ndarray
    stamp: float
    phash: int = 0
    size_px: tuple[int, int] = (0, 0)

    def direction_from(self, target: np.ndarray) -> np.ndarray:
        d = np.asarray(target, float) - self.camera_position
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else v3(1, 0, 0)


@dataclass
class Ray:
    id: int
    origin: np.ndarray
    direction: np.ndarray                   # unit
    label: str
    canon: str
    score: float
    stamp: float
    used_by: int | None = None              # object id that consumed this ray


@dataclass
class Region:
    """A room / corridor region of the 2D map."""
    id: int
    name: str = ''
    name_source: str = ''                   # '' | 'rules' | 'vlm' | 'shape'
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # xmin, ymin, xmax, ymax
    area_m2: float = 0.0
    width_m: float = 0.0
    length_m: float = 0.0
    vlm_name_key: str = ''                  # object-set signature the VLM name was computed for
