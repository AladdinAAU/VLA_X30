"""Point-cloud operations on a gravity-aligned cloud (z = height above the local ground). ROS-free."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pointcloud import voxel_downsample


@dataclass
class HeightBand:
    """Height rules relative to the local ground plane (metres)."""
    ground_max: float = 0.08      # points below this are ground (also drives free-space carving)
    obstacle_min: float = 0.15    # lowest height counted as obstacle (clears floor noise and door sills)
    obstacle_max: float = 0.90    # highest height counted as obstacle: robot height plus clearance, excludes ceilings
    min_range: float = 0.10       # sensor minimum range
    body_half_x: float = 0.60     # points inside this box around base_link are the robot itself (legs, body, camera)
    body_half_y: float = 0.30
    max_range: float = 12.0


def classify(level_xyz: np.ndarray, band: HeightBand) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (ground, obstacle) over points of a levelled cloud."""
    r = np.hypot(level_xyz[:, 0], level_xyz[:, 1])
    in_body = (np.abs(level_xyz[:, 0]) < band.body_half_x) & (np.abs(level_xyz[:, 1]) < band.body_half_y)
    in_range = (r >= band.min_range) & (r <= band.max_range) & ~in_body
    z = level_xyz[:, 2]
    ground = in_range & (z < band.ground_max) & (z > -0.5)
    obstacle = in_range & (z >= band.obstacle_min) & (z <= band.obstacle_max)
    return ground, obstacle


def body_hits(level_xyz: np.ndarray, band: HeightBand) -> np.ndarray:
    """Mask of points inside the body box at obstacle height: the robot's own legs, or an obstacle closer than the box.
    They are not obstacles for the costmaps (the legs would block every trajectory) but they do stop free-space carving,
    otherwise a wall nearer than the box would be carved away and the rays would run through it."""
    r = np.hypot(level_xyz[:, 0], level_xyz[:, 1])
    in_body = (np.abs(level_xyz[:, 0]) < band.body_half_x) & (np.abs(level_xyz[:, 1]) < band.body_half_y)
    z = level_xyz[:, 2]
    return in_body & (r >= band.min_range) & (z >= band.obstacle_min) & (z <= band.obstacle_max)


def obstacle_points(level_xyz: np.ndarray, band: HeightBand, voxel: float = 0.05) -> np.ndarray:
    """Voxel-downsampled obstacle points (ground and ceiling removed)."""
    _, obs = classify(level_xyz, band)
    return voxel_downsample(level_xyz[obs], voxel)[0]


def scan_from_points(level_xyz: np.ndarray, band: HeightBand, num_beams: int = 720) -> np.ndarray:
    """Minimum obstacle range per azimuth bin over [-pi, pi). Bins without a return are inf."""
    _, obs = classify(level_xyz, band)
    p = level_xyz[obs]
    ranges = np.full(num_beams, np.inf, dtype=np.float32)
    if len(p) == 0:
        return ranges
    az = np.arctan2(p[:, 1], p[:, 0])
    rng = np.hypot(p[:, 0], p[:, 1])
    bins = np.minimum(((az + np.pi) / (2.0 * np.pi) * num_beams).astype(np.int64), num_beams - 1)
    np.minimum.at(ranges, bins, rng.astype(np.float32))
    return ranges
