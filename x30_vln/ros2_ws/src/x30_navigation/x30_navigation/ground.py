"""Ground plane estimation and gravity levelling of a lidar cloud.

The body pitches and rolls while a quadruped walks, so a fixed height slice of the cloud in `base_link` cuts
the floor into the scan. We estimate the local ground plane (and so roll/pitch and the height above ground)
from the cloud itself, optionally seeded by the IMU, and express points in a gravity-aligned frame
(`base_footprint`: x forward along the body heading, z up, origin on the ground below `base_link`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class GroundParams:
    """Tunable parameters of the ground estimator (all lengths in metres)."""
    near_min_range: float = 0.5        # ignore the robot's own body
    near_max_range: float = 4.0        # fit the plane in the near field only (local ground under and around the robot)
    low_band: float = 0.25             # points within this height above the lowest populated bin are candidates
    bin_size: float = 0.03
    min_bin_points: int = 30           # a height bin counts as populated above this many points (rejects outliers)
    inlier_dist: float = 0.04
    ransac_iters: int = 60
    min_inliers: int = 150
    max_tilt_deg: float = 30.0         # reject planes tilted more than this from the prior up direction
    max_points: int = 6000             # random subsample for RANSAC
    seed: int = 7
    smooth: float = 0.5                # weight of the new estimate in the running average (1 = no smoothing)


@dataclass
class GroundEstimate:
    normal: np.ndarray = field(default_factory=lambda: UP.copy())   # unit normal pointing up, input frame
    offset: float = 0.0                                              # plane: normal . p = offset (< 0: below origin)
    inliers: int = 0
    valid: bool = False

    @property
    def height_above_ground(self) -> float:
        """Height of the input-frame origin above the plane."""
        return -self.offset


def level_rotation(normal: np.ndarray, heading: np.ndarray | None = None) -> np.ndarray:
    """Rotation whose columns are the axes of the gravity-aligned frame expressed in the input frame.

    z = `normal`; x = `heading` (default: input x axis) projected onto the plane.
    """
    z = normal / np.linalg.norm(normal)
    h = np.array([1.0, 0.0, 0.0]) if heading is None else heading
    x = h - np.dot(h, z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-6:  # heading parallel to the normal: pick any
        x = np.array([0.0, 1.0, 0.0]) - z[1] * z
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def footprint_transform(est: GroundEstimate) -> np.ndarray:
    """4x4 pose of the levelled frame (base_footprint) expressed in the input frame (base_link)."""
    t = np.eye(4)
    t[:3, :3] = level_rotation(est.normal)
    t[:3, 3] = est.normal * est.offset  # foot of the perpendicular from the origin onto the plane
    return t


def to_level_frame(xyz: np.ndarray, est: GroundEstimate) -> np.ndarray:
    """Points in the input frame -> gravity-aligned frame (z = height above the local ground)."""
    t = footprint_transform(est)
    return (xyz - t[:3, 3]) @ t[:3, :3]


class GroundEstimator:
    """Robust near-field ground plane fit with temporal smoothing and an optional prior up direction."""

    def __init__(self, params: GroundParams | None = None):
        self.p = params or GroundParams()
        self._rng = np.random.default_rng(self.p.seed)
        self.last = GroundEstimate()

    def reset(self) -> None:
        self.last = GroundEstimate()

    def fit(self, xyz: np.ndarray, prior_up: np.ndarray | None = None) -> GroundEstimate:
        """Fit the ground plane to `xyz` (N, 3, input frame). Returns the last valid estimate (valid=False) on failure."""
        p = self.p
        up0 = UP if prior_up is None else prior_up / np.linalg.norm(prior_up)
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        cand = xyz[(r > p.near_min_range) & (r < p.near_max_range) & (np.abs(xyz[:, 2]) < 20.0)]
        if len(cand) < p.min_inliers:
            return self._fallback()
        h = cand @ up0
        hist, edges = np.histogram(h, bins=np.arange(h.min(), h.max() + p.bin_size, p.bin_size))
        populated = np.nonzero(hist >= p.min_bin_points)[0]
        if len(populated) == 0:
            return self._fallback()
        low = edges[populated[0]]
        band = cand[(h >= low - p.bin_size) & (h <= low + p.low_band)]
        if len(band) < p.min_inliers:
            return self._fallback()
        if len(band) > p.max_points:
            band = band[self._rng.choice(len(band), p.max_points, replace=False)]
        plane = self._ransac(band, up0)
        if plane is None:
            return self._fallback()
        return self._smooth(*plane)

    def _ransac(self, pts: np.ndarray, up0: np.ndarray):
        p = self.p
        idx = self._rng.integers(0, len(pts), size=(p.ransac_iters, 3))
        a, b, c = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        ok = norms > 1e-9
        normals, a = normals[ok] / norms[ok, None], a[ok]
        if len(normals) == 0:
            return None
        normals[(normals @ up0) < 0.0] *= -1.0
        cos_max = np.cos(np.radians(p.max_tilt_deg))
        good = (normals @ up0) > cos_max
        if not good.any():
            return None
        normals, a = normals[good], a[good]
        offs = np.einsum("ij,ij->i", normals, a)
        dist = np.abs(pts @ normals.T - offs[None, :])
        counts = (dist < p.inlier_dist).sum(axis=0)
        best = int(np.argmax(counts))
        if counts[best] < p.min_inliers:
            return None
        inl = pts[dist[:, best] < p.inlier_dist]
        centroid = inl.mean(axis=0)
        nrm = np.linalg.svd(inl - centroid, full_matrices=False)[2][2]
        if nrm @ up0 < 0.0:
            nrm = -nrm
        if nrm @ up0 < cos_max:
            return None
        return nrm, float(nrm @ centroid), int(counts[best])

    def _smooth(self, normal: np.ndarray, offset: float, inliers: int) -> GroundEstimate:
        a = self.p.smooth
        if self.last.valid and a < 1.0:
            normal = a * normal + (1.0 - a) * self.last.normal
            normal = normal / np.linalg.norm(normal)
            offset = a * offset + (1.0 - a) * self.last.offset
        self.last = GroundEstimate(normal=normal, offset=offset, inliers=inliers, valid=True)
        return self.last

    def _fallback(self) -> GroundEstimate:
        return GroundEstimate(normal=self.last.normal.copy(), offset=self.last.offset, inliers=0, valid=False)
