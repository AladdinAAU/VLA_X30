"""Pinhole camera model with optional lens distortion (pure numpy, no ROS imports).

Conventions: optical frame, x right, y down, z forward. Pixel (u, v) = (column, row), u to the right.
Supported distortion models, named as in sensor_msgs/CameraInfo:
  plumb_bob   (k1, k2, p1, p2, k3)    - radial-tangential, the default for rectilinear lenses
  equidistant (k1, k2, k3, k4)        - the fisheye model of OpenCV / image_geometry
An empty or all-zero `dist` means an ideal pinhole.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple = field(default_factory=tuple)
    model: str = 'plumb_bob'
    min_z: float = 0.05  # points closer than this along the optical axis are not projectable (plumb_bob)
    max_theta_rad: float = 1.45  # equidistant: ignore rays further than ~83 deg from the optical axis

    # ------------------------------------------------------------ construction
    @classmethod
    def from_camera_info(cls, width, height, k, d=(), model='plumb_bob') -> 'CameraModel':
        """Build from the fields of sensor_msgs/CameraInfo (k is the row-major 3x3 intrinsic matrix)."""
        k = np.asarray(k, float).reshape(3, 3)
        d = tuple(float(x) for x in d)
        if not any(d):
            d = ()
        model = model or 'plumb_bob'
        if model == 'rational_polynomial':  # only the first 5 coefficients are used
            model = 'plumb_bob'
            d = d[:5]
        if model not in ('plumb_bob', 'equidistant'):
            raise ValueError(f'unsupported distortion model {model!r}')
        return cls(int(width), int(height), float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), d, model)

    @classmethod
    def from_fov(cls, width, height, hfov_rad) -> 'CameraModel':
        f = 0.5 * width / np.tan(0.5 * hfov_rad)
        return cls(int(width), int(height), f, f, 0.5 * width, 0.5 * height)

    def scaled(self, sx: float, sy: float | None = None) -> 'CameraModel':
        """Same camera seen at a different image resolution (e.g. the downscaled image sent to the server)."""
        sy = sx if sy is None else sy
        return CameraModel(
            int(round(self.width * sx)),
            int(round(self.height * sy)),
            self.fx * sx,
            self.fy * sy,
            self.cx * sx,
            self.cy * sy,
            self.dist,
            self.model,
            self.min_z,
            self.max_theta_rad,
        )

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    # ------------------------------------------------------------ projection
    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N,3) camera-frame points. Returns (uv (N,2) float, valid (N,) bool).

        `valid` only says the point is in front of the camera and finite; the caller still checks the image bounds."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        if self.model == 'equidistant' and self.dist:
            r = np.hypot(x, y)
            theta = np.arctan2(r, z)
            valid = np.isfinite(theta) & (theta < self.max_theta_rad)
            k = np.zeros(4)
            k[: len(self.dist[:4])] = self.dist[:4]
            t2 = theta * theta
            td = theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
            s = np.where(r > 1e-9, td / np.maximum(r, 1e-9), 1.0)
            xd, yd = x * s, y * s
            # an ideal-pinhole point on the axis (r=0) maps to the principal point
        else:
            valid = np.isfinite(z) & (z > self.min_z)
            zs = np.where(valid, z, 1.0)
            xn, yn = x / zs, y / zs
            if self.dist:
                k1, k2, p1, p2, k3 = (list(self.dist) + [0.0] * 5)[:5]
                r2 = xn * xn + yn * yn
                rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
                xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
                yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
            else:
                xd, yd = xn, yn
        uv = np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], axis=1)
        return uv, valid

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit viewing rays (N,3) in the camera frame for pixel coordinates (N,2); inverts the distortion iteratively."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        if self.model == 'equidistant' and self.dist:
            k = np.zeros(4)
            k[: len(self.dist[:4])] = self.dist[:4]
            rd = np.hypot(xd, yd)
            theta = rd.copy()
            for _ in range(10):
                t2 = theta * theta
                f = theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - rd
                df = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
                theta = theta - f / df
            s = np.where(rd > 1e-9, np.sin(theta) / np.maximum(rd, 1e-9), 1.0)
            v = np.stack([xd * s, yd * s, np.cos(theta)], axis=1)
        else:
            xn, yn = xd.copy(), yd.copy()
            if self.dist:
                k1, k2, p1, p2, k3 = (list(self.dist) + [0.0] * 5)[:5]
                for _ in range(8):
                    r2 = xn * xn + yn * yn
                    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
                    dx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
                    dy = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
                    xn, yn = (xd - dx) / rad, (yd - dy) / rad
            v = np.stack([xn, yn, np.ones_like(xn)], axis=1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def in_bounds(self, uv: np.ndarray) -> np.ndarray:
        return (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
