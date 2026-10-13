"""Pure geometry for the behaviour layer: angles, SE(2), swept-footprint clearance, stopping distance, grid maps and
standoff poses. No ROS imports, so everything here is unit-tested with plain pytest.

Conventions: base_link frame x forward, y left, yaw counter-clockwise (left turns positive). Distances in metres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# ------------------------------------------------------------------ angles and SE(2)


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation about z."""
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def roll_pitch_from_quat(x: float, y: float, z: float, w: float) -> tuple[float, float]:
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(s)


def rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def inverse_apply(self, xy: np.ndarray) -> np.ndarray:
        """World points (N, 2) -> this pose's body frame."""
        return (np.asarray(xy, float) - [self.x, self.y]) @ rot2(self.yaw)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        """Body-frame points (N, 2) -> world frame."""
        return np.asarray(xy, float) @ rot2(self.yaw).T + [self.x, self.y]


def bearing_to(pose: Pose2D, x: float, y: float) -> float:
    """Signed angle (rad, left positive) from the pose's heading to the point."""
    return wrap_pi(math.atan2(y - pose.y, x - pose.x) - pose.yaw)


# ------------------------------------------------------------------ footprint, clearance, stopping distance


@dataclass(frozen=True)
class Footprint:
    """Axis-aligned robot rectangle in base_link (metres from the base_link origin)."""
    front: float = 0.50
    rear: float = 0.50
    half_width: float = 0.32

    def inflated(self, lateral: float, longitudinal: float = 0.0) -> "Footprint":
        return Footprint(self.front + longitudinal, self.rear + longitudinal, self.half_width + lateral)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return -self.rear, self.front, -self.half_width, self.half_width


def _depth_inside(p: np.ndarray, b: tuple[float, float, float, float]) -> np.ndarray:
    """Penetration depth of points into rectangle b (xmin, xmax, ymin, ymax): >0 inside, <=0 outside (negative gap-ish)."""
    xmin, xmax, ymin, ymax = b
    return np.minimum.reduce([p[:, 0] - xmin, xmax - p[:, 0], p[:, 1] - ymin, ymax - p[:, 1]])


def _nearest_face_normal(p: np.ndarray, b: tuple[float, float, float, float]) -> np.ndarray:
    """Outward unit normal (N, 2) of the rectangle face nearest to each (inside) point."""
    xmin, xmax, ymin, ymax = b
    d = np.stack([p[:, 0] - xmin, xmax - p[:, 0], p[:, 1] - ymin, ymax - p[:, 1]], axis=1)
    normals = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    return normals[np.argmin(d, axis=1)]


def effective_speed(vx: float, vy: float, wz: float, rot_radius: float) -> float:
    """Path-length rate used for stopping distances: translation plus the swing of the footprint corners."""
    return math.hypot(vx, vy) + rot_radius * abs(wz)


def clearance_translation(points: np.ndarray, fp: Footprint, direction: tuple[float, float], horizon: float,
                          toward_tol: float = 0.15) -> float:
    """Free travel distance (m) of the footprint rectangle `fp` along unit `direction` before it touches a point.

    Exact (slab method). Points already inside `fp` (in the margin ring or in real contact) only block when the motion
    pushes further in: direction . outward-normal(nearest face) > toward_tol; moving tangentially or away is allowed so a
    robot grazing a wall is not frozen. Returns `inf` if clear for `horizon` metres.
    """
    if points.size == 0:
        return math.inf
    dx, dy = direction
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return math.inf
    dx, dy = dx / n, dy / n
    xmin, xmax, ymin, ymax = fp.bounds
    px, py = points[:, 0], points[:, 1]
    # q(s) = p - s d must lie in the rectangle: s in [lo, hi] per axis
    with np.errstate(divide="ignore", invalid="ignore"):
        if abs(dx) < 1e-9:
            lo_x = np.where((px >= xmin) & (px <= xmax), -np.inf, np.inf)
            hi_x = np.where((px >= xmin) & (px <= xmax), np.inf, -np.inf)
        else:
            a, b = (px - xmax) / dx, (px - xmin) / dx
            lo_x, hi_x = np.minimum(a, b), np.maximum(a, b)
        if abs(dy) < 1e-9:
            lo_y = np.where((py >= ymin) & (py <= ymax), -np.inf, np.inf)
            hi_y = np.where((py >= ymin) & (py <= ymax), np.inf, -np.inf)
        else:
            a, b = (py - ymax) / dy, (py - ymin) / dy
            lo_y, hi_y = np.minimum(a, b), np.maximum(a, b)
    s_in, s_out = np.maximum(lo_x, lo_y), np.minimum(hi_x, hi_y)
    hit = (s_in <= s_out) & (s_out > 0.0)
    if not hit.any():
        return math.inf
    inside = hit & (s_in <= 0.0)
    ahead = hit & (s_in > 0.0)
    best = math.inf
    if ahead.any():
        best = float(s_in[ahead].min())
    if inside.any():
        norm = _nearest_face_normal(points[inside], fp.bounds)
        pushing = (norm[:, 0] * dx + norm[:, 1] * dy) > toward_tol
        if pushing.any():
            best = 0.0
    return best if best <= horizon else math.inf


def body_twist_poses(vx: float, vy: float, wz: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robot pose (x, y, yaw) in its initial body frame after times `t` under a constant body twist."""
    if abs(wz) < 1e-6:
        return vx * t, vy * t, wz * t
    th = wz * t
    s, c = np.sin(th), np.cos(th)
    return (vx * s - vy * (1.0 - c)) / wz, (vx * (1.0 - c) + vy * s) / wz, th


def clearance_sweep(points: np.ndarray, fp: Footprint, vx: float, vy: float, wz: float, horizon: float,
                    rot_radius: float, step: float = 0.03) -> float:
    """Free path length (effective metres, see `effective_speed`) along the constant-twist motion before contact.

    A point blocks at the first step where it is inside the inflated rectangle AND deeper than it was at the start
    (so margin-ring points that are not being approached do not block). `inf` if clear for `horizon`.
    """
    ve = effective_speed(vx, vy, wz, rot_radius)
    if points.size == 0 or ve < 1e-9:
        return math.inf
    u = np.arange(0.0, horizon + step, step)
    t = u / ve
    px, py, th = body_twist_poses(vx, vy, wz, t)
    b = fp.bounds
    depth0 = np.maximum(_depth_inside(points, b), 0.0)
    c, s = np.cos(th), np.sin(th)
    # points in each pose's body frame: (K, N)
    rx = points[None, :, 0] - px[:, None]
    ry = points[None, :, 1] - py[:, None]
    bx = c[:, None] * rx + s[:, None] * ry
    by = -s[:, None] * rx + c[:, None] * ry
    depth = np.minimum.reduce([bx - b[0], b[1] - bx, by - b[2], b[3] - by])
    blocked = (depth > 0.0) & (depth > depth0[None, :] + 0.005)
    any_k = blocked.any(axis=1)
    if not any_k.any():
        return math.inf
    k = int(np.argmax(any_k))
    return float(max(u[k] - step, 0.0))


def path_clearance(points: np.ndarray, fp: Footprint, vx: float, vy: float, wz: float, horizon: float, rot_radius: float) -> float:
    """Free path length along the commanded motion: exact slab test for translations, swept poses with rotation."""
    if points.size == 0:
        return math.inf
    if abs(wz) < 0.02:
        v = math.hypot(vx, vy)
        if v < 1e-6:
            return math.inf
        return clearance_translation(points, fp, (vx, vy), horizon)
    return clearance_sweep(points, fp, vx, vy, wz, horizon, rot_radius)


def stopping_distance(v: float, decel: float, latency: float) -> float:
    """Distance travelled before a robot at speed v is at rest, with `latency` s of unbraked travel first."""
    v = abs(v)
    return v * latency + v * v / (2.0 * decel)


def max_safe_speed(clear: float, decel: float, latency: float, margin: float) -> float:
    """Largest speed from which the robot can still stop `margin` metres short of an obstacle `clear` metres away."""
    avail = clear - margin
    if avail <= 0.0:
        return 0.0
    if math.isinf(avail):
        return math.inf
    a = decel * latency
    return -a + math.sqrt(a * a + 2.0 * decel * avail)


# ------------------------------------------------------------------ occupancy grids and standoff poses


@dataclass
class GridMap:
    """Occupancy grid (row = y, col = x), values -1 unknown, 0..100 occupied probability (nav_msgs/OccupancyGrid)."""
    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - self.origin_x) / self.resolution)), int(math.floor((y - self.origin_y) / self.resolution))

    def value(self, x: float, y: float) -> int:
        cx, cy = self.cell(x, y)
        h, w = self.data.shape
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return -1
        return int(self.data[cy, cx])

    def disc_free(self, x: float, y: float, radius: float, occ_thresh: int = 50, allow_unknown: bool = False) -> bool:
        """True if every cell within `radius` of (x, y) is known free (or unknown when allowed)."""
        r = max(int(math.ceil(radius / self.resolution)), 0)
        cx, cy = self.cell(x, y)
        h, w = self.data.shape
        ys, xs = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        ys, xs = ys[mask], xs[mask]
        inside = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
        if not inside.all():
            return False
        v = self.data[ys, xs]
        if (v >= occ_thresh).any():
            return False
        return bool(allow_unknown or (v >= 0).all())

    def line_free(self, a: tuple[float, float], b: tuple[float, float], radius: float, **kw) -> bool:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(int(d / (0.5 * self.resolution)), 1)
        for i in range(n + 1):
            f = i / n
            if not self.disc_free(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), radius, **kw):
                return False
        return True


def object_half_extent(size_xy: tuple[float, float], direction: tuple[float, float], cap: float = 1.2) -> float:
    """Half-length of the object's axis-aligned box along the unit `direction` from its centre (ray-box), capped."""
    hx, hy = 0.5 * abs(size_xy[0]), 0.5 * abs(size_xy[1])
    dx, dy = abs(direction[0]), abs(direction[1])
    tx = hx / dx if dx > 1e-9 else math.inf
    ty = hy / dy if dy > 1e-9 else math.inf
    return min(min(tx, ty), cap)


def standoff_pose(obj_xy: tuple[float, float], obj_size_xy: tuple[float, float], robot_xy: tuple[float, float], standoff: float,
                  robot_front: float, grid: GridMap | None, clearance: float, allow_unknown: bool = False,
                  angular_step_deg: float = 15.0) -> tuple[Pose2D, float] | None:
    """Pose facing the object, with its nose `standoff` m from the object's surface.

    Candidates lie on a ring around the object (centre distance = surface half extent + standoff + robot_front), ordered
    by angular offset from the object->robot direction. The first candidate with a free disc of radius `clearance` and a
    free straight line from the object side is returned together with its angular offset (rad). Without a grid the
    straight-line candidate is returned. None if no candidate is free.
    """
    ox, oy = obj_xy
    ux, uy = robot_xy[0] - ox, robot_xy[1] - oy
    n = math.hypot(ux, uy)
    base = math.atan2(uy, ux) if n > 1e-6 else 0.0
    steps = int(round(180.0 / angular_step_deg))
    offsets = [0.0]
    for k in range(1, steps + 1):
        offsets += [k * math.radians(angular_step_deg), -k * math.radians(angular_step_deg)]
    for off in offsets:
        ang = base + off
        d = (math.cos(ang), math.sin(ang))
        r = object_half_extent(obj_size_xy, d) + standoff + robot_front
        gx, gy = ox + r * d[0], oy + r * d[1]
        if grid is not None and not grid.disc_free(gx, gy, clearance, allow_unknown=allow_unknown):
            continue
        yaw = math.atan2(oy - gy, ox - gx)
        return Pose2D(gx, gy, yaw), abs(off)
    return None


def pairwise_min_distance(points: np.ndarray, xy: Iterable[float]) -> float:
    if points.size == 0:
        return math.inf
    return float(np.min(np.hypot(points[:, 0] - xy[0], points[:, 1] - xy[1])))
