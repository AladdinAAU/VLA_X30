"""Safety guard logic without ROS: obstacle-aware speed limiting, hazards, acceleration limiting.

Contract (INTERFACES section 5): the guard consumes the muxed command (`/cmd_vel_raw`), obstacle points in `base_link`,
the soft e-stop and the body tilt, and produces the command for the locomotion backend at a fixed rate. The decision:

    allowed speed from the clearance c along the commanded direction:   v_stop(v) = v*latency + v^2/(2*decel) <= c - stop_margin

so the robot can always come to rest `stop_margin` short of the nearest point, provided the output decelerates at no
more than `decel` (the same limit is used by the slew limiter, which makes the bound self-consistent).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Footprint, effective_speed, max_safe_speed, path_clearance

REASON_OBSTACLE = "obstacle"
REASON_ESTOP = "estop"
REASON_STALE = "stale_cmd"
REASON_NO_LIDAR = "no_lidar"
REASON_TILT = "tilt"

Twist3 = tuple[float, float, float]
ZERO: Twist3 = (0.0, 0.0, 0.0)


@dataclass
class GuardConfig:
    # geometry of the robot (base_link) and safety margins
    front: float = 0.50
    rear: float = 0.50
    half_width: float = 0.32
    lateral_margin: float = 0.05        # extra width on each side when checking the swept footprint
    stop_margin: float = 0.20           # clearance kept between footprint edge and obstacle when stopped
    hard_stop_margin: float = 0.06      # inside this clearance the output is zeroed immediately (no ramp)
    latency_s: float = 0.25             # command-to-motion plus sensor latency
    decel: float = 1.0                  # m/s^2 assumed (and enforced) when slowing
    ang_decel: float = 2.0              # rad/s^2
    emergency_decel_factor: float = 3.0
    rot_radius: float = 0.45            # converts rad/s to equivalent m/s for stopping distances
    horizon: float = 3.0                # m of travel checked ahead
    # acceleration limits of the output (positive = speeding up), deceleration enforced via decel above
    accel_x: float = 0.8
    accel_y: float = 0.6
    accel_w: float = 1.5
    # hard speed limits
    vx_max: float = 0.6
    vx_min: float = -0.3
    vy_max: float = 0.3
    wz_max: float = 0.8
    # watchdogs
    cmd_timeout_s: float = 0.5
    obstacle_timeout_s: float = 1.0
    require_obstacles: bool = True
    max_tilt_deg: float = 35.0
    min_move_speed: float = 0.03        # effective m/s below which a limited command is replaced by zero
    self_filter: bool = True            # drop points inside the footprint grown by self_margin (own body, legs, sensor housings)
    self_margin: float = 0.06           # m; also the smallest obstacle gap that can be seen, must stay below hard_stop_margin + a few cm
    obstacle_max_range: float = 6.0     # points farther than this from base_link are ignored
    legs_x: float = 0.0                 # >0: also ignore the leg zone |x| < legs_x, legs_y_min < |y| < legs_y_max (raw lidar sees the legs beside the body)
    legs_y_min: float = 0.0
    legs_y_max: float = 0.0
    min_cell_points: int = 3            # a 10 cm cell needs this many points to count as an obstacle (rejects lidar speckle and body leaks)

    def footprint(self) -> Footprint:
        return Footprint(self.front, self.rear, self.half_width)


@dataclass
class GuardOutput:
    twist: Twist3 = ZERO
    limiting: bool = False
    stopped: bool = False
    min_obstacle_distance: float = math.inf
    scale: float = 1.0
    reason: str = ""


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def slew(cur: float, tgt: float, dt: float, accel: float, decel: float) -> float:
    """Move cur toward tgt with separate limits for speeding up (|v| growing / sign flip) and slowing down."""
    if tgt == cur:
        return cur
    if cur * tgt < 0.0:                      # reversing: brake to zero first
        step = decel * dt
        return cur - math.copysign(min(abs(cur), step), cur)
    growing = abs(tgt) > abs(cur)
    step = (accel if growing else decel) * dt
    if abs(tgt - cur) <= step:
        return tgt
    return cur + math.copysign(step, tgt - cur)


def gap_to_footprint(points: np.ndarray, fp: Footprint) -> float:
    """Smallest Euclidean distance from any point to the footprint rectangle (0 if inside)."""
    if points.size == 0:
        return math.inf
    xmin, xmax, ymin, ymax = fp.bounds
    dx = np.maximum(np.maximum(xmin - points[:, 0], points[:, 0] - xmax), 0.0)
    dy = np.maximum(np.maximum(ymin - points[:, 1], points[:, 1] - ymax), 0.0)
    return float(np.min(np.hypot(dx, dy)))


class SafetyGuardCore:
    """Stateful guard. Call `step` at a fixed rate; it returns the output command and the status fields."""

    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.out: Twist3 = ZERO

    # ------------------------------------------------------------ helpers
    def clamp_command(self, cmd: Twist3) -> Twist3:
        c = self.cfg
        return (clamp(cmd[0], c.vx_min, c.vx_max), clamp(cmd[1], -c.vy_max, c.vy_max), clamp(cmd[2], -c.wz_max, c.wz_max))

    def prepare_points(self, pts: np.ndarray | None) -> np.ndarray | None:
        """Range gate and self filter for obstacle points already expressed in base_link, shape (N, 2)."""
        if pts is None:
            return None
        c = self.cfg
        if pts.size == 0:
            return pts.reshape(0, 2)
        pts = np.asarray(pts, float)[:, :2]
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= c.obstacle_max_range]
        if c.self_filter and pts.size:
            xmin, xmax, ymin, ymax = c.footprint().inflated(c.self_margin, c.self_margin).bounds
            inside = (pts[:, 0] > xmin) & (pts[:, 0] < xmax) & (pts[:, 1] > ymin) & (pts[:, 1] < ymax)
            if c.legs_x > 0.0:
                inside |= (np.abs(pts[:, 0]) < c.legs_x) & (np.abs(pts[:, 1]) > c.legs_y_min) & (np.abs(pts[:, 1]) < c.legs_y_max)
            pts = pts[~inside]
        if c.min_cell_points > 1 and pts.shape[0] >= c.min_cell_points:
            cells = np.floor(pts / 0.1).astype(np.int64)
            key = cells[:, 0] * 100003 + cells[:, 1]
            _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
            pts = pts[counts[inv] >= c.min_cell_points]
        return pts

    def _a_eff(self) -> float:
        return min(self.cfg.decel, self.cfg.ang_decel * self.cfg.rot_radius)

    def _clearance(self, pts: np.ndarray, tw: Twist3) -> float:
        c = self.cfg
        fp = c.footprint().inflated(c.lateral_margin)
        return path_clearance(pts, fp, tw[0], tw[1], tw[2], c.horizon, c.rot_radius)

    # ------------------------------------------------------------ main step
    def step(self, now: float, dt: float, cmd: Twist3 | None, cmd_stamp: float | None, points: np.ndarray | None,
             points_stamp: float | None, estop: bool, tilt_deg: float | None) -> GuardOutput:
        c = self.cfg
        res = GuardOutput()
        cmd_c = self.clamp_command(cmd) if cmd is not None else ZERO
        wants_motion = effective_speed(*cmd_c, c.rot_radius) > 1e-3
        cmd_fresh = cmd is not None and cmd_stamp is not None and (now - cmd_stamp) <= c.cmd_timeout_s
        have_pts = points is not None and points_stamp is not None and (now - points_stamp) <= c.obstacle_timeout_s
        pts = self.prepare_points(points) if have_pts else None

        # --- hazards: immediate zero, no ramp
        hazard = ""
        if estop:
            hazard = REASON_ESTOP
        elif tilt_deg is not None and abs(tilt_deg) > c.max_tilt_deg:
            hazard = REASON_TILT
        elif not cmd_fresh:
            hazard = REASON_STALE
        elif c.require_obstacles and pts is None:
            hazard = REASON_NO_LIDAR
        if hazard:
            self.out = ZERO
            res.twist, res.reason, res.scale = ZERO, hazard, 0.0
            # estop and tilt are standing safety states; a missing lidar or a stale command only "stops" a real request
            res.stopped = hazard in (REASON_ESTOP, REASON_TILT) or (hazard == REASON_NO_LIDAR and wants_motion)
            if pts is not None:
                res.min_obstacle_distance = gap_to_footprint(pts, c.footprint())
            return res

        # --- clearance along the commanded direction and along the current output direction
        target = cmd_c
        scale = 1.0
        emergency = False
        ve_cmd = effective_speed(*cmd_c, c.rot_radius)
        ve_out = effective_speed(*self.out, c.rot_radius)
        clear_cmd = math.inf
        if pts is not None and pts.size:
            if ve_cmd > 1e-6:
                clear_cmd = self._clearance(pts, cmd_c)
                vmax = max_safe_speed(clear_cmd, self._a_eff(), c.latency_s, c.stop_margin)
                if vmax < ve_cmd:
                    scale = max(vmax / ve_cmd, 0.0)
                    res.limiting = True
                    res.reason = REASON_OBSTACLE
                    if clear_cmd <= c.hard_stop_margin:
                        emergency = True
            if ve_out > 1e-6:
                clear_out = self._clearance(pts, self.out)
                vmax_out = max_safe_speed(clear_out, self._a_eff(), c.latency_s, c.stop_margin)
                if ve_out > vmax_out + 1e-3:
                    scale = 0.0
                    emergency = True
                    res.limiting = True
                    res.reason = REASON_OBSTACLE
                    if math.isinf(clear_cmd):
                        clear_cmd = clear_out
            if ve_cmd <= 1e-6:
                res.min_obstacle_distance = gap_to_footprint(pts, c.footprint())
        if ve_cmd > 1e-6:
            res.min_obstacle_distance = clear_cmd
        target = (cmd_c[0] * scale, cmd_c[1] * scale, cmd_c[2] * scale)
        if effective_speed(*target, c.rot_radius) < c.min_move_speed and scale < 1.0:
            target = ZERO
            scale = 0.0
        res.scale = scale

        # --- slew limiter (acceleration limits; deceleration at the assumed rate, faster in an emergency)
        dec_f = c.emergency_decel_factor if emergency else 1.0
        if emergency and scale == 0.0 and clear_cmd <= c.hard_stop_margin:
            self.out = ZERO
        else:
            self.out = (slew(self.out[0], target[0], dt, c.accel_x, c.decel * dec_f),
                        slew(self.out[1], target[1], dt, c.accel_y, c.decel * dec_f),
                        slew(self.out[2], target[2], dt, c.accel_w, c.ang_decel * dec_f))
        res.twist = self.out
        res.stopped = wants_motion and effective_speed(*self.out, c.rot_radius) < 1e-3
        return res
