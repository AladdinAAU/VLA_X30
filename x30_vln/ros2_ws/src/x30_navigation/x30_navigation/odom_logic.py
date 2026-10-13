"""Odometry fusion and divergence detection, ROS-free so they can be unit tested.

ComplementaryFusion: leg-odometry body velocities are integrated at their (high) rate for smoothness; each lidar
odometry pose pulls the estimate back (position and yaw gains), so leg-odometry drift is bounded by the lidar and a
lidar dropout or an ICP hiccup is bridged by the legs. Roll, pitch and height come from the lidar pose.
JumpDetector: flags impossible motion (speed above a physical limit), stale data and, for two sources, a scale
disagreement over a window.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import invert, matrix_to_rpy, rpy_to_matrix, wrap_angle


class ComplementaryFusion:
    """Planar leg-velocity propagation corrected by lidar poses. Poses are 4x4 matrices in the lidar-odometry frame."""

    def __init__(self, k_pos: float = 0.35, k_yaw: float = 0.35, gate_pos: float = 1.0, gate_yaw: float = 0.8):
        self.k_pos, self.k_yaw = k_pos, k_yaw
        self.gate_pos, self.gate_yaw = gate_pos, gate_yaw
        self.pose: np.ndarray | None = None
        self.last_leg_t: float | None = None
        self.rejected_updates = 0

    def reset(self) -> None:
        self.pose = None
        self.last_leg_t = None

    def on_leg_velocity(self, t: float, vx: float, vy: float, wz: float) -> np.ndarray | None:
        """Propagate with body-frame velocities. Returns the new pose (None until the first lidar pose arrived)."""
        if self.pose is None:
            self.last_leg_t = t
            return None
        if self.last_leg_t is not None:
            dt = t - self.last_leg_t
            if 0.0 < dt < 0.5:
                roll, pitch, yaw = matrix_to_rpy(self.pose[:3, :3])
                c, s = math.cos(yaw), math.sin(yaw)
                self.pose[0, 3] += (c * vx - s * vy) * dt
                self.pose[1, 3] += (s * vx + c * vy) * dt
                self.pose[:3, :3] = rpy_to_matrix(roll, pitch, wrap_angle(yaw + wz * dt))
        self.last_leg_t = t
        return self.pose

    def on_lidar_pose(self, pose: np.ndarray) -> np.ndarray:
        """Correct with an absolute lidar-odometry pose."""
        if self.pose is None:
            self.pose = pose.copy()
            return self.pose
        _, _, yaw_f = matrix_to_rpy(self.pose[:3, :3])
        roll, pitch, yaw_l = matrix_to_rpy(pose[:3, :3])
        dxy = pose[:2, 3] - self.pose[:2, 3]
        dyaw = wrap_angle(yaw_l - yaw_f)
        if np.linalg.norm(dxy) > self.gate_pos or abs(dyaw) > self.gate_yaw:
            self.rejected_updates += 1  # leg estimate and lidar disagree wildly: trust neither blindly, keep the lidar z/tilt
            self.pose[2, 3] = pose[2, 3]
            return self.pose
        self.pose[:2, 3] += self.k_pos * dxy
        self.pose[2, 3] = pose[2, 3]
        self.pose[:3, :3] = rpy_to_matrix(roll, pitch, wrap_angle(yaw_f + self.k_yaw * dyaw))
        return self.pose


@dataclass
class JumpConfig:
    max_speed: float = 3.0       # m/s, step speed above this is a jump (the X30 tops out around 1.5)
    max_yaw_rate: float = 6.0    # rad/s
    max_step: float = 0.8        # m, absolute single-step limit when dt is tiny
    stale_after: float = 1.0     # s without a message
    hold: float = 5.0            # s the jump flag stays set


class JumpDetector:
    """Per-source sanity checks on consecutive poses."""

    def __init__(self, cfg: JumpConfig | None = None):
        self.cfg = cfg or JumpConfig()
        self.last_pose: np.ndarray | None = None
        self.last_t: float | None = None
        self.last_jump_t: float = -1e9
        self.jumps = 0
        self.last_msg_wall: float | None = None

    def update(self, t: float, pose: np.ndarray, wall: float | None = None) -> bool:
        """Feed a pose; returns True if this step is a jump."""
        self.last_msg_wall = wall if wall is not None else t
        jump = False
        if self.last_pose is not None and self.last_t is not None:
            dt = t - self.last_t
            d = np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3])
            dyaw = abs(wrap_angle(matrix_to_rpy(pose[:3, :3])[2] - matrix_to_rpy(self.last_pose[:3, :3])[2]))
            if dt <= 1e-3:
                jump = d > self.cfg.max_step
            elif dt > 0:
                jump = d / dt > self.cfg.max_speed and d > 0.05 or dyaw / dt > self.cfg.max_yaw_rate and dyaw > 0.1
        if jump:
            self.jumps += 1
            self.last_jump_t = t
        self.last_pose, self.last_t = pose.copy(), t
        return jump

    def flagged(self, t: float) -> bool:
        return (t - self.last_jump_t) < self.cfg.hold

    def stale(self, wall: float) -> bool:
        return self.last_msg_wall is None or (wall - self.last_msg_wall) > self.cfg.stale_after


class DisagreementMonitor:
    """Compares path length and net rotation of two odometry sources over a sliding window."""

    def __init__(self, window: float = 6.0, min_travel: float = 0.8, max_ratio: float = 1.6, max_yaw_diff: float = 0.6):
        self.window, self.min_travel, self.max_ratio, self.max_yaw_diff = window, min_travel, max_ratio, max_yaw_diff
        self.a: deque = deque()
        self.b: deque = deque()
        self.last_ratio: float | None = None

    @staticmethod
    def _trim(q: deque, t: float, window: float) -> None:
        while q and t - q[0][0] > window:
            q.popleft()

    def add(self, which: str, t: float, pose: np.ndarray) -> None:
        q = self.a if which == "a" else self.b
        q.append((t, pose.copy()))
        self._trim(q, t, self.window)

    @staticmethod
    def _motion(q: deque) -> tuple[float, float]:
        if len(q) < 2:
            return 0.0, 0.0
        path = sum(np.linalg.norm(q[i + 1][1][:2, 3] - q[i][1][:2, 3]) for i in range(len(q) - 1))
        rel = invert(q[0][1]) @ q[-1][1]
        return float(path), matrix_to_rpy(rel[:3, :3])[2]

    def check(self) -> bool:
        """True when the two sources clearly disagree about how far / how much they turned."""
        pa, ya = self._motion(self.a)
        pb, yb = self._motion(self.b)
        if max(pa, pb) < self.min_travel:
            self.last_ratio = None
            return False
        ratio = max(pa, pb) / max(min(pa, pb), 1e-3)
        self.last_ratio = ratio
        return ratio > self.max_ratio or abs(wrap_angle(ya - yb)) > self.max_yaw_diff
