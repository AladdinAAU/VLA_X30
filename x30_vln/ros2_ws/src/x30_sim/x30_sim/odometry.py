"""Planar leg-odometry model fed by ground-truth poses (numpy only, unit-tested without ROS).

Maps the ground-truth base pose into an odom frame whose origin is the pose at (re)start, keeps only x, y, yaw (leg odometry of
the real X30 is planar, digest 1.2) and optionally degrades it: multiplicative scale errors drawn once per run, random errors
proportional to the step, and a slow yaw drift per metre travelled. noise_scale = 0 gives the exact (planar) ground truth.
"""
from __future__ import annotations

import math

import numpy as np


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class LegOdometry:
    def __init__(self, noise_scale: float, nominal: dict, seed: int = 1):
        self.k = float(noise_scale)
        self.nom = nominal
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Forget the origin; the next update() defines it. Redraws the per-run scale errors."""
        self.origin = None          # (x, y, yaw) of the ground truth at start
        self.prev = None            # previous ground-truth (x, y, yaw)
        self.pose = np.zeros(3)     # odom pose
        self.scale = 1.0 + self.k * self.nom['scale_error_std'] * self.rng.standard_normal()
        self.yaw_scale = 1.0 + self.k * self.nom['yaw_scale_error_std'] * self.rng.standard_normal()
        self.distance = 0.0

    def update(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """Feed a ground-truth pose, return the (noisy) odom pose (x, y, yaw)."""
        if self.origin is None:
            self.origin = (x, y, yaw)
            self.prev = (x, y, yaw)
            return 0.0, 0.0, 0.0
        px, py, pyaw = self.prev
        c, s = math.cos(pyaw), math.sin(pyaw)
        dx_w, dy_w = x - px, y - py
        dx, dy = c * dx_w + s * dy_w, -s * dx_w + c * dy_w        # step in the previous body frame
        dyaw = wrap(yaw - pyaw)
        self.prev = (x, y, yaw)
        dist = math.hypot(dx, dy)
        self.distance += dist
        if self.k > 0.0:
            n = self.nom
            frac = self.k * n['step_noise_fraction']
            dx = dx * self.scale + frac * abs(dx) * self.rng.standard_normal()
            dy = dy * self.scale + frac * abs(dy) * self.rng.standard_normal()
            drift = self.k * math.radians(n['yaw_drift_deg_per_m']) * dist
            noise = self.k * math.radians(n['yaw_noise_deg_per_m']) * dist * self.rng.standard_normal()
            dyaw = dyaw * self.yaw_scale + drift + noise
        ox, oy, oyaw = self.pose
        co, so = math.cos(oyaw), math.sin(oyaw)
        self.pose = np.array([ox + co * dx - so * dy, oy + so * dx + co * dy, wrap(oyaw + dyaw)])
        return float(self.pose[0]), float(self.pose[1]), float(self.pose[2])

    def exact_frame(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """Ground truth expressed in the start frame (what update() returns when noise_scale == 0)."""
        ox, oy, oyaw = self.origin
        c, s = math.cos(-oyaw), math.sin(-oyaw)
        return c * (x - ox) - s * (y - oy), s * (x - ox) + c * (y - oy), wrap(yaw - oyaw)
