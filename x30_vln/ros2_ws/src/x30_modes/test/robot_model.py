"""Tiny planar robot model for controller tests: first-order velocity lag and a command delay, no ROS."""
from __future__ import annotations

import math
from collections import deque

from x30_modes.geometry import Pose2D


class SimRobot:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, tau=0.15, delay=0.05, dt=0.02):
        self.x, self.y, self.yaw = x, y, yaw
        self.v = [0.0, 0.0, 0.0]
        self.tau, self.dt = tau, dt
        n = max(int(round(delay / dt)), 0)
        self.queue = deque([(0.0, 0.0, 0.0)] * n)
        self.t = 0.0

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    def step(self, cmd):
        self.queue.append(tuple(cmd))
        applied = self.queue.popleft()
        a = self.dt / (self.tau + self.dt)
        self.v = [v + a * (c - v) for v, c in zip(self.v, applied)]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * self.v[0] - s * self.v[1]) * self.dt
        self.y += (s * self.v[0] + c * self.v[1]) * self.dt
        self.yaw += self.v[2] * self.dt
        self.t += self.dt
