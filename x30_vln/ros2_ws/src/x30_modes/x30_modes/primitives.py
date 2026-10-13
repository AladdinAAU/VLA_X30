"""Velocity primitives (NaVILA-style language actions) as pure closed-loop controllers.

`TranslateController` and `TurnController` take the measured odometry pose each cycle and return the body-frame command.
They contain no clocks or ROS, so the accuracy logic is unit-tested against a simple robot model.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .geometry import Pose2D, rot2, wrap_pi

BEARING_WORDS: dict[str, float] = {
    "ahead": 0.0, "front": 0.0, "forward": 0.0, "front-left": 45.0, "left": 90.0, "back-left": 135.0, "behind": 180.0,
    "back": 180.0, "around": 180.0, "back-right": -135.0, "right": -90.0, "front-right": -45.0,
}

ACTIONS = ("move_forward", "move_backward", "strafe_left", "strafe_right", "turn_left", "turn_right", "stop")


def parse_bearing(value, words: dict[str, float] | None = None) -> float:
    """Direction word ('left', 'behind', 'front-right') or number / numeric string -> degrees (left positive)."""
    words = words or BEARING_WORDS
    if isinstance(value, bool):
        raise ValueError("bearing must be a word or a number")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if s in words:
        return float(words[s])
    try:
        return float(s.rstrip("deg").rstrip("°"))
    except ValueError:
        raise ValueError(f"unknown bearing {value!r}; use a number of degrees (left positive) or one of {sorted(words)}") from None


@dataclass(frozen=True)
class Primitive:
    action: str
    value: float = 0.0       # cm for moves/strafes, degrees for turns


_NUM = r"(-?\d+(?:\.\d+)?)"
_PATTERNS = [
    (re.compile(rf"(?:move\s*|go\s*)?forward\D*{_NUM}\s*(?:cm|centimet\w*)?", re.I), "move_forward"),
    (re.compile(rf"(?:move\s*|go\s*)?(?:backward|back)\D*{_NUM}\s*(?:cm|centimet\w*)?", re.I), "move_backward"),
    (re.compile(rf"(?:strafe|step|move)\s*left\D*{_NUM}", re.I), "strafe_left"),
    (re.compile(rf"(?:strafe|step|move)\s*right\D*{_NUM}", re.I), "strafe_right"),
    (re.compile(rf"turn\s*left\D*{_NUM}", re.I), "turn_left"),
    (re.compile(rf"turn\s*right\D*{_NUM}", re.I), "turn_right"),
]


def parse_primitive_text(text: str) -> Primitive:
    """'move forward 75 cm', 'turn right 30 degrees', 'strafe_left(20)', 'stop' -> Primitive. Raises ValueError."""
    t = text.strip().replace("_", " ")
    if re.search(r"\bstop\b", t, re.I) and not re.search(r"\d", t):
        return Primitive("stop")
    for pat, action in _PATTERNS:
        m = pat.search(t)
        if m:
            return Primitive(action, abs(float(m.group(1))))
    raise ValueError(f"cannot parse a velocity primitive from {text!r}; expected e.g. 'move forward 75 cm', "
                     "'turn left 30 degrees', 'strafe right 20 cm' or 'stop'")


def primitive_to_skill(p: Primitive) -> tuple[str, dict]:
    """Primitive -> (skill name, args) for the skill server."""
    if p.action == "stop":
        return "stop", {}
    if p.action.startswith("move_") or p.action.startswith("strafe_"):
        direction = {"move_forward": "forward", "move_backward": "backward", "strafe_left": "left", "strafe_right": "right"}[p.action]
        return "move_relative", {"direction": direction, "distance_cm": p.value}
    return "turn_to", {"degrees": p.value if p.action == "turn_left" else -p.value}


DIRECTIONS: dict[str, tuple[float, float]] = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TranslateController:
    """Drive `distance_m` along a body-frame direction from the start pose, holding heading and the straight line."""

    def __init__(self, start: Pose2D, direction: str, distance_m: float, v_max: float, accel: float, decel: float,
                 min_speed: float = 0.06, tol: float = 0.01, k_heading: float = 1.5, k_lateral: float = 1.2, coast_s: float = 0.2):
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
        ux, uy = DIRECTIONS[direction]
        self.start = start
        self.dir_world = rot2(start.yaw) @ [ux, uy]
        self.perp_world = rot2(start.yaw) @ [-uy, ux]
        self.distance = float(distance_m)
        self.v_max, self.accel, self.decel = v_max, accel, decel
        self.min_speed, self.tol, self.k_heading, self.k_lateral = min_speed, tol, k_heading, k_lateral
        self.coast_s = coast_s          # time constant of the locomotion lag: the robot coasts v*coast_s after a zero command
        self._v_cmd = 0.0
        self.travelled = 0.0

    def progress(self, pose: Pose2D) -> tuple[float, float]:
        d = [pose.x - self.start.x, pose.y - self.start.y]
        return float(d[0] * self.dir_world[0] + d[1] * self.dir_world[1]), float(d[0] * self.perp_world[0] + d[1] * self.perp_world[1])

    def step(self, pose: Pose2D, elapsed: float) -> tuple[tuple[float, float, float], bool]:
        along, lateral = self.progress(pose)
        self.travelled = along
        remaining = self.distance - along - self._v_cmd * self.coast_s
        if remaining <= self.tol:
            return (0.0, 0.0, 0.0), True
        v = min(self.v_max, math.sqrt(2.0 * self.decel * remaining), self.min_speed + self.accel * elapsed)
        v = max(v, min(self.min_speed, self.v_max))
        self._v_cmd = v
        vel_world = v * self.dir_world - _clip(self.k_lateral * lateral, -0.15, 0.15) * self.perp_world
        vb = rot2(-pose.yaw) @ vel_world
        wz = _clip(self.k_heading * wrap_pi(self.start.yaw - pose.yaw), -0.3, 0.3)
        return (float(vb[0]), float(vb[1]), wz), False


class AngleTracker:
    """Accumulates the unwrapped rotation from successive yaw samples."""

    def __init__(self, yaw0: float):
        self.last = yaw0
        self.total = 0.0

    def update(self, yaw: float) -> float:
        self.total += wrap_pi(yaw - self.last)
        self.last = yaw
        return self.total


class TurnController:
    """Rotate in place by a signed angle (rad, left positive) with a decelerating profile."""

    def __init__(self, yaw0: float, delta: float, w_max: float, accel: float, decel: float, min_speed: float = 0.12,
                 tol: float = math.radians(2.0), coast_s: float = 0.2):
        self.tracker = AngleTracker(yaw0)
        self.delta = float(delta)
        self.w_max, self.accel, self.decel, self.min_speed, self.tol = w_max, accel, decel, min_speed, tol
        self.coast_s = coast_s
        self._w_cmd = 0.0

    @property
    def turned(self) -> float:
        return self.tracker.total

    def step(self, yaw: float, elapsed: float) -> tuple[tuple[float, float, float], bool]:
        turned = self.tracker.update(yaw)
        remaining = self.delta - turned - self._w_cmd * self.coast_s
        if abs(remaining) <= self.tol or (self.delta != 0.0 and remaining * self.delta < 0.0 and abs(remaining) <= 2.0 * self.tol):
            return (0.0, 0.0, 0.0), True
        w = min(self.w_max, math.sqrt(2.0 * self.decel * abs(remaining)), self.min_speed + self.accel * elapsed)
        w = max(w, min(self.min_speed, self.w_max))
        self._w_cmd = math.copysign(w, remaining)
        return (0.0, 0.0, self._w_cmd), False
