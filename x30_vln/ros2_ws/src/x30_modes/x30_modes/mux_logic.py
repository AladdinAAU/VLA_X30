"""Velocity multiplexer logic without ROS: priority, per-source timeouts, mode gating, teleop lockout."""
from __future__ import annotations

from dataclasses import dataclass, field

Twist3 = tuple[float, float, float]
ZERO: Twist3 = (0.0, 0.0, 0.0)

# which sources may command in each mode (INTERFACES section 5: teleop > nav; idle commands nothing)
MODE_SOURCES: dict[str, tuple[str, ...]] = {
    "idle": (),
    "teleop": ("teleop",),
    "velocity_follow": ("teleop", "nav"),
    "object_nav": ("teleop", "nav"),
    "follow_person": ("teleop", "nav"),
    "goal_pose": ("teleop", "nav"),
    "explore": ("teleop", "nav"),
}


@dataclass
class SourceState:
    name: str
    priority: int                      # higher wins
    timeout_s: float
    twist: Twist3 = ZERO
    stamp: float | None = None


@dataclass
class MuxResult:
    twist: Twist3
    source: str                         # "" when nothing is commanding
    teleop_active: bool = False


class MuxCore:
    """Select the highest-priority fresh source among those the current mode allows."""

    def __init__(self, timeouts: dict[str, float] | None = None, teleop_lockout_s: float = 0.5, mode: str = "idle"):
        timeouts = timeouts or {}
        self.sources: dict[str, SourceState] = {
            "teleop": SourceState("teleop", 2, timeouts.get("teleop", 0.5)),
            "nav": SourceState("nav", 1, timeouts.get("nav", 0.5)),
        }
        self.teleop_lockout_s = teleop_lockout_s
        self.mode = mode
        self._last_teleop_active: float | None = None

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in MODE_SOURCES else "idle"

    def update(self, name: str, twist: Twist3, stamp: float) -> None:
        s = self.sources[name]
        s.twist, s.stamp = twist, stamp

    def _fresh(self, s: SourceState, now: float) -> bool:
        return s.stamp is not None and (now - s.stamp) <= s.timeout_s

    def select(self, now: float) -> MuxResult:
        allowed = MODE_SOURCES.get(self.mode, ())
        tele = self.sources["teleop"]
        # a zero teleop message is a release (joystick centred): it does not hold the priority
        teleop_active = "teleop" in allowed and self._fresh(tele, now) and tele.twist != ZERO
        if teleop_active:
            self._last_teleop_active = now
            return MuxResult(tele.twist, "teleop", True)
        nav = self.sources["nav"]
        if "nav" in allowed and self._fresh(nav, now):
            locked = self._last_teleop_active is not None and (now - self._last_teleop_active) < self.teleop_lockout_s
            if not locked:
                return MuxResult(nav.twist, "nav", False)
        return MuxResult(ZERO, "", False)
