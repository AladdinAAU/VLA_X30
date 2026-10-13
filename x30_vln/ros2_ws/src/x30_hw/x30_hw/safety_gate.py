"""Pure logic of the real_adapter command path (no ROS): arming, interlocks, clamping, rate monitors."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

Twist = Tuple[float, float, float]


@dataclass
class GateLimits:
    vx_forward: float = 0.2
    vx_backward: float = 0.1
    vy: float = 0.1
    wz: float = 0.3
    no_lateral_above_vx: float = 0.5  # digest 5.1: no vy while vx > 0.5
    cmd_timeout_s: float = 0.4
    allow_motion: bool = True  # stage A: False


@dataclass
class Interlocks:
    """Snapshot of everything that must be true before and while velocity is forwarded."""

    estop: bool = False  # /x30/estop requested or nav_bridge estop_active
    telemetry_ok: bool = False
    robot_ready: bool = False  # standing in a gait that accepts velocity
    critical_error: bool = False  # robot error bits that must stop autonomy
    battery_percent: Optional[float] = None


class SafetyGate:
    """Default DISARMED. Arming is explicit; any interlock failure disarms and latches until the operator arms again."""

    def __init__(self, limits: GateLimits, battery_min_to_arm: float = 25.0, battery_disarm: float = 12.0):
        self.limits = limits
        self.battery_min_to_arm = battery_min_to_arm
        self.battery_disarm = battery_disarm
        self.armed = False
        self.last_disarm_reason = "never armed"
        self.clamp_events = 0
        self._stale_reported = False

    # -- arming ---------------------------------------------------------------------------------------------------
    def blocking_reason(self, il: Interlocks, for_arming: bool) -> Optional[str]:
        if not self.limits.allow_motion:
            return "stage does not allow motion (stage A: sensors only)"
        if il.estop:
            return "e-stop active"
        if not il.telemetry_ok:
            return "robot telemetry not alive"
        if il.critical_error:
            return "robot reports a critical error"
        if not il.robot_ready:
            return "robot not ready (stand it first: x30ctl stand)"
        if il.battery_percent is None:
            return "battery level unknown"
        if for_arming and il.battery_percent < self.battery_min_to_arm:
            return f"battery {il.battery_percent:.0f} % below the arming minimum {self.battery_min_to_arm:.0f} %"
        if il.battery_percent < self.battery_disarm:
            return f"battery {il.battery_percent:.0f} % below the safe minimum {self.battery_disarm:.0f} %"
        return None

    def arm(self, il: Interlocks) -> Tuple[bool, str]:
        why = self.blocking_reason(il, for_arming=True)
        if why:
            return False, "refused: " + why
        self.armed = True
        self.last_disarm_reason = ""
        return True, "armed"

    def disarm(self, reason: str) -> None:
        if self.armed:
            self.last_disarm_reason = reason
        self.armed = False

    def check_interlocks(self, il: Interlocks) -> Optional[str]:
        """Disarm (latched) on the first failing interlock; returns the reason if it disarmed just now."""
        if not self.armed:
            return None
        why = self.blocking_reason(il, for_arming=False)
        if why:
            self.disarm(why)
            return why
        return None

    # -- command path ---------------------------------------------------------------------------------------------
    def clamp(self, vx: float, vy: float, wz: float) -> Tuple[Twist, List[str]]:
        flags: List[str] = []
        vals = [vx, vy, wz]
        for i, v in enumerate(vals):
            if not math.isfinite(v):
                vals[i] = 0.0
                flags.append("non_finite")
        vx, vy, wz = vals
        lim = self.limits
        cx = min(vx, lim.vx_forward) if vx >= 0 else max(vx, -lim.vx_backward)
        cy = max(-lim.vy, min(lim.vy, vy))
        cw = max(-lim.wz, min(lim.wz, wz))
        if (cx, cy, cw) != (vx, vy, wz):
            flags.append("clamped")
            self.clamp_events += 1
        if abs(cx) > lim.no_lateral_above_vx and cy != 0.0:
            cy = 0.0
            flags.append("lateral_removed")
        return (cx, cy, cw), flags

    def evaluate(self, cmd: Twist, cmd_age_s: float) -> Tuple[Twist, str]:
        """Return (twist to forward, status). Zero twist whenever the command must not move the robot."""
        zero = (0.0, 0.0, 0.0)
        if not self.armed:
            return zero, "disarmed"
        if cmd_age_s > self.limits.cmd_timeout_s:
            return zero, "stale"
        out, flags = self.clamp(*cmd)
        return out, ("clamped" if "clamped" in flags else "ok")


class RateMonitor:
    """Message rate over a sliding window (wall seconds supplied by the caller)."""

    def __init__(self, window_s: float = 3.0):
        self.window = window_s
        self.times: Deque[float] = deque()
        self.first: Optional[float] = None
        self.total = 0

    def tick(self, t: float) -> None:
        if self.first is None:
            self.first = t
        self.total += 1
        self.times.append(t)
        self._trim(t)

    def _trim(self, now: float) -> None:
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()

    def rate(self, now: float) -> float:
        self._trim(now)
        if not self.times or self.first is None:
            return 0.0
        span = min(self.window, max(now - self.first, 1e-6))
        return len(self.times) / span

    def age(self, now: float) -> float:
        return math.inf if not self.times and self.total == 0 else now - (self.times[-1] if self.times else self.first)


def rate_level(rate: float, expected: float) -> int:
    """DiagnosticStatus level: 0 OK, 1 WARN (< 60 %), 2 ERROR (< 20 %)."""
    if expected <= 0:
        return 0
    frac = rate / expected
    return 2 if frac < 0.2 else (1 if frac < 0.6 else 0)


def clock_lag_level(lag_min_s: float) -> Tuple[int, str]:
    """Coarse clock check from robot-stamped data: lag = pc_now - stamp. Negative means the robot clock is AHEAD."""
    if lag_min_s < -0.05:
        return 2, f"robot clock appears ahead of the PC by >= {-lag_min_s:.2f} s (sync with chrony)"
    if lag_min_s > 0.5:
        return 1, f"data are {lag_min_s:.2f} s old at minimum: robot clock behind the PC or a slow link"
    return 0, f"apparent minimum lag {lag_min_s * 1000:.0f} ms"
