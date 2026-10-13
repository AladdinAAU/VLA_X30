"""Mode state machine, task supervision, progress watchdog and path length. No ROS imports."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable

from .geometry import wrap_pi

MODES = ("idle", "teleop", "velocity_follow", "object_nav", "goal_pose", "explore", "follow_person")
TASK_MODES = ("velocity_follow", "object_nav", "goal_pose", "explore", "follow_person")
STOP_REASONS = ("finished", "timeout", "aborted", "safety", "no_progress", "error")


class TaskHandle:
    """One running task. `cancel(reason)` is cooperative: the task polls `cancelled` and calls `finish()` when done."""

    def __init__(self, mode: str):
        self.mode = mode
        self._cancel = threading.Event()
        self._done = threading.Event()
        self.reason = ""
        self.stop_reason = "aborted"

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self, reason: str, stop_reason: str = "aborted") -> None:
        if not self._cancel.is_set():
            self.reason = reason
            self.stop_reason = stop_reason
            self._cancel.set()

    def finish(self) -> None:
        self._done.set()

    def wait_done(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait_cancelled(self, timeout: float) -> bool:
        return self._cancel.wait(timeout)


class ModeCore:
    """Owns the current mode and at most one running task.

    `set_mode` and `begin_task` cancel the running task first (waiting for it to finish), then zero the velocity, so two
    commanders never overlap. `stop_motion` and `on_mode` are callbacks supplied by the ROS node.
    """

    def __init__(self, stop_motion: Callable[[], None], on_mode: Callable[[str], None], cancel_wait_s: float = 6.0):
        self._stop_motion = stop_motion
        self._on_mode = on_mode
        self._cancel_wait = cancel_wait_s
        self._lock = threading.RLock()
        self.mode = "idle"
        self.task: TaskHandle | None = None

    def _cancel_running(self, reason: str, stop_reason: str = "aborted") -> bool:
        t = self.task
        if t is None or t.done:
            return True
        t.cancel(reason, stop_reason)
        ok = t.wait_done(self._cancel_wait)
        if self.task is t:          # the task thread may still be about to call end_task; make that a no-op
            self.task = None
        return ok

    def set_mode(self, new: str, reason: str = "") -> tuple[bool, str, str]:
        if new not in MODES:
            return False, f"unknown mode {new!r}; valid modes: {', '.join(MODES)}", self.mode
        with self._lock:
            prev = self.mode
            if new == prev and (self.task is None or self.task.done):
                return True, f"already in {new}", prev
            clean = self._cancel_running(reason or f"mode switched from {prev} to {new}")
            self._stop_motion()
            self.mode = new
            self._on_mode(new)
            msg = f"mode {prev} -> {new}" + ("" if clean else " (previous task did not stop within the cancel window)")
            return True, msg, prev

    def begin_task(self, mode: str) -> TaskHandle:
        """Start a task in `mode`. A task that is already running is preempted (cancelled and awaited)."""
        if mode not in TASK_MODES:
            raise ValueError(f"{mode!r} is not a task mode")
        with self._lock:
            self._cancel_running("preempted by a new goal")
            self._stop_motion()
            if self.mode != mode:
                self.mode = mode
                self._on_mode(mode)
            self.task = TaskHandle(mode)
            return self.task

    def end_task(self, handle: TaskHandle, return_to_idle: bool = True) -> None:
        handle.finish()
        with self._lock:
            if self.task is handle:
                self.task = None
                self._stop_motion()
                if return_to_idle and self.mode in TASK_MODES:
                    self.mode = "idle"
                    self._on_mode("idle")

    def cancel_current(self, reason: str, stop_reason: str = "aborted") -> None:
        t = self.task
        if t is not None and not t.done:
            t.cancel(reason, stop_reason)


# ------------------------------------------------------------------ progress and distance


@dataclass
class ProgressWatchdog:
    """Declares a stall when the robot has neither moved nor rotated enough, and nothing else made progress, for `window_s`."""
    window_s: float
    min_distance: float = 0.15
    min_yaw: float = math.radians(20.0)
    _ref: tuple[float, float, float] | None = None
    _t_ref: float = 0.0

    def reset(self, now: float, pose: tuple[float, float, float] | None) -> None:
        self._ref, self._t_ref = pose, now

    def note_progress(self, now: float, pose: tuple[float, float, float] | None = None) -> None:
        """Something other than odometry advanced (agent turn, new skill, feedback distance decreasing)."""
        self._t_ref = now
        if pose is not None:
            self._ref = pose

    def update(self, now: float, pose: tuple[float, float, float] | None) -> bool:
        """Feed the current pose; True if stalled for more than window_s."""
        if pose is None:
            return False
        if self._ref is None:
            self.reset(now, pose)
            return False
        d = math.hypot(pose[0] - self._ref[0], pose[1] - self._ref[1])
        a = abs(wrap_pi(pose[2] - self._ref[2]))
        if d >= self.min_distance or a >= self.min_yaw:
            self.reset(now, pose)
            return False
        return (now - self._t_ref) > self.window_s


class PathLength:
    """Integrates odometry path length; ignores teleports larger than `max_jump` per sample."""

    def __init__(self, max_jump: float = 1.0):
        self.total = 0.0
        self._last: tuple[float, float] | None = None
        self.max_jump = max_jump

    def update(self, x: float, y: float) -> float:
        if self._last is not None:
            d = math.hypot(x - self._last[0], y - self._last[1])
            if d <= self.max_jump:
                self.total += d
        self._last = (x, y)
        return self.total

    def reset(self) -> None:
        self.total = 0.0
        self._last = None
