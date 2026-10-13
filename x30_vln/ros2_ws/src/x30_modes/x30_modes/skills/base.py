"""Skill configuration, execution context and result types shared by all skills."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SkillConfig:
    """Parameters of the skill server (flat names so they can be set from a ROS parameter file)."""
    # frames and topics
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    map_frame: str = "map"
    map_topic: str = "/map"
    costmap_topic: str = "/global_costmap/costmap"
    objects_topic: str = "/semantic_map/objects"
    rgb_topic: str = "/camera/color/image_raw"
    nav_cmd_topic: str = "/cmd_vel_nav"
    nav2_action: str = "navigate_to_pose"
    frontier_service: str = "/nav/next_frontier"
    query_service: str = "/semantic_map/query"
    summary_service: str = "/semantic_map/summary"
    obstacle_source: str = "costmap_cloud"
    costmap_cloud_topic: str = "/costmap_cloud"
    scan_topic: str = "/scan"
    z_min: float = -0.25
    z_max: float = 0.70
    obstacle_max_age_s: float = 1.0
    # robot footprint (base_link), used for standoffs; must match the safety guard
    front: float = 0.50
    rear: float = 0.50
    half_width: float = 0.32
    self_margin: float = 0.06           # points within this distance of the footprint are the robot's own body (as in the guard)
    # velocity primitives (closed loop on odometry)
    lin_speed: float = 0.30
    back_speed: float = 0.15
    strafe_speed: float = 0.15
    turn_speed: float = 0.50
    lin_accel: float = 0.40
    lin_decel: float = 0.50
    turn_accel: float = 0.80
    turn_decel: float = 1.00
    coast_s: float = 0.25               # locomotion lag: distance still travelled after a zero command = speed * coast_s
    move_tol_m: float = 0.015
    turn_tol_deg: float = 2.0
    primitive_blocked_abort_s: float = 1.5
    primitive_rate_hz: float = 20.0
    # planner navigation
    goal_tolerance_m: float = 0.5
    nav_blocked_abort_s: float = 6.0
    nav_stuck_s: float = 30.0
    nav_stuck_distance_m: float = 0.25
    clearance_radius_m: float = 0.55    # free disc required around a standoff pose
    allow_unknown_for_standoff: bool = False
    # approach
    min_standoff_m: float = 0.50
    approach_speed: float = 0.20
    approach_far_range_m: float = 4.0
    approach_tol_m: float = 0.07
    approach_bearing_tol_deg: float = 8.0
    # exploration and search
    explore_goal_timeout_s: float = 120.0
    explore_scan_step_deg: float = 90.0
    explore_scan_dwell_s: float = 1.0
    failed_frontier_ttl_s: float = 600.0
    max_consecutive_failures: int = 3
    confident_instance: float = 0.3
    search_bearing_hops: int = 3
    # verify
    vlm_timeout_s: float = 40.0
    vlm_image_max_side: int = 640
    vlm_max_tokens: int = 200
    vlm_temperature: float = 0.1
    models_config: str = ""
    skills_registry: str = ""


@dataclass
class SkillResult:
    success: bool
    message: str
    data: dict = field(default_factory=dict)


def describe_bearing(deg: float) -> str:
    """Bearing relative to the robot heading (left positive) in words for the reasoner."""
    d = (deg + 180.0) % 360.0 - 180.0
    a = abs(d)
    if a < 10:
        return "straight ahead"
    if a > 170:
        return "directly behind"
    side = "left" if d > 0 else "right"
    return f"{a:.0f} deg to the {side}" if a < 160 else f"behind to the {side} ({a:.0f} deg)"


class SkillContext:
    """Handed to a skill while it runs: cancellation, deadline, feedback and a clock."""

    def __init__(self, now: Callable[[], float], timeout_s: float, feedback: Callable[[str, float], None] | None = None,
                 cancel: threading.Event | None = None):
        self._now = now
        self.cancel_event = cancel or threading.Event()
        self.t0 = now()
        self.deadline = self.t0 + timeout_s if timeout_s > 0 else math.inf
        self._feedback = feedback
        self.timeout_s = timeout_s

    def now(self) -> float:
        return self._now()

    @property
    def elapsed(self) -> float:
        return self._now() - self.t0

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def timed_out(self) -> bool:
        return self._now() > self.deadline

    def child(self, timeout_s: float) -> "SkillContext":
        """Context with its own, shorter deadline that shares cancellation and feedback with this one."""
        return SkillContext(self._now, min(timeout_s, self.remaining()), self._feedback, self.cancel_event)

    def remaining(self) -> float:
        return max(self.deadline - self._now(), 0.0)

    def stop_reason(self) -> str:
        """'' while the skill may continue, else 'cancelled' or 'timeout'."""
        if self.cancelled:
            return "cancelled"
        if self.timed_out:
            return "timeout"
        return ""

    def sleep(self, seconds: float) -> None:
        """Wall-clock sleep that wakes early on cancellation."""
        self.cancel_event.wait(seconds)

    def feedback(self, text: str, progress: float = 0.0) -> None:
        if self._feedback:
            self._feedback(text, progress)
