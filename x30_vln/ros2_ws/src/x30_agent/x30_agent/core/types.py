"""Plain data types and the environment interface shared by the agents. No ROS imports here."""
from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# stop_reason values of x30_msgs/action/Navigate (INTERFACES.md section 5 / action file)
FINISHED, TIMEOUT, ABORTED, SAFETY, NO_PROGRESS, ERROR = "finished", "timeout", "aborted", "safety", "no_progress", "error"


@dataclass
class SkillResult:
    success: bool
    message: str = ""
    data: dict = field(default_factory=dict)
    duration_s: float = 0.0
    cancelled: bool = False        # the call was interrupted by the agent's cancel event
    error: str = ""                # transport-level problem (server missing, goal rejected), not a skill failure

    def to_dict(self) -> dict:
        return {"success": self.success, "message": self.message, "data": self.data, "duration_s": round(self.duration_s, 2),
                "cancelled": self.cancelled, "error": self.error}


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0       # radians, map frame, 0 = +x, positive = left

    def dist(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Task:
    instruction: str
    mode: str = "object_nav"
    target_label: str = ""
    episode_id: str = "episode"
    max_turns: int = 30
    time_budget_s: float = 300.0
    params: dict = field(default_factory=dict)         # free overrides from params_json
    reference_path: list = field(default_factory=list)   # [[x, y], ...] in the robot's map frame (dead_reckon baseline)
    goal_pose: Pose2D | None = None


@dataclass
class AgentResult:
    success: bool                 # the agent finished and claims success (NOT ground truth)
    stop_reason: str
    message: str
    turns: int = 0
    duration_s: float = 0.0
    guard_overridden: bool = False
    finish_refusals: int = 0
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"success": self.success, "stop_reason": self.stop_reason, "message": self.message, "turns": self.turns,
                "duration_s": round(self.duration_s, 2), "guard_overridden": self.guard_overridden,
                "finish_refusals": self.finish_refusals, "stats": self.stats}


class AgentEnv(ABC):
    """Everything an agent needs from the robot side. The ROS implementation lives in x30_agent.ros.ros_env;
    tests use a pure-python world. Methods must not block longer than necessary and must honour `cancel`."""

    @abstractmethod
    def run_skill(self, name: str, args: dict, timeout_s: float, cancel: threading.Event) -> SkillResult: ...

    @abstractmethod
    def pose(self) -> Pose2D | None:
        """Robot pose in the map frame, or None if unknown."""

    def map_summary(self, pose: Pose2D | None, max_objects: int = 30) -> str:
        return ""

    def map_image(self) -> Any:
        """Annotated top-down map as a numpy array / PIL image / JPEG bytes, or None."""
        return None

    def camera_image(self) -> Any:
        return None

    def recent_frames(self, n: int, spacing_s: float = 1.0) -> list:
        """Up to n recent camera frames, oldest first (the last one is the current frame)."""
        img = self.camera_image()
        return [img] if img is not None else []

    def object_info(self, object_id: int) -> dict | None:
        """Semantic map record {id, label, x, y, size:[sx,sy], ...} or None."""
        return None

    def query_map(self, op: str, **kwargs) -> dict:
        return {"ok": False, "error": "map queries are not available"}

    def now(self) -> float:
        import time
        return time.time()

    def feedback(self, status: str, skill: str = "", distance: float = float("nan"), turn: int = 0) -> None:
        pass
