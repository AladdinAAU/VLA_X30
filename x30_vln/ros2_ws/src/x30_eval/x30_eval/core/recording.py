"""Data recorded during one episode (kept as plain lists so it serialises to JSON) and the task result."""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TaskResult:
    """Result of the /x30/navigate action as seen by the runner (or set by the runner on timeout)."""
    success: bool = False                 # the system claims success
    stop_reason: str = "error"            # finished | timeout | aborted | safety | no_progress | error | evaluator_timeout
    message: str = ""
    path_length_m: float = 0.0            # as reported by the system (we recompute from GT)
    duration_s: float = 0.0
    agent_turns: int = 0
    stopped_by_system: bool = False       # the task ended before the evaluator's time limit

    @property
    def stop_issued(self) -> bool:
        """The system explicitly ended the task by itself ('finished'); timeouts, aborts and errors are not stops."""
        return self.stop_reason == "finished"


@dataclass
class Recording:
    """Everything the scorer needs. Times are seconds since the task goal was sent (node clock: sim time in sim)."""
    gt: list = field(default_factory=list)            # [t, x, y, yaw] world frame, from /gt/pose
    pairs: list = field(default_factory=list)         # [t, slam_x, slam_y, slam_yaw, gt_x, gt_y, gt_yaw]
    odom_pairs: list = field(default_factory=list)    # same layout, odom-frame pose of the active odometry vs GT
    safety: list = field(default_factory=list)        # {t, stopped, limiting, reason, min_dist}
    contacts: list = field(default_factory=list)      # times of contact events reported by the sim (optional)
    contacts_available: bool = False                  # True when the sim publishes contacts (an empty list then means none)
    interventions: list = field(default_factory=list) # {t, kind}  teleop | estop | mode_change
    feedback: list = field(default_factory=list)      # {t, status, skill, dist, turn}
    map_objects: list = field(default_factory=list)   # final /semantic_map/objects, map frame
    gt_objects: list = field(default_factory=list)    # /gt/objects, world frame
    map_to_world: dict | None = None                  # SE2 map->world measured at the start from TF + GT (for goal conversion)
    scene_info: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)        # {t, path, gt:[x,y,yaw]} when frame recording is enabled
    detections: list = field(default_factory=list)    # {t, image_stamp, items:[{label,score,box}]} when enabled
    notes: list = field(default_factory=list)         # evaluator remarks (e.g. "map_to_world assumed identity")
    t_end: float = 0.0                                # time the task ended (system stop, cancel or evaluator timeout)

    def gt_array(self) -> np.ndarray:
        return np.asarray(self.gt, float).reshape(-1, 4)

    def pairs_array(self) -> np.ndarray:
        return np.asarray(self.pairs, float).reshape(-1, 7)

    def odom_pairs_array(self) -> np.ndarray:
        return np.asarray(self.odom_pairs, float).reshape(-1, 7)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls(**json.load(f))
