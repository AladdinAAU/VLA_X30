"""Finish guard: finish(success=true) is accepted only with fresh evidence.

Evidence = the latest positive `verify` (answer yes) such that the robot has not moved more than `verify_stale_m` since,
and, when the verify was about a map object, the robot is within a distance of that object. The approach/goto status is
deliberately NOT part of the test (the drone project refused a correct finish because a later approach had failed).
After `override_after_refusals` refusals the finish is accepted anyway and the override is recorded in the result."""
from __future__ import annotations

from dataclasses import dataclass, field

from .hints import HintEngine
from .memory import HarnessMemory
from .types import AgentEnv, Pose2D


@dataclass
class GuardDecision:
    allowed: bool
    overridden: bool = False
    reason: str = ""                # why it was refused (or what the override ignored)
    message: str = ""               # text for the model
    evidence: dict = field(default_factory=dict)


class FinishGuard:
    def __init__(self, cfg, hints: HintEngine):
        self.cfg = cfg
        self.hints = hints

    def check(self, success: bool, mem: HarnessMemory, env: AgentEnv, pose: Pose2D | None) -> GuardDecision:
        if not success:
            return GuardDecision(True, evidence={"claim": "failure"})
        c = self.cfg
        if not c.require_verify:
            return GuardDecision(True, evidence={"claim": "success", "verify_required": False})
        v = mem.latest_positive_verify(c.accept_unsure_plausibility)
        missing, text = "", ""
        ev: dict = {"claim": "success"}
        if v is None:
            missing, text = "no_verify_yes", self.hints.text("finish_missing_verify")
        else:
            ev["verify"] = {"object_id": v.object_id, "answer": v.answer, "turn": v.turn, "plausibility": v.plausibility}
            if v.answer != "yes":
                ev["accepted_unsure"] = True
            moved = pose.dist(v.pose) if pose is not None and v.pose is not None else 0.0
            ev["moved_since_verify_m"] = round(moved, 2)
            if moved > c.verify_stale_m:
                missing, text = "stale_verify", self.hints.text("finish_stale", d=moved)
            elif v.object_id is not None:
                dist = self._distance(env, v.object_id, pose, v.distance_m if moved < 0.3 else None)
                ev["distance_m"] = None if dist is None else round(dist[0], 2)
                if dist is not None and dist[0] > c.max_object_distance_m and dist[1] > c.max_surface_distance_m:
                    missing, text = "too_far", self.hints.text("finish_far", d=dist[0], object_id=v.object_id)
        if not missing:
            return GuardDecision(True, evidence=ev)
        mem.finish_refusals += 1
        if mem.finish_refusals > c.override_after_refusals:
            msg = self.hints.text("finish_overridden", n=mem.finish_refusals - 1, why=missing)
            return GuardDecision(True, overridden=True, reason=missing, message=msg, evidence=ev)
        return GuardDecision(False, reason=missing, message=text, evidence=ev)

    @staticmethod
    def _distance(env: AgentEnv, object_id: int, pose: Pose2D | None, fallback: float | None):
        """(centre distance, surface distance) robot -> object, or None if unknown."""
        info = env.object_info(object_id) if pose is not None else None
        if info and pose is not None and "x" in info and "y" in info:
            centre = ((pose.x - info["x"]) ** 2 + (pose.y - info["y"]) ** 2) ** 0.5
            size = info.get("size") or [0.0, 0.0]
            half = 0.5 * min(max(float(size[0]), float(size[1])) if len(size) >= 2 else 0.0, 1.5)
            return centre, max(0.0, centre - half)
        if fallback is not None:
            return float(fallback), float(fallback)
        return None
