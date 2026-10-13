"""Target lock: which tracked person is the one the operator locked on (no ROS imports).

`lock()` stores the target's appearance signature, position and velocity at the moment of the operator's command. From then on `select()` is
called every cycle with all current tracks and decides, from position continuity and appearance:
  ok         exactly one candidate is plausible, or one is clearly better than the rest -> follow it
  ambiguous  two or more candidates are plausible and not separable (e.g. a person in the same outfit walks next to the target) -> the
             follower must hold still; it never guesses. Resolves by itself when the candidates separate
  lost       no candidate is plausible
A candidate is plausible if it lies within a position gate around the target's predicted position (the gate grows with the time since the last
sure sighting) and, when both signatures exist, its appearance distance is below `app_max`. Cost = position term + appearance term; the
current track gets a small bonus (stability). The signature is adapted slowly (`sig_alpha`) only on sure, well-matching sightings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import appearance
from .tracker import PersonTracker, Track


@dataclass
class LockParams:
    app_max: float = 0.45               # appearance distance above which a candidate is rejected
    app_weight: float = 1.0
    gate_base_m: float = 0.8
    gate_speed_mps: float = 1.0
    gate_max_m: float = 3.0
    margin: float = 0.20                # best and second-best cost closer than this = ambiguous
    margin_lookalike: float = 0.60      # ... when the descriptor cannot tell the two apart (distances within look_alike_delta, or one unknown)
    look_alike_delta: float = 0.10
    keep_bonus: float = 0.10
    unknown_app_cost: float = 0.5       # appearance cost when a signature is missing
    sig_alpha: float = 0.05
    sig_adapt_max: float = 0.25         # adapt the signature only when the match is this good
    recent_s: float = 1.5               # only tracks updated this recently are candidates
    max_extrapolate_s: float = 2.5
    max_candidates: int = 8             # at most this many people nearest to the predicted target position are compared by descriptor


@dataclass
class Decision:
    state: str                          # ok | ambiguous | lost
    track: Track | None = None
    cost: float = math.inf
    second_cost: float = math.inf
    candidates: int = 0


class TargetLock:
    def __init__(self, params: LockParams | None = None):
        self.p = params or LockParams()
        self.locked = False
        self.sig: np.ndarray | None = None
        self.track_id: int | None = None
        self.pos = (0.0, 0.0)
        self.vel = (0.0, 0.0)
        self.t_sure = 0.0               # time of the last sure sighting

    def lock(self, track: Track, t: float) -> None:
        self.locked, self.track_id = True, track.id
        self.sig = None if track.sig is None else track.sig.copy()
        self.pos, self.vel, self.t_sure = (track.x, track.y), (track.vx, track.vy), t

    def release(self) -> None:
        self.locked, self.sig, self.track_id = False, None, None

    def predicted(self, t: float) -> tuple[float, float]:
        dt = min(max(0.0, t - self.t_sure), self.p.max_extrapolate_s)
        return self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt

    def segment(self, t: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Where the target can be: the segment from the last sure position to the position extrapolated with its last velocity. If it stopped
        it is at the start, if it kept walking at the end: a candidate is as plausible anywhere on the segment."""
        return self.pos, self.predicted(t)

    def search_region(self, t: float) -> tuple[tuple[float, float], float]:
        """(centre, radius) of a disc around the segment plus the gate: people outside it cannot be the target."""
        (ax, ay), (bx, by) = self.segment(t)
        return ((ax + bx) / 2, (ay + by) / 2), self.gate(t) + math.hypot(bx - ax, by - ay) / 2

    @staticmethod
    def _seg_dist(pts: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        ap = pts - np.array(a)
        L2 = float(ab @ ab)
        u = np.zeros(len(pts)) if L2 < 1e-9 else np.clip(ap @ ab / L2, 0.0, 1.0)
        return np.hypot(*(pts - (np.array(a) + u[:, None] * ab)).T)

    def gate(self, t: float) -> float:
        return min(self.p.gate_max_m, self.p.gate_base_m + self.p.gate_speed_mps * max(0.0, t - self.t_sure))

    def explain(self, tracker: PersonTracker, t: float) -> list:
        """Diagnostics: [id, position distance to the prediction (m), gate (m), appearance distance or None, age (s)] of every confirmed track."""
        a, b = self.segment(t)
        out = []
        for tr in tracker.confirmed():
            px, py = tracker.predict(tr, t)
            da = appearance.distance(self.sig, tr.sig) if tr.clean_last else None
            out.append([tr.id, round(float(self._seg_dist(np.array([[px, py]]), a, b)[0]), 2), round(self.gate(t), 2), None if da is None else round(da, 2), round(t - tr.last_t, 1)])
        return out

    def select(self, tracker: PersonTracker, t: float) -> Decision:
        if not self.locked:
            return Decision('lost')
        p = self.p
        gate = self.gate(t)
        seg_a, seg_b = self.segment(t)
        (cx, cy), radius = self.search_region(t)
        scored: list[tuple[float, Track, float | None]] = []
        tracks = [tr for tr in tracker.confirmed() if t - tr.last_t <= p.recent_s]
        if not tracks:
            return Decision('lost')
        pts = np.array([tracker.predict(tr, t) for tr in tracks])
        near = cKDTree(pts).query_ball_point([cx, cy], r=radius)       # only people inside the search region around the segment
        seg_d = self._seg_dist(pts, seg_a, seg_b)
        near = sorted((j for j in near if seg_d[j] <= gate), key=lambda j: seg_d[j])[:p.max_candidates]
        for j in near:
            tr = tracks[j]
            dpos = float(seg_d[j])
            dapp = appearance.distance(self.sig, tr.sig) if tr.clean_last else None      # unclean (overlapping) update: appearance is no evidence
            if dapp is not None and dapp > p.app_max:
                continue
            cost = dpos / gate + p.app_weight * ((dapp if dapp is not None else p.unknown_app_cost * p.app_max) / p.app_max)
            if tr.id == self.track_id:
                cost -= p.keep_bonus
            scored.append((cost, tr, dapp))
        if not scored:
            return Decision('lost')
        scored.sort(key=lambda a: a[0])
        best_cost, best, dapp = scored[0]
        second = scored[1][0] if len(scored) > 1 else math.inf
        margin = p.margin
        if len(scored) > 1:                              # the descriptor gives no evidence between the two best: only a big position advantage counts
            da2 = scored[1][2]
            if dapp is None or da2 is None or abs(dapp - da2) < p.look_alike_delta:
                margin = p.margin_lookalike
        if second - best_cost < margin:
            return Decision('ambiguous', best, best_cost, second, len(scored))
        self.track_id = best.id
        self.pos, self.vel, self.t_sure = tracker.predict(best, t), (best.vx, best.vy), t
        if dapp is not None and dapp <= p.sig_adapt_max and best.sig is not None and self.sig is not None:
            self.sig = appearance.blend(self.sig, best.sig, p.sig_alpha)
        return Decision('ok', best, best_cost, second, len(scored))
