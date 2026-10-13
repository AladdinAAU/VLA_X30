"""Person tracker on 2D map-frame positions (no ROS imports).

Input: person detections already lifted to 3D by lidar projection (x30_perception), one batch per processed camera frame.
Each track is an alpha-beta filter (constant velocity). Association is greedy nearest-first inside a gate that grows with the time since
the track was last updated. Detections with a weak score or too few lidar points (spurious blobs) are ignored. The detector runs at only
a few Hz, so `predict` extrapolates a track with its velocity (capped) between updates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import appearance


@dataclass
class Det:
    x: float
    y: float
    score: float
    points: int
    t: float
    sig: np.ndarray | None = None          # appearance signature of the crop (appearance.py)
    clean: bool = True                      # False when another person is close: the crop then contains pixels of both


@dataclass
class Track:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    first_t: float = 0.0
    last_t: float = 0.0
    score: float = 0.0
    points: int = 0
    sig: np.ndarray | None = None
    clean_last: bool = True                 # the last update was a clean detection (no other person close): only then is `sig` evidence
    history: list = field(default_factory=list)       # (t, x, y) of the last updates, newest last

    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class TrackerParams:
    min_score: float = 0.5
    min_points: int = 30
    gate_base_m: float = 0.7
    gate_speed_mps: float = 1.2
    max_gate_m: float = 2.0
    confirm_hits: int = 2
    drop_after_s: float = 4.0
    alpha: float = 0.6
    beta: float = 0.3
    max_speed_mps: float = 2.0
    max_extrapolate_s: float = 2.5
    merge_dist_m: float = 0.5                          # two detections of one frame closer than this are one person
    history: int = 12
    app_veto: float = 0.6                              # a detection this unlike a track in appearance is never matched to it
    overlap_m: float = 1.2                             # another person closer than this makes a detection unclean (its signature is not used)
    sig_alpha: float = 0.2


class PersonTracker:
    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    # ------------------------------------------------------------------ update
    def update(self, t: float, dets: list[Det]) -> list[Track]:
        p = self.p
        good = [d for d in dets if d.score >= p.min_score and d.points >= p.min_points]
        good.sort(key=lambda d: -d.points)
        merged: list[Det] = []
        if good:                                        # KD-tree on the detections of this frame: neighbours are found without comparing everybody
            dpts = np.array([[d.x, d.y] for d in good])
            dtree = cKDTree(dpts)
            for i, d in enumerate(good):                # overlap: the crop of a person next to another contains both of them
                d.clean = len(dtree.query_ball_point(dpts[i], p.overlap_m)) <= 1
            removed: set[int] = set()
            for i, d in enumerate(good):                # merge duplicates of the same person inside one frame (strongest detection wins)
                if i in removed:
                    continue
                merged.append(d)
                removed.update(j for j in dtree.query_ball_point(dpts[i], p.merge_dist_m) if j != i)
        pairs = []
        tlist = list(self.tracks.values())
        if merged and tlist:                            # candidates = tracks near the detection (KD-tree over the predicted track positions)
            tpts = np.array([self.predict(tr, t) for tr in tlist])
            ttree = cKDTree(tpts)
            for d in merged:
                for j in ttree.query_ball_point([d.x, d.y], p.max_gate_m):
                    tr = tlist[j]
                    dist = math.hypot(d.x - tpts[j, 0], d.y - tpts[j, 1])
                    if dist <= self._gate(tr, t):
                        da = appearance.distance(d.sig, tr.sig) if d.clean else None
                        if da is not None and da > p.app_veto:
                            continue
                        pairs.append((dist + (0.5 * da if da is not None else 0.0), d, tr))
        pairs.sort(key=lambda a: a[0])
        used_d, used_t = set(), set()
        for dist, d, tr in pairs:
            if id(d) in used_d or tr.id in used_t:
                continue
            used_d.add(id(d)), used_t.add(tr.id)
            self._correct(tr, d, t)
        for d in merged:
            if id(d) not in used_d:
                self.tracks[self.next_id] = Track(self.next_id, d.x, d.y, 0.0, 0.0, 1, t, t, d.score, d.points, d.sig if d.clean else None, d.clean, [(t, d.x, d.y)])
                self.next_id += 1
        for tid in [i for i, tr in self.tracks.items() if t - tr.last_t > p.drop_after_s]:
            del self.tracks[tid]
        return list(self.tracks.values())

    def _gate(self, tr: Track, t: float) -> float:
        dt = max(0.0, t - tr.last_t)
        return min(self.p.max_gate_m, self.p.gate_base_m + self.p.gate_speed_mps * dt)

    def _correct(self, tr: Track, d: Det, t: float) -> None:
        p = self.p
        dt = t - tr.last_t
        if dt > 1e-3:
            px, py = tr.x + tr.vx * min(dt, p.max_extrapolate_s), tr.y + tr.vy * min(dt, p.max_extrapolate_s)
            rx, ry = d.x - px, d.y - py
            tr.x, tr.y = px + p.alpha * rx, py + p.alpha * ry
            if dt > 0.05:
                tr.vx, tr.vy = tr.vx + p.beta * rx / dt, tr.vy + p.beta * ry / dt
                sp = math.hypot(tr.vx, tr.vy)
                if sp > p.max_speed_mps:
                    tr.vx, tr.vy = tr.vx * p.max_speed_mps / sp, tr.vy * p.max_speed_mps / sp
        tr.hits += 1
        tr.last_t = t
        tr.score, tr.points = d.score, d.points
        tr.clean_last = d.clean
        if d.sig is not None and d.clean:
            tr.sig = d.sig.copy() if tr.sig is None else appearance.blend(tr.sig, d.sig, p.sig_alpha)
        tr.history.append((t, tr.x, tr.y))
        del tr.history[:-p.history]

    # ------------------------------------------------------------------ queries
    def predict(self, tr: Track, t: float) -> tuple[float, float]:
        dt = min(max(0.0, t - tr.last_t), self.p.max_extrapolate_s)
        return tr.x + tr.vx * dt, tr.y + tr.vy * dt

    def confirmed(self) -> list[Track]:
        return [tr for tr in self.tracks.values() if tr.hits >= self.p.confirm_hits]

    def get(self, tid: int) -> Track | None:
        return self.tracks.get(tid)
