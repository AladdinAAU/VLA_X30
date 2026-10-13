"""Persistent 3D semantic object map (pure python + numpy, no ROS).

Ported and generalised from the drone project's semantic_map.py. Per object instance it keeps label votes, a
robust position/size estimate from a bounded observation history, sighting counts, thermal statistics, the best
views and VLM annotations. Detections arrive per camera frame; time is always passed in (stamps), so the map is
deterministic and unit-testable.

Lifecycle of an instance: tentative -> confirmed (enough counted observations and mean score, per category) ->
removed only when the camera repeatedly looked at its place and saw nothing (static objects), when it was not seen
for a while (dynamic objects, unconfirmed ones) or on explicit rejection. Objects seen only as bearings are
triangulated from rays taken at different robot poses.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import attributes as attr
from .config import SemanticMapConfig
from .datatypes import Detection, Frame, ObjectView, Ray, Region, ViewGeometry, v3
from .vocabulary import Vocabulary

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------------------------- geometry
def box_of(pos: np.ndarray, size: np.ndarray, floor: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    half = np.maximum(np.asarray(size, float), floor) / 2.0
    return pos - half, pos + half


def box_iou(a_pos, a_size, b_pos, b_size) -> float:
    alo, ahi = box_of(a_pos, a_size)
    blo, bhi = box_of(b_pos, b_size)
    inter = np.maximum(0.0, np.minimum(ahi, bhi) - np.maximum(alo, blo))
    vi = float(np.prod(inter))
    vu = float(np.prod(ahi - alo) + np.prod(bhi - blo) - vi)
    return vi / vu if vu > 1e-12 else 0.0


def point_ray_distance(p: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    """(perpendicular distance, signed distance along the ray) of point p."""
    v = p - origin
    s = float(v @ direction)
    return float(np.linalg.norm(v - s * direction)), s


# ----------------------------------------------------------------------------------------------- object
@dataclass
class MapObject:
    id: int
    first_seen: float
    last_seen: float
    votes: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0                          # counted observations (3D and ray sightings)
    n_3d: int = 0
    score_sum: float = 0.0
    hist: list[tuple[float, np.ndarray, np.ndarray, float]] = field(default_factory=list)   # t, pos, ext, weight
    position: np.ndarray = field(default_factory=v3)
    size: np.ndarray = field(default_factory=v3)
    point_count: int = 0
    last_counted: float = -1e9
    last_miss: float = -1e9
    misses: float = 0.0
    bearing_only: bool = False
    ray_support: int = 0
    ray_a: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))   # weighted ray normal equations
    ray_b: np.ndarray = field(default_factory=v3)
    ray_c: float = 0.0
    dynamic: bool = False
    velocity: np.ndarray = field(default_factory=v3)
    moving: bool = False
    track: list[tuple[float, np.ndarray]] = field(default_factory=list)
    extra_obs: int = 0                      # additional observations demanded (re-appearing rejected place)
    rejected: bool = False
    pinned: bool = False                    # confirmed by feedback: kept regardless of misses
    thermal_n: int = 0
    thermal_mean_sum: float = 0.0
    thermal_max: float = float('nan')
    views: list[ObjectView] = field(default_factory=list)
    colour: str = ''
    colour_key: int = -1
    caption: str = ''
    caption_quality: float = 0.0
    captions_done: int = 0
    vlm_attrs: dict[str, str] = field(default_factory=dict)
    label_ok_votes: int = 0
    label_bad_votes: int = 0
    distinct_from: set[int] = field(default_factory=set)   # VLM judged these ids as different objects

    @property
    def mean_score(self) -> float:
        return self.score_sum / max(self.n_obs, 1)

    @property
    def thermal_mean(self) -> float:
        return self.thermal_mean_sum / self.thermal_n if self.thermal_n else float('nan')


@dataclass
class FrameResult:
    created: list[int] = field(default_factory=list)
    updated: list[int] = field(default_factory=list)
    rays_added: int = 0
    triangulated: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    ignored: int = 0


# ----------------------------------------------------------------------------------------------- map
class SemanticMap:
    def __init__(self, cfg: SemanticMapConfig | None = None):
        self.cfg = cfg or SemanticMapConfig()
        self.vocab = Vocabulary(self.cfg)
        self.lock = threading.RLock()
        self.los_fn: Callable[[np.ndarray, np.ndarray], bool] | None = None   # optional occlusion test (2D grid)
        self.room_map: Any = None                                              # rooms.RoomMap (duck typed)
        self.regions: dict[int, Region] = {}
        self.places: dict = {}                       # taught places (places.py), saved with the map
        self.next_place_id = 1
        self.reset()

    # ------------------------------------------------------------------------------------ state
    def reset(self) -> None:
        with self.lock:
            self.objects: dict[int, MapObject] = {}
            self.rays: list[Ray] = []
            self.rejected_places: list[dict[str, Any]] = []
            self.next_id = 1
            self.next_ray_id = 1
            self.now = 0.0
            self.places = {}                     # a reset map has no taught places either (load_dict restores them from the file)
            self.next_place_id = 1
            self.last_merge = -1e9
            self.ambient_c: float | None = None
            self.stats = {'frames': 0, 'detections': 0, 'created': 0, 'merged': 0, 'removed_ttl': 0,
                          'removed_miss': 0, 'removed_dynamic': 0, 'rays': 0, 'triangulated': 0, 'ignored': 0}

    # ------------------------------------------------------------------------------------ labels, status
    def label(self, o: MapObject) -> str:
        if not o.votes:
            return 'object'
        return max(sorted(o.votes), key=lambda k: o.votes[k])

    def canon(self, o: MapObject) -> str:
        return self.vocab.canon(self.label(o))

    def required_obs(self, o: MapObject) -> int:
        cat = self.vocab.cat_cfg(self.label(o))
        base = cat.min_obs if cat.min_obs is not None else self.cfg.confirm.min_obs
        return base + o.extra_obs

    def required_score(self, o: MapObject) -> float:
        cat = self.vocab.cat_cfg(self.label(o))
        return cat.min_mean_score if cat.min_mean_score is not None else self.cfg.confirm.min_mean_score

    def status(self, o: MapObject) -> str:
        if o.rejected:
            return 'rejected'
        if o.pinned:
            return 'confirmed'
        if o.bearing_only:
            return 'bearing' if o.ray_support >= self.cfg.bearing.min_rays else 'tentative'
        if o.n_obs >= self.required_obs(o) and o.mean_score >= self.required_score(o):
            return 'confirmed'
        return 'tentative'

    def confidence(self, o: MapObject) -> float:
        total = sum(o.votes.values()) or 1.0
        purity = max(o.votes.values(), default=0.0) / total
        c = o.mean_score * (1.0 - math.exp(-o.n_obs / 3.0)) * purity
        if o.bearing_only:
            c *= 0.6
        return float(min(max(c, 0.0), 1.0))

    # ------------------------------------------------------------------------------------ ingestion
    def process_frame(self, frame: Frame) -> FrameResult:
        """Integrate all detections of one camera frame."""
        with self.lock:
            res = FrameResult()
            t = frame.stamp
            self.now = max(self.now, t)
            self.stats['frames'] += 1
            depth, bearing = [], []
            for d in sorted(frame.detections, key=lambda d: (-d.score, d.label, tuple(np.round(d.centroid, 3)))):
                self.stats['detections'] += 1
                kind = self._screen(d)
                if kind == 'drop':
                    res.ignored += 1
                    self.stats['ignored'] += 1
                elif kind == 'ray':
                    bearing.append(d)
                else:
                    depth.append(d)
            matched = self._associate(depth, t, frame.camera_position, res)
            if self.cfg.bearing.enabled:
                for d in bearing:
                    self._add_ray(d, t, frame.camera_position, res, matched)
            if frame.view is not None:
                self._apply_misses(frame.view, t, matched, res)
            self.housekeeping(t, res)
            return res

    def _screen(self, d: Detection) -> str:
        """'depth' | 'ray' | 'drop' for a detection."""
        cat = self.vocab.cat_cfg(d.label)
        min_pts = cat.min_points if cat.min_points is not None else self.cfg.association.min_points
        max_ext = cat.max_extent_m if cat.max_extent_m is not None else self.cfg.association.max_extent_m
        has_pos = bool(np.all(np.isfinite(d.centroid)))
        if d.bearing_only or not has_pos or d.point_count < min_pts:
            if not np.all(np.isfinite(d.bearing)) or np.linalg.norm(d.bearing) < 0.5 or cat.dynamic:
                return 'drop'          # moving objects cannot be triangulated from rays taken at different times
            return 'ray'
        ext = np.nan_to_num(d.extent, nan=0.0)
        if np.max(ext) > max_ext:
            return 'drop'
        if cat.min_height_m is not None and ext[2] < cat.min_height_m:
            return 'drop'
        return 'depth'

    # ------------------------------------------------------------------------------------ association
    def _gate(self, o: MapObject, d_ext: np.ndarray, dt: float) -> float:
        a = self.cfg.association
        cat = self.vocab.cat_cfg(self.label(o))
        if o.dynamic:
            base = cat.gate_m if cat.gate_m is not None else 1.0
            return min(self.cfg.dynamic.gate_max_m, base + self.cfg.dynamic.speed_max_mps * min(max(dt, 0.0), 2.0))
        if cat.gate_m is not None:
            g = cat.gate_m
        else:
            horiz = max(float(np.max(o.size[:2])), float(np.max(d_ext[:2])))
            g = min(max(a.gate_size_factor * horiz, a.gate_min_m), a.gate_max_m)
        if o.bearing_only:
            g = max(g, a.bearing_gate_m)
        return g

    def _predicted(self, o: MapObject, t: float) -> np.ndarray:
        if o.dynamic and o.moving:
            return o.position + o.velocity * min(max(t - o.last_seen, 0.0), 2.0)
        return o.position

    def _suppress_same_frame_duplicates(self, depth: list[Detection]) -> tuple[list[Detection], dict[int, list[Detection]]]:
        """The detector may tag one physical object twice in a frame (bottle + cup). The weaker detection of a
        related-label pair that overlaps the stronger one becomes an extra label vote instead of a second object."""
        a = self.cfg.association
        kept: list[Detection] = []
        extra: dict[int, list[Detection]] = {}
        for d in depth:                                       # sorted by score, strongest first
            ext = np.nan_to_num(d.extent, nan=0.0)
            for i, k in enumerate(kept):
                level = self.vocab.compat(d.label, k.label)
                if level == 0 or self.vocab.is_dynamic(d.label):
                    continue
                iou = box_iou(d.centroid, ext, k.centroid, np.nan_to_num(k.extent, nan=0.0))
                close = float(np.linalg.norm(d.centroid - k.centroid)) <= a.cross_label_dist_factor * a.gate_min_m
                if (level == 1 and (iou >= a.cross_label_iou or close)) or (level == 2 and iou >= 0.7):
                    extra.setdefault(i, []).append(d)
                    break
            else:
                kept.append(d)
        return kept, extra

    def _associate(self, depth: list[Detection], t: float, cam: np.ndarray, res: FrameResult) -> set[int]:
        a = self.cfg.association
        depth, extra = self._suppress_same_frame_duplicates(depth)
        owner: dict[int, int] = {}
        pairs = []
        for i, d in enumerate(depth):
            ext = np.nan_to_num(d.extent, nan=0.0)
            for oid in sorted(self.objects):
                o = self.objects[oid]
                if o.rejected:
                    continue
                level = self.vocab.compat(d.label, self.label(o))
                dt = t - o.last_seen
                pred = self._predicted(o, t)
                dist = float(np.linalg.norm(d.centroid - pred))
                iou = box_iou(d.centroid, ext, o.position, o.size)
                gate = self._gate(o, ext, dt)
                cost = None
                if level == 2:
                    if dist <= gate or iou >= a.iou_assoc:
                        cost = min(dist / gate, 1.0 - iou)
                elif level == 1:
                    if (iou >= a.cross_label_iou or dist <= a.cross_label_dist_factor * gate) and o.n_3d >= 2:
                        cost = 0.5 + (1.0 - iou)
                elif o.n_3d >= 3 and iou >= a.cross_label_iou + 0.2 and not o.dynamic:
                    cost = 1.0 + (1.0 - iou)
                if cost is None:
                    continue
                cost *= 1.0 - 0.2 * math.exp(-max(dt, 0.0) / a.recency_tau_s)
                pairs.append((cost, i, oid))
        pairs.sort()
        used_d, used_o, matched = set(), set(), set()
        for cost, i, oid in pairs:
            if i in used_d or oid in used_o:
                continue
            used_d.add(i)
            used_o.add(oid)
            self._update(self.objects[oid], depth[i], t, cam)
            owner[i] = oid
            matched.add(oid)
            res.updated.append(oid)
        for i, d in enumerate(depth):
            if i not in used_d:
                o = self._create(d, t, cam)
                owner[i] = o.id
                matched.add(o.id)
                res.created.append(o.id)
        for i, dups in extra.items():
            o = self.objects[owner[i]]
            for d in dups:
                o.votes[d.label] = o.votes.get(d.label, 0.0) + 0.5 * max(d.score, 1e-3)
        return matched

    def _create(self, d: Detection, t: float, cam: np.ndarray) -> MapObject:
        o = MapObject(id=self.next_id, first_seen=t, last_seen=t)
        self.next_id += 1
        self.stats['created'] += 1
        o.dynamic = self.vocab.is_dynamic(d.label)
        o.position = d.centroid.astype(float).copy()
        o.size = np.nan_to_num(d.extent, nan=0.0).astype(float).copy()
        o.extra_obs = self._rejected_penalty(d.centroid, d.label, t)
        self.objects[o.id] = o
        self._update(o, d, t, cam)
        return o

    def _rejected_penalty(self, pos: np.ndarray, label: str, t: float) -> int:
        f = self.cfg.forget
        canon = self.vocab.canon(label)
        for r in self.rejected_places:
            if t - r['t'] <= f.rejected_ttl_s and r['canon'] == canon and \
                    np.linalg.norm(np.array(r['pos']) - pos) < f.rejected_radius_m:
                return f.rejected_extra_obs
        return 0

    def _update(self, o: MapObject, d: Detection, t: float, cam: np.ndarray) -> None:
        a = self.cfg.association
        if o.bearing_only:                                   # a depth sighting replaces the triangulated guess
            o.bearing_only = False
            o.hist.clear()
            o.n_3d = 0
        o.votes[d.label] = o.votes.get(d.label, 0.0) + max(d.score, 1e-3)
        o.last_seen = t
        o.misses = max(0.0, o.misses - self.cfg.forget.hit_credit)
        o.point_count = int(d.point_count)
        counted = (t - o.last_counted) >= a.obs_min_dt_s or o.n_obs == 0
        if counted:
            o.last_counted = t
            o.n_obs += 1
            o.n_3d += 1
            o.score_sum += d.score
            w = math.sqrt(min(max(d.point_count, 1), 500)) * max(d.score, 0.05)
            o.hist.append((t, d.centroid.astype(float).copy(), np.nan_to_num(d.extent, nan=0.0).astype(float), w))
            if len(o.hist) > a.history:
                o.hist.pop(0)
            if o.dynamic:
                o.track.append((t, d.centroid.astype(float).copy()))
                o.track = [(ts, p) for ts, p in o.track if t - ts <= 3.0 * self.cfg.dynamic.window_s]
            self._refresh_geometry(o)
            self._update_thermal(o, d)
        self._maybe_add_view(o, d, cam, t)

    def _refresh_geometry(self, o: MapObject) -> None:
        a = self.cfg.association
        if not o.hist:
            return
        pos = np.array([h[1] for h in o.hist])
        ext = np.array([h[2] for h in o.hist])
        w = np.array([h[3] for h in o.hist])
        if o.dynamic:
            o.position = pos[-2:].mean(axis=0) if len(pos) > 1 else pos[-1]
            self._update_motion(o)
        else:
            if len(pos) >= 4:
                med = np.median(pos, axis=0)
                dist = np.linalg.norm(pos - med, axis=1)
                mad = np.median(dist) + 1e-3
                keep = dist <= med_thr(mad, a.position_trim)
                if keep.sum() >= 2:
                    pos, w, ext = pos[keep], w[keep], ext[keep]
            o.position = (pos * w[:, None]).sum(axis=0) / w.sum()
        o.size = np.percentile(ext, a.size_percentile, axis=0)

    def _update_motion(self, o: MapObject) -> None:
        dyn = self.cfg.dynamic
        if len(o.track) < 3:
            return
        t_end = o.track[-1][0]
        window = [(ts, p) for ts, p in o.track if t_end - ts <= dyn.window_s]
        if len(window) < 3 or window[-1][0] - window[0][0] < 0.5:
            return
        ts = np.array([w[0] for w in window])
        ps = np.array([w[1] for w in window])
        tc = ts - ts.mean()
        vel = (tc[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / max((tc ** 2).sum(), 1e-9)   # LS slope
        speed = float(np.linalg.norm(vel[:2]))
        o.velocity = np.array([vel[0], vel[1], 0.0])
        o.moving = speed > dyn.moving_speed_mps

    def _update_thermal(self, o: MapObject, d: Detection) -> None:
        if not self.cfg.thermal.enabled:
            return
        if math.isfinite(d.thermal_mean_c):
            o.thermal_n += 1
            o.thermal_mean_sum += d.thermal_mean_c
        if math.isfinite(d.thermal_max_c):
            o.thermal_max = d.thermal_max_c if not math.isfinite(o.thermal_max) else max(o.thermal_max, d.thermal_max_c)

    # ------------------------------------------------------------------------------------ views
    def _maybe_add_view(self, o: MapObject, d: Detection, cam: np.ndarray, t: float) -> None:
        vc = self.cfg.views
        if not d.crop:
            return
        x1, y1, x2, y2 = d.box_xyxy
        area = max(0, x2 - x1) * max(0, y2 - y1)
        pre = d.score * math.sqrt(max(area, 1.0)) / 100.0
        if len(o.views) >= vc.per_object and pre < 0.7 * min(v.quality for v in o.views):
            return                                         # cannot beat the stored views: skip the decode
        rgb = attr.decode_jpeg(d.crop)
        if rgb is None or min(rgb.shape[:2]) < vc.min_crop_px:
            return
        if area <= 0:
            area = rgb.shape[0] * rgb.shape[1]
        q = attr.view_quality(d.score, area, attr.sharpness(rgb), vc.sharpness_weight)
        view = ObjectView(jpeg=d.crop, quality=q, camera_position=np.asarray(cam, float).copy(), stamp=t,
                          phash=attr.ahash(rgb), size_px=(rgb.shape[1], rgb.shape[0]))
        self._insert_view(o, view)

    def _insert_view(self, o: MapObject, view: ObjectView) -> None:
        vc = self.cfg.views
        target = o.position
        new_dir = view.direction_from(target)
        similar = [v for v in o.views if
                   math.degrees(math.acos(float(np.clip(v.direction_from(target) @ new_dir, -1, 1)))) < vc.min_separation_deg]
        if similar:
            best_similar = max(similar, key=lambda v: v.quality)
            if view.quality <= best_similar.quality:
                return
            o.views = [v for v in o.views if v not in similar]
        o.views.append(view)
        o.views.sort(key=lambda v: -v.quality)
        del o.views[vc.per_object:]
        self._refresh_colour(o)

    def best_view(self, o: MapObject) -> ObjectView | None:
        return o.views[0] if o.views else None

    def _refresh_colour(self, o: MapObject) -> None:
        bv = self.best_view(o)
        if bv is None or bv.phash == o.colour_key:
            return
        o.colour = attr.dominant_colour(bv.jpeg, self.cfg.colors or None)
        o.colour_key = bv.phash

    # ------------------------------------------------------------------------------------ rays
    def _ray_terms(self, origin: np.ndarray, direction: np.ndarray, point: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, float]:
        """Weighted normal-equation terms (A, b, c) of one ray; weight = 1 / (range * sigma)^2 at `point`."""
        b = self.cfg.bearing
        r = max(float(np.linalg.norm(point - origin)), b.min_range_m)
        w = 1.0 / (r * b.sigma_rad) ** 2
        p = np.eye(3) - np.outer(direction, direction)
        return w * p, w * (p @ origin), w * float(origin @ p @ origin)

    @staticmethod
    def _solve_terms(a: np.ndarray, b: np.ndarray, c: float, n: int) -> tuple[np.ndarray, float, float] | None:
        """(point, chi2 per degree of freedom, position std in m) of accumulated ray terms; None if degenerate."""
        try:
            x = np.linalg.solve(a, b)
            cov = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            return None
        chi2 = max(float(x @ a @ x - 2.0 * x @ b + c), 0.0)
        return x, chi2 / max(2 * n - 3, 1), float(math.sqrt(max(np.trace(cov), 0.0)))

    def _ray_fit_ok(self, sol: tuple[np.ndarray, float, float] | None, origins: list[np.ndarray],
                    directions: list[np.ndarray]) -> bool:
        b = self.cfg.bearing
        if sol is None or sol[1] > b.chi_max or sol[2] > b.max_pos_std_m:
            return False
        for o, d in zip(origins, directions):
            _, s = point_ray_distance(sol[0], o, d)
            if not (b.min_range_m <= s <= b.max_range_m):
                return False
            if self.los_fn is not None and not self.los_fn(o, sol[0]):
                return False
        return True

    def _add_ray(self, d: Detection, t: float, cam: np.ndarray, res: FrameResult, matched: set[int]) -> None:
        b = self.cfg.bearing
        cam = np.asarray(cam, float)
        direction = d.bearing / np.linalg.norm(d.bearing)
        # 1. an existing 3D object lying on the ray gets a supporting sighting (no position change)
        if b.support_objects:
            best = None
            for oid in sorted(self.objects):
                o = self.objects[oid]
                if o.rejected or o.bearing_only or oid in matched or self.vocab.compat(d.label, self.label(o)) < 2:
                    continue
                dist, s = point_ray_distance(o.position, cam, direction)
                if s > 0 and dist <= max(b.assoc_ray_dist_m, 0.5 * float(np.max(o.size))) and \
                        (best is None or dist < best[0]):
                    best = (dist, o)
            if best is not None:
                o = best[1]
                o.last_seen = t
                o.misses = max(0.0, o.misses - self.cfg.forget.hit_credit)
                if t - o.last_counted >= self.cfg.association.obs_min_dt_s:
                    o.last_counted = t
                    o.n_obs += 1
                    o.score_sum += d.score
                    o.votes[d.label] = o.votes.get(d.label, 0.0) + max(d.score, 1e-3)
                matched.add(o.id)
                res.updated.append(o.id)
                self._maybe_add_view(o, d, cam, t)
                return
        # 2. a triangulated object that the ray is consistent with is refined by it
        best_b = None
        for oid in sorted(self.objects):
            o = self.objects[oid]
            if not o.bearing_only or o.rejected or oid in matched or self.vocab.compat(d.label, self.label(o)) < 2:
                continue
            dist, s = point_ray_distance(o.position, cam, direction)
            if 0 < s <= b.max_range_m and dist <= b.assoc_ray_dist_m + 2.0 * b.sigma_rad * s and \
                    (best_b is None or dist < best_b[0]):
                best_b = (dist, o)
        if best_b is not None:
            o = best_b[1]
            ta, tb, tc = self._ray_terms(cam, direction, o.position)
            sol = self._solve_terms(o.ray_a + ta, o.ray_b + tb, o.ray_c + tc, o.ray_support + 1)
            if sol is not None and sol[1] <= b.chi_max and sol[2] <= b.max_pos_std_m:
                o.ray_a, o.ray_b, o.ray_c = o.ray_a + ta, o.ray_b + tb, o.ray_c + tc
                o.position = sol[0]
                o.ray_support += 1
                o.n_obs += 1
                o.score_sum += d.score
                o.votes[d.label] = o.votes.get(d.label, 0.0) + max(d.score, 1e-3)
                o.last_seen = t
                matched.add(o.id)
                res.updated.append(o.id)
                self._maybe_add_view(o, d, cam, t)
                return
        # 3. otherwise keep the ray and look for a partner from another pose
        ray = Ray(id=self.next_ray_id, origin=cam.copy(), direction=direction, label=d.label,
                  canon=self.vocab.canon(d.label), score=d.score, stamp=t)
        self.next_ray_id += 1
        self.rays.append(ray)
        res.rays_added += 1
        self.stats['rays'] += 1
        if len(self.rays) > b.max_rays:
            del self.rays[:len(self.rays) - b.max_rays]
        oid = self._try_triangulate(ray, d, t, cam)
        if oid is not None:
            res.triangulated.append(oid)
            matched.add(oid)

    def _try_triangulate(self, new: Ray, d: Detection, t: float, cam: np.ndarray) -> int | None:
        b = self.cfg.bearing
        partners = [r for r in self.rays[:-1]
                    if r.used_by is None and self.vocab.compat(r.label, new.label) == 2
                    and np.linalg.norm(r.origin - new.origin) >= b.min_baseline_m]
        if len(partners) + 1 < max(2, b.min_rays):
            return None
        best = None
        for old in partners:
            point = self._pair_point(new, old)
            if point is None:
                continue
            skew = point_ray_distance(point, old.origin, old.direction)[0]
            if best is None or skew < best[0]:
                best = (skew, point, old)
        if best is None:
            return None
        _, point, first = best
        members = [new, first]
        for other in partners:
            if other is first:
                continue
            dist, s = point_ray_distance(point, other.origin, other.direction)
            if dist <= 2.0 * b.max_skew_m and b.min_range_m <= s <= b.max_range_m:
                members.append(other)
        for group in (members, [new, first]):
            terms = [self._ray_terms(r.origin, r.direction, point) for r in group]
            a = sum(x[0] for x in terms)
            bb = sum(x[1] for x in terms)
            c = sum(x[2] for x in terms)
            sol = self._solve_terms(a, bb, c, len(group))
            if self._ray_fit_ok(sol, [r.origin for r in group], [r.direction for r in group]):
                return self._place_triangulated(sol[0], group, (a, bb, c), d, t, cam)
        return None

    def _pair_point(self, a: Ray, b: Ray) -> np.ndarray | None:
        cfg = self.cfg.bearing
        if math.degrees(math.acos(min(1.0, abs(float(a.direction @ b.direction))))) < cfg.min_angle_deg:
            return None
        w = a.origin - b.origin
        m = np.array([[a.direction @ a.direction, -a.direction @ b.direction],
                      [a.direction @ b.direction, -b.direction @ b.direction]])
        rhs = np.array([-(w @ a.direction), -(w @ b.direction)])
        try:
            s, tt = np.linalg.solve(m, rhs)
        except np.linalg.LinAlgError:
            return None
        if not (cfg.min_range_m <= s <= cfg.max_range_m and cfg.min_range_m <= tt <= cfg.max_range_m):
            return None
        pa, pb = a.origin + s * a.direction, b.origin + tt * b.direction
        if np.linalg.norm(pa - pb) > cfg.max_skew_m:
            return None
        return (pa + pb) / 2.0

    def _place_triangulated(self, pos: np.ndarray, rays: list[Ray], terms: tuple[np.ndarray, np.ndarray, float],
                            d: Detection, t: float, cam: np.ndarray) -> int | None:
        b = self.cfg.bearing
        for oid in sorted(self.objects):
            o = self.objects[oid]
            if o.rejected or self.vocab.compat(d.label, self.label(o)) < 2:
                continue
            if o.bearing_only:
                delta = pos - o.position
                inside = float(delta @ o.ray_a @ delta) <= b.explain_mahalanobis ** 2 or \
                    np.linalg.norm(delta) <= self.cfg.association.bearing_gate_m
            else:
                inside = np.linalg.norm(o.position - pos) <= self._gate(o, np.zeros(3), 0.0) + 0.5 * float(np.max(o.size))
            if not inside:
                continue
            for r in rays:
                r.used_by = o.id
            if o.bearing_only:
                a2, b2, c2 = o.ray_a + terms[0], o.ray_b + terms[1], o.ray_c + terms[2]
                sol = self._solve_terms(a2, b2, c2, o.ray_support + len(rays))
                if sol is not None and sol[1] <= b.chi_max and sol[2] <= b.max_pos_std_m:
                    o.ray_a, o.ray_b, o.ray_c, o.position = a2, b2, c2, sol[0]
                    o.ray_support += len(rays)
                    o.n_obs += len(rays)
            o.last_seen = t
            return o.id
        o = MapObject(id=self.next_id, first_seen=min(r.stamp for r in rays), last_seen=t)
        self.next_id += 1
        self.stats['created'] += 1
        self.stats['triangulated'] += 1
        o.bearing_only = True
        o.dynamic = self.vocab.is_dynamic(d.label)
        o.position = pos
        o.size = np.full(3, b.default_size_m)
        o.ray_a, o.ray_b, o.ray_c = terms
        for r in rays:
            o.votes[r.label] = o.votes.get(r.label, 0.0) + max(r.score, 1e-3)
            o.score_sum += r.score
            r.used_by = o.id
        o.n_obs = o.ray_support = len(rays)
        o.last_counted = t
        self.objects[o.id] = o
        self._maybe_add_view(o, d, cam, t)
        return o.id

    # ------------------------------------------------------------------------------------ negative evidence
    def _in_view(self, view: ViewGeometry, p: np.ndarray) -> bool:
        f = self.cfg.forget
        v = p - view.position
        r = float(np.linalg.norm(v))
        if not (f.view_min_range_m <= r <= min(f.view_max_range_m, view.max_range_m)):
            return False
        ang = math.acos(float(np.clip((v / r) @ view.forward, -1.0, 1.0)))
        if ang > f.view_margin * 0.5 * min(view.hfov_rad, view.vfov_rad):
            return False
        return self.los_fn is None or self.los_fn(view.position, p)

    def _apply_misses(self, view: ViewGeometry, t: float, matched: set[int], res: FrameResult) -> None:
        f = self.cfg.forget
        for oid in sorted(self.objects):
            o = self.objects[oid]
            if oid in matched or o.dynamic or o.bearing_only or o.pinned or o.rejected:
                continue
            if self.status(o) != 'confirmed' or t - o.last_miss < self.cfg.association.obs_min_dt_s:
                continue
            if self._in_view(view, o.position):
                o.last_miss = t
                o.misses += 1.0
                if o.misses >= f.miss_limit:
                    self._remove(oid, 'removed_miss', reject=False)
                    res.removed.append(oid)

    # ------------------------------------------------------------------------------------ housekeeping
    def housekeeping(self, t: float, res: FrameResult | None = None) -> None:
        with self.lock:
            self.now = max(self.now, t)
            f, dyn, b = self.cfg.forget, self.cfg.dynamic, self.cfg.bearing
            for oid in sorted(self.objects):
                o = self.objects[oid]
                age = t - o.last_seen
                gone = None
                if o.dynamic:
                    ttl = dyn.ttl_moving_s if (o.moving or o.n_obs < 3) else dyn.ttl_static_s
                    if age > ttl:
                        gone = 'removed_dynamic'
                elif o.bearing_only:
                    if age > b.object_ttl_s:
                        gone = 'removed_ttl'
                elif self.status(o) == 'tentative' and not o.pinned and age > f.tentative_ttl_s:
                    gone = 'removed_ttl'
                if gone:
                    self._remove(oid, gone, reject=(gone == 'removed_ttl' and not o.bearing_only))
                    if res is not None:
                        res.removed.append(oid)
            self.rays = [r for r in self.rays if t - r.stamp <= b.ttl_s]
            self.rejected_places = [r for r in self.rejected_places if t - r['t'] <= f.rejected_ttl_s]
            if t - self.last_merge >= self.cfg.merge.period_s:
                self.last_merge = t
                self.merge_duplicates()
                self._absorb_bearing_objects()
            self._update_ambient()
            if len(self.objects) > f.max_objects:
                worst = sorted(self.objects.values(), key=lambda o: (self.confidence(o), o.last_seen))
                for o in worst[:len(self.objects) - f.max_objects]:
                    self._remove(o.id, 'removed_ttl', reject=False)

    def _absorb_bearing_objects(self) -> None:
        """A triangulated object is dropped when a confirmed 3D object of its class lies inside its uncertainty
        ellipsoid: the rays that created it were just looking at that object (range ambiguity along the ray)."""
        gate = self.cfg.bearing.explain_mahalanobis ** 2
        for oid in sorted(self.objects):
            b = self.objects.get(oid)
            if b is None or not b.bearing_only:
                continue
            for did in sorted(self.objects):
                d = self.objects[did]
                if d.bearing_only or d.rejected or self.status(d) != 'confirmed' or                         self.vocab.compat(self.label(b), self.label(d)) < 2:
                    continue
                delta = d.position - b.position
                if float(delta @ b.ray_a @ delta) <= gate:
                    d.ray_support += b.ray_support
                    d.last_seen = max(d.last_seen, b.last_seen)
                    self._remove(oid, 'absorbed', reject=False)
                    break

    def _remove(self, oid: int, reason: str, reject: bool) -> None:
        o = self.objects.pop(oid, None)
        if o is None:
            return
        self.stats[reason] = self.stats.get(reason, 0) + 1
        if reject:
            self.rejected_places.append({'pos': o.position.tolist(), 'canon': self.canon(o), 't': self.now,
                                         'reason': reason})
        for r in self.rays:
            if r.used_by == oid:
                r.used_by = None

    def _update_ambient(self) -> None:
        if not self.cfg.thermal.enabled:
            self.ambient_c = None
            return
        vals = [o.thermal_mean for o in self.objects.values() if o.thermal_n >= 2 and not o.dynamic]
        self.ambient_c = float(np.median(vals)) if len(vals) >= 3 else None

    # ------------------------------------------------------------------------------------ merging
    def merge_distance(self, a: MapObject, b: MapObject) -> float:
        m = self.cfg.merge
        horiz = max(float(np.max(a.size[:2])), float(np.max(b.size[:2])))
        return min(max(m.dist_factor * horiz, m.dist_min_m), m.dist_max_m)

    def _mergeable(self, a: MapObject, b: MapObject) -> bool:
        if a.rejected or b.rejected or b.id in a.distinct_from or a.id in b.distinct_from:
            return False
        level = self.vocab.compat(self.label(a), self.label(b))
        dist = float(np.linalg.norm(a.position - b.position))
        if level < 2:
            # a triangulated object only knows its label from noisy rays: same-category neighbours are one object
            return level == 1 and (a.bearing_only or b.bearing_only) and dist < self.merge_distance(a, b)
        if a.dynamic or b.dynamic:
            return dist < self.cfg.dynamic.merge_dist_m
        if dist < self.merge_distance(a, b):
            return True
        if a.bearing_only or b.bearing_only:
            return dist < self.cfg.association.bearing_gate_m
        return box_iou(a.position, a.size, b.position, b.size) >= self.cfg.merge.iou

    def merge_duplicates(self) -> list[tuple[int, int]]:
        """Merge compatible instances that ended up on top of each other; returns (kept, dropped) ids."""
        merged: list[tuple[int, int]] = []
        changed = True
        while changed:
            changed = False
            ids = sorted(self.objects)
            for i, ia in enumerate(ids):
                for ib in ids[i + 1:]:
                    a, b = self.objects[ia], self.objects[ib]
                    if self._mergeable(a, b):
                        self.merge_objects(ia, ib)
                        merged.append((ia, ib))
                        changed = True
                        break
                if changed:
                    break
        return merged

    def merge_objects(self, keep_id: int, drop_id: int) -> None:
        a, b = self.objects[keep_id], self.objects.pop(drop_id)
        self.stats['merged'] += 1
        for k, v in b.votes.items():
            a.votes[k] = a.votes.get(k, 0.0) + v
        a.first_seen = min(a.first_seen, b.first_seen)
        a.last_seen = max(a.last_seen, b.last_seen)
        a.n_obs += b.n_obs
        a.n_3d += b.n_3d
        a.score_sum += b.score_sum
        a.misses = min(a.misses, b.misses)
        a.ray_support += b.ray_support
        if a.bearing_only and b.bearing_only:
            a.ray_a, a.ray_b, a.ray_c = a.ray_a + b.ray_a, a.ray_b + b.ray_b, a.ray_c + b.ray_c
        a.point_count = max(a.point_count, b.point_count)
        a.pinned = a.pinned or b.pinned
        a.extra_obs = min(a.extra_obs, b.extra_obs)
        a.thermal_n += b.thermal_n
        a.thermal_mean_sum += b.thermal_mean_sum
        if math.isfinite(b.thermal_max):
            a.thermal_max = b.thermal_max if not math.isfinite(a.thermal_max) else max(a.thermal_max, b.thermal_max)
        if a.bearing_only and not b.bearing_only:
            a.bearing_only = False
            a.hist = []
            a.n_3d = b.n_3d
        if not b.bearing_only:
            a.hist = sorted(a.hist + b.hist, key=lambda h: h[0])[-self.cfg.association.history:]
        a.track = sorted(a.track + b.track, key=lambda h: h[0])
        if a.hist:
            self._refresh_geometry(a)
        elif not b.bearing_only:
            a.position = b.position
        for v in b.views:
            self._insert_view(a, v)
        if b.captions_done > a.captions_done or (b.caption and not a.caption):
            a.caption, a.caption_quality, a.captions_done = b.caption, b.caption_quality, b.captions_done
        for k, v in b.vlm_attrs.items():
            a.vlm_attrs.setdefault(k, v)
        a.label_ok_votes += b.label_ok_votes
        a.label_bad_votes += b.label_bad_votes
        a.distinct_from |= b.distinct_from
        for r in self.rays:
            if r.used_by == drop_id:
                r.used_by = keep_id
        for o in self.objects.values():
            if drop_id in o.distinct_from:
                o.distinct_from.discard(drop_id)
                o.distinct_from.add(keep_id)

    def doubtful_pairs(self, limit: int = 5) -> list[tuple[int, int, float]]:
        """Compatible, not merged pairs that are close enough to be the same object (candidates for the VLM)."""
        out = []
        ids = sorted(self.objects)
        for i, ia in enumerate(ids):
            a = self.objects[ia]
            if a.dynamic or a.rejected or not a.views:
                continue
            for ib in ids[i + 1:]:
                b = self.objects[ib]
                if b.dynamic or b.rejected or not b.views or ib in a.distinct_from:
                    continue
                if self.vocab.compat(self.label(a), self.label(b)) < 2:
                    continue
                dist = float(np.linalg.norm(a.position - b.position))
                md = self.merge_distance(a, b)
                if md <= dist < md * self.cfg.merge.doubtful_factor:
                    out.append((ia, ib, dist))
        out.sort(key=lambda r: r[2])
        return out[:limit]

    # ------------------------------------------------------------------------------------ feedback
    def reject_object(self, oid: int, reason: str = 'rejected') -> bool:
        """Remove an instance for good (e.g. verification said it is not what the label claims) and remember the place."""
        with self.lock:
            if oid not in self.objects:
                return False
            self._remove(oid, 'removed_ttl', reject=True)
            self.rejected_places[-1]['reason'] = reason
            return True

    def confirm_object(self, oid: int) -> bool:
        with self.lock:
            o = self.objects.get(oid)
            if o is None:
                return False
            o.pinned = True
            return True

    def relabel_object(self, oid: int, label: str) -> bool:
        with self.lock:
            o = self.objects.get(oid)
            if o is None:
                return False
            o.votes = {label: max(sum(o.votes.values()), 1.0) * 2.0}
            return True

    # ------------------------------------------------------------------------------------ attributes
    def attribute_dict(self, o: MapObject) -> dict[str, str]:
        out: dict[str, str] = {}
        out['size'] = attr.size_class(o.size, self.cfg.size)
        colour = o.vlm_attrs.get('colour') or o.colour
        if colour:
            out['colour'] = colour
        th = attr.thermal_attributes(o.thermal_mean, o.thermal_max, self.ambient_c, self.cfg.thermal)
        if th:
            out['thermal'] = ' '.join(th)
        if o.dynamic:
            out['motion'] = 'moving' if o.moving else 'stationary'
        for k, v in o.vlm_attrs.items():
            if k != 'colour' and v:
                out[k] = v
        return out

    def attribute_list(self, o: MapObject) -> list[str]:
        """Flat words for SemanticObject.attributes: 'warm', 'hot', 'person-like', 'red', 'open', 'large', ..."""
        words: list[str] = []
        for k, v in self.attribute_dict(o).items():
            words.extend(v.split())
        th = self.cfg.thermal
        if o.dynamic and self.canon(o) == 'person':
            words.append('person-like')
        elif th.enabled and math.isfinite(o.thermal_mean) and o.thermal_mean >= th.person_min_mean_c and \
                0.3 <= o.size[0] <= 1.2 and 0.3 <= o.size[1] <= 1.2 and 1.0 <= o.size[2] <= 2.1:
            words.append('person-like')
        seen, uniq = set(), []
        for w in words:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        return uniq

    # ------------------------------------------------------------------------------------ rooms
    def set_rooms(self, room_map: Any, regions: dict[int, Region]) -> None:
        with self.lock:
            self.room_map = room_map
            self.regions = regions

    def room_id_at(self, pos: np.ndarray) -> int | None:
        if self.room_map is None:
            return None
        rid = self.room_map.label_at(float(pos[0]), float(pos[1]))
        return rid if rid and rid in self.regions else None

    def room_name_of(self, o: MapObject) -> str:
        rid = self.room_id_at(o.position)
        return self.regions[rid].name if rid is not None else ''

    # ------------------------------------------------------------------------------------ description, query
    def describe(self, o: MapObject, pose: tuple[float, float, float] | None = None, detail: bool = False
                 ) -> dict[str, Any]:
        d: dict[str, Any] = {
            'id': o.id, 'label': self.label(o), 'canon': self.canon(o), 'status': self.status(o),
            'confidence': round(self.confidence(o), 3),
            'position': [round(float(x), 3) for x in o.position],
            'size': [round(float(x), 3) for x in o.size],
            'point_count': o.point_count, 'observations': o.n_obs,
            'first_seen': round(o.first_seen, 3), 'last_seen': round(o.last_seen, 3),
            'caption': o.caption, 'attributes': self.attribute_list(o),
            'room': self.room_name_of(o), 'bearing_only': o.bearing_only,
            'moving': o.moving, 'has_view': bool(o.views),
        }
        if pose is not None:
            dx, dy = o.position[0] - pose[0], o.position[1] - pose[1]
            d['range_m'] = round(math.hypot(dx, dy), 2)
            d['bearing_deg'] = round(math.degrees(_wrap(math.atan2(dy, dx) - pose[2])), 1)   # + = left of heading
        if detail:
            d['attribute_map'] = self.attribute_dict(o)
            d['votes'] = {k: round(v, 3) for k, v in sorted(o.votes.items())}
            d['velocity'] = [round(float(x), 3) for x in o.velocity]
            d['thermal_mean_c'] = None if not o.thermal_n else round(o.thermal_mean, 2)
            d['thermal_max_c'] = None if not math.isfinite(o.thermal_max) else round(o.thermal_max, 2)
            d['ray_support'] = o.ray_support
            d['misses'] = round(o.misses, 2)
            d['views'] = len(o.views)
        return d

    def visible_objects(self, include_tentative: bool = False) -> list[MapObject]:
        keep = ('confirmed', 'bearing', 'tentative') if include_tentative else ('confirmed', 'bearing')
        return [self.objects[i] for i in sorted(self.objects) if self.status(self.objects[i]) in keep]

    def query(self, label: str | None = None, near: tuple[float, float] | None = None, radius: float | None = None,
              attributes: list[str] | None = None, room: str | None = None, limit: int = 10,
              include_tentative: bool = False, min_confidence: float = 0.0) -> list[MapObject]:
        with self.lock:
            out = []
            want = {a.lower() for a in (attributes or [])}
            for o in self.visible_objects(include_tentative):
                if label and not self._label_matches(o, label):
                    continue
                if near is not None and radius is not None and \
                        math.hypot(o.position[0] - near[0], o.position[1] - near[1]) > radius:
                    continue
                if want and not want.issubset({w.lower() for w in self.attribute_list(o)} |
                                              {w.lower() for w in o.vlm_attrs.values()}):
                    continue
                if room and room.lower() != self.room_name_of(o).lower():
                    continue
                if self.confidence(o) < min_confidence:
                    continue
                out.append(o)
            if near is not None:
                out.sort(key=lambda o: (math.hypot(o.position[0] - near[0], o.position[1] - near[1]), o.id))
            else:
                out.sort(key=lambda o: (-self.confidence(o), o.id))
            return out[:limit]

    def _label_matches(self, o: MapObject, text: str) -> bool:
        t = self.vocab.canon(text)
        if self.canon(o) == t:
            return True
        lab = self.label(o)
        return t in lab or lab in t if len(t) > 2 else False

    # ------------------------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                'schema': SCHEMA_VERSION, 'now': self.now, 'next_id': self.next_id, 'next_ray_id': self.next_ray_id,
                'objects': [_obj_to_dict(self.objects[i]) for i in sorted(self.objects)],
                'rays': [_ray_to_dict(r) for r in self.rays],
                'rejected_places': self.rejected_places,
                'regions': [_region_to_dict(self.regions[i]) for i in sorted(self.regions)],
                'places': [dict(vars(p)) for p in sorted(self.places.values(), key=lambda q: q.id)],
                'stats': self.stats,
            }

    def load_dict(self, data: dict[str, Any], crops: dict[str, bytes] | None = None) -> None:
        if data.get('schema') != SCHEMA_VERSION:
            raise ValueError(f"unsupported semantic map schema {data.get('schema')!r}")
        crops = crops or {}
        with self.lock:
            self.reset()
            self.now = float(data['now'])
            self.next_id = int(data['next_id'])
            self.next_ray_id = int(data['next_ray_id'])
            for od in data['objects']:
                o = _obj_from_dict(od, crops)
                self.objects[o.id] = o
            self.rays = [_ray_from_dict(r) for r in data['rays']]
            self.rejected_places = list(data['rejected_places'])
            self.stats.update(data.get('stats', {}))
            self.regions = {r['id']: _region_from_dict(r) for r in data.get('regions', [])}
            from .places import from_dicts
            from_dicts(self, data.get('places', []))

    def save(self, path: str | os.PathLike) -> Path:
        """Write <path> (JSON) and the crops as <path stem>_crops/<id>_<k>.jpg. The JSON is deterministic."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        crop_dir = path.with_name(path.stem + '_crops')
        with self.lock:
            data = self.to_dict()
            if any(o.views for o in self.objects.values()):
                crop_dir.mkdir(exist_ok=True)
            for o in self.objects.values():
                for k, v in enumerate(o.views):
                    (crop_dir / f'{o.id}_{k}.jpg').write_bytes(v.jpeg)
        path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding='utf-8')
        return path

    def load(self, path: str | os.PathLike) -> None:
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        crop_dir = path.with_name(path.stem + '_crops')
        crops = {p.name: p.read_bytes() for p in crop_dir.glob('*.jpg')} if crop_dir.is_dir() else {}
        self.load_dict(data, crops)


# ----------------------------------------------------------------------------------------------- helpers
def med_thr(mad: float, k: float) -> float:
    return max(k * mad * 1.4826, 0.05)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _arr(x: np.ndarray, nd: int = 5) -> list[float]:
    return [round(float(v), nd) for v in x]


def _nan(x: float):
    return None if not math.isfinite(x) else x


def _obj_to_dict(o: MapObject) -> dict[str, Any]:
    return {
        'id': o.id, 'first_seen': o.first_seen, 'last_seen': o.last_seen, 'votes': o.votes, 'n_obs': o.n_obs,
        'n_3d': o.n_3d, 'score_sum': o.score_sum,
        'hist': [[h[0], _arr(h[1]), _arr(h[2]), h[3]] for h in o.hist],
        'position': _arr(o.position), 'size': _arr(o.size), 'point_count': o.point_count,
        'last_counted': o.last_counted, 'misses': o.misses, 'bearing_only': o.bearing_only,
        'ray_support': o.ray_support, 'ray_a': o.ray_a.tolist(), 'ray_b': o.ray_b.tolist(), 'ray_c': o.ray_c,
        'dynamic': o.dynamic, 'velocity': _arr(o.velocity), 'moving': o.moving,
        'track': [[t, _arr(p)] for t, p in o.track], 'extra_obs': o.extra_obs, 'rejected': o.rejected,
        'pinned': o.pinned, 'thermal_n': o.thermal_n, 'thermal_mean_sum': o.thermal_mean_sum,
        'thermal_max': _nan(o.thermal_max),
        'views': [{'quality': v.quality, 'camera_position': _arr(v.camera_position), 'stamp': v.stamp,
                   'phash': v.phash, 'size_px': list(v.size_px)} for v in o.views],
        'colour': o.colour, 'colour_key': o.colour_key, 'caption': o.caption, 'caption_quality': o.caption_quality,
        'captions_done': o.captions_done, 'vlm_attrs': o.vlm_attrs, 'label_ok_votes': o.label_ok_votes,
        'label_bad_votes': o.label_bad_votes, 'distinct_from': sorted(o.distinct_from),
    }


def _obj_from_dict(d: dict[str, Any], crops: dict[str, bytes]) -> MapObject:
    o = MapObject(id=d['id'], first_seen=d['first_seen'], last_seen=d['last_seen'])
    o.votes = dict(d['votes'])
    o.n_obs, o.n_3d, o.score_sum = d['n_obs'], d['n_3d'], d['score_sum']
    o.hist = [(h[0], np.array(h[1]), np.array(h[2]), h[3]) for h in d['hist']]
    o.position, o.size = np.array(d['position']), np.array(d['size'])
    o.point_count, o.last_counted, o.misses = d['point_count'], d['last_counted'], d['misses']
    o.bearing_only, o.ray_support, o.dynamic = d['bearing_only'], d['ray_support'], d['dynamic']
    o.ray_a, o.ray_b, o.ray_c = np.array(d['ray_a']), np.array(d['ray_b']), d['ray_c']
    o.velocity, o.moving = np.array(d['velocity']), d['moving']
    o.track = [(t, np.array(p)) for t, p in d['track']]
    o.extra_obs, o.rejected, o.pinned = d['extra_obs'], d['rejected'], d['pinned']
    o.thermal_n, o.thermal_mean_sum = d['thermal_n'], d['thermal_mean_sum']
    o.thermal_max = float('nan') if d['thermal_max'] is None else d['thermal_max']
    for k, vd in enumerate(d['views']):
        o.views.append(ObjectView(jpeg=crops.get(f"{o.id}_{k}.jpg", b''), quality=vd['quality'],
                                  camera_position=np.array(vd['camera_position']), stamp=vd['stamp'],
                                  phash=vd['phash'], size_px=tuple(vd['size_px'])))
    o.colour, o.colour_key = d['colour'], d['colour_key']
    o.caption, o.caption_quality, o.captions_done = d['caption'], d['caption_quality'], d['captions_done']
    o.vlm_attrs = dict(d['vlm_attrs'])
    o.label_ok_votes, o.label_bad_votes = d['label_ok_votes'], d['label_bad_votes']
    o.distinct_from = set(d['distinct_from'])
    return o


def _ray_to_dict(r: Ray) -> dict[str, Any]:
    return {'id': r.id, 'origin': _arr(r.origin), 'direction': _arr(r.direction), 'label': r.label,
            'canon': r.canon, 'score': r.score, 'stamp': r.stamp, 'used_by': r.used_by}


def _ray_from_dict(d: dict[str, Any]) -> Ray:
    return Ray(id=d['id'], origin=np.array(d['origin']), direction=np.array(d['direction']), label=d['label'],
               canon=d['canon'], score=d['score'], stamp=d['stamp'], used_by=d['used_by'])


def _region_to_dict(r: Region) -> dict[str, Any]:
    return {'id': r.id, 'name': r.name, 'name_source': r.name_source, 'centroid': _arr(r.centroid),
            'bbox': list(r.bbox), 'area_m2': r.area_m2, 'width_m': r.width_m, 'length_m': r.length_m,
            'vlm_name_key': r.vlm_name_key}


def _region_from_dict(d: dict[str, Any]) -> Region:
    return Region(id=d['id'], name=d['name'], name_source=d['name_source'], centroid=np.array(d['centroid']),
                  bbox=tuple(d['bbox']), area_m2=d['area_m2'], width_m=d['width_m'], length_m=d['length_m'],
                  vlm_name_key=d['vlm_name_key'])
