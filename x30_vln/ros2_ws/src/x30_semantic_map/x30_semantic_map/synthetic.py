"""Synthetic indoor scenario for tests, measurements and rendering demos (no ROS imports).

A 12 x 8 m floor plan (kitchen, living room, corridor, three doors), ground-truth objects, a robot patrol with a
swaying camera, and a noisy detector model: missed detections, label confusion, partial-view size and centroid
bias that grows with range, bearing-only detections for small/far/sparse objects, transient and flickering false
positives, a walking person and a chair that is moved half way through.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .datatypes import Detection, Frame, ViewGeometry
from .gridtools import GridView

RES = 0.05
WIDTH_M, HEIGHT_M = 12.0, 8.0


@dataclass
class GtObject:
    name: str
    label: str
    pos: np.ndarray
    size: np.ndarray
    rgb: tuple[int, int, int]
    dynamic: bool = False


def floorplan() -> GridView:
    """Kitchen (x<6, y>2), living room (x>6, y>2) and a corridor (y<2); known free space, walls, doors."""
    w, h = int(WIDTH_M / RES), int(HEIGHT_M / RES)
    data = np.zeros((h, w), np.int8)

    def wall(x0, y0, x1, y1):
        data[int(y0 / RES):int(y1 / RES), int(x0 / RES):int(x1 / RES)] = 100

    wall(0, 0, WIDTH_M, 0.15)
    wall(0, HEIGHT_M - 0.15, WIDTH_M, HEIGHT_M)
    wall(0, 0, 0.15, HEIGHT_M)
    wall(WIDTH_M - 0.15, 0, WIDTH_M, HEIGHT_M)
    wall(5.9, 2.0, 6.1, HEIGHT_M)                 # kitchen | living room
    wall(0, 1.9, WIDTH_M, 2.1)                    # corridor | rooms
    for x0, x1 in ((2.45, 3.35), (8.65, 9.55)):    # doors corridor <-> rooms (0.9 m wide)
        data[int(1.9 / RES):int(2.1 / RES), int(x0 / RES):int(x1 / RES)] = 0
    data[int(4.45 / RES):int(5.35 / RES), int(5.9 / RES):int(6.1 / RES)] = 0     # door kitchen <-> living room
    return GridView(data, RES, 0.0, 0.0)


def objects(variant_moved: bool = False) -> list[GtObject]:
    def o(name, label, x, y, sx, sy, sz, rgb, dyn=False, z0=0.0):
        return GtObject(name, label, np.array([x, y, z0 + sz / 2.0]), np.array([sx, sy, sz]), rgb, dyn)
    objs = [
        o('fridge', 'refrigerator', 0.6, 7.3, 0.7, 0.7, 1.8, (230, 230, 230)),
        o('kitchen_table', 'table', 3.0, 5.0, 1.4, 0.9, 0.75, (150, 100, 60)),
        o('chair_k1', 'chair', 2.1, 5.0, 0.45, 0.45, 0.9, (200, 40, 40)),
        o('chair_k2', 'chair', 3.9, 5.0, 0.45, 0.45, 0.9, (40, 80, 200)),
        o('chair_k3', 'chair', 3.0, 3.9, 0.45, 0.45, 0.9, (40, 150, 60)),
        o('chair_k4', 'chair', 3.0, 6.1, 0.45, 0.45, 0.9, (30, 30, 30)),
        o('sink', 'sink', 5.5, 7.4, 0.8, 0.5, 0.9, (190, 190, 200)),
        o('bottle', 'bottle', 3.3, 5.15, 0.08, 0.08, 0.26, (40, 150, 60), z0=0.75),   # on the kitchen table
        o('trash_k', 'trash can', 0.5, 2.7, 0.35, 0.35, 0.5, (90, 90, 90)),
        o('extinguisher_k', 'fire extinguisher', 5.7, 3.0, 0.15, 0.15, 0.5, (220, 30, 30)),
        o('couch', 'couch', 9.0, 7.3, 2.0, 0.9, 0.85, (110, 70, 40)),
        o('coffee_table', 'table', 9.0, 5.5, 1.0, 0.6, 0.4, (150, 100, 60)),
        o('armchair_l', 'chair', 7.2, 6.0, 0.8, 0.8, 0.9, (215, 195, 155)),
        o('tv', 'tv', 11.7, 5.5, 0.15, 1.1, 0.7, (20, 20, 20)),
        o('plant', 'potted plant', 11.4, 7.5, 0.5, 0.5, 1.1, (40, 150, 60)),
        o('chair_l', 'chair', 7.0, 3.0, 0.45, 0.45, 0.9, (240, 130, 20)),
        o('extinguisher_l', 'fire extinguisher', 11.6, 2.9, 0.15, 0.15, 0.5, (220, 30, 30)),
        o('extinguisher_c', 'fire extinguisher', 11.5, 1.0, 0.15, 0.15, 0.5, (220, 30, 30)),
        o('trash_c', 'trash can', 6.0, 0.4, 0.35, 0.35, 0.5, (90, 90, 90)),
        o('chair_c', 'chair', 1.0, 1.0, 0.45, 0.45, 0.9, (130, 60, 160)),
    ]
    if variant_moved:
        for ob in objs:
            if ob.name == 'chair_k3':
                ob.pos = np.array([1.3, 3.4, ob.pos[2]])
    return objs


SYNONYM_NOISE = {'chair': ['armchair', 'seat', 'office chair'], 'table': ['desk', 'dining table'],
                 'tv': ['monitor', 'television'], 'trash can': ['bin'], 'couch': ['sofa']}
CONFUSE = {'chair': 'couch', 'bottle': 'cup', 'trash can': 'bottle', 'fire extinguisher': 'bottle'}


@dataclass
class ScenarioConfig:
    seed: int = 0
    rate_hz: float = 5.0
    speed_mps: float = 0.5
    p_detect: float = 0.85
    p_detect_small: float = 0.6
    label_synonym_p: float = 0.08
    label_confuse_p: float = 0.02
    p_transient_fp: float = 0.04
    p_flicker: float = 0.12
    flicker_pos: tuple[float, float] = (4.6, 6.8)
    bearing_range_m: float = 5.5
    bearing_small_range_m: float = 3.2
    hfov_deg: float = 100.0
    vfov_deg: float = 75.0
    max_range_m: float = 7.0
    sway_deg: float = 40.0
    person: bool = True
    move_chair: bool = True
    emit_view: bool = True
    crops: bool = True


@dataclass
class Scenario:
    grid: GridView
    gt_initial: list[GtObject]
    gt_final: list[GtObject]
    frames: list[Frame]
    robot_poses: list[tuple[float, float, float]]
    person_track: list[tuple[float, float, float]] = field(default_factory=list)
    cfg: ScenarioConfig = field(default_factory=ScenarioConfig)
    move_time: float = 0.0


PATROL = [(1.0, 1.0), (5.5, 1.0), (9.1, 1.0), (9.1, 3.0), (7.5, 4.9), (6.0, 4.9), (4.5, 4.9), (3.0, 3.0),
          (2.9, 1.0), (1.0, 1.0), (2.9, 1.0), (2.9, 3.2), (1.5, 6.0), (4.5, 6.4), (5.0, 3.0), (3.0, 3.0),
          (2.9, 1.0), (9.1, 1.0), (9.1, 3.2), (10.5, 5.0), (8.0, 7.0), (7.0, 4.9), (4.5, 4.9), (3.0, 3.0),
          (2.9, 1.0), (1.0, 1.0)]
PERSON_PATH = [(11.0, 1.0), (7.0, 1.0), (9.1, 1.0), (9.1, 3.5), (8.5, 4.5), (7.0, 4.5), (10.0, 4.0), (10.5, 6.5),
               (7.5, 6.0), (8.5, 3.0), (9.1, 1.0), (11.0, 1.0)]


def _densify(points, step: float):
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        d = math.dist(a, b)
        n = max(1, int(d / step))
        for k in range(1, n + 1):
            s = k / n
            out.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
    return out


def _crop(rgb: tuple[int, int, int], rng: np.random.Generator, size: int = 64) -> bytes:
    img = np.full((size, size, 3), rng.integers(90, 170), np.uint8)
    img += rng.integers(0, 25, (size, size, 3), dtype=np.uint8)
    m = size // 5
    img[m:size - m, m:size - m] = np.clip(np.array(rgb) + rng.integers(-12, 12, 3), 0, 255)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def _unit(v):
    return v / np.linalg.norm(v)


def generate(cfg: ScenarioConfig | None = None, duration_s: float | None = None) -> Scenario:
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)
    grid = floorplan()
    gt0 = objects(False)
    gt1 = objects(True) if cfg.move_chair else objects(False)
    path = _densify(PATROL, cfg.speed_mps / cfg.rate_hz)
    n_frames = len(path) if duration_s is None else min(len(path), int(duration_s * cfg.rate_hz))
    ppath = _densify(PERSON_PATH, 0.8 / cfg.rate_hz)
    move_k = n_frames // 2
    hfov, vfov = math.radians(cfg.hfov_deg), math.radians(cfg.vfov_deg)
    frames, poses, ptrack = [], [], []
    for k in range(n_frames):
        t = 100.0 + k / cfg.rate_hz
        x, y = path[k]
        nxt = path[min(k + 1, len(path) - 1)]
        heading = math.atan2(nxt[1] - y, nxt[0] - x) if nxt != (x, y) else (poses[-1][2] if poses else 0.0)
        yaw = heading + math.radians(cfg.sway_deg) * math.sin(2 * math.pi * k / (cfg.rate_hz * 6.0))
        poses.append((x, y, yaw))
        cam = np.array([x, y, 0.5])
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        gts = list(gt1 if (cfg.move_chair and k >= move_k) else gt0)
        if cfg.person:
            px, py = ppath[k % len(ppath)]
            ptrack.append((t, px, py))
            gts.append(GtObject('person', 'person', np.array([px, py, 0.9]), np.array([0.5, 0.5, 1.75]),
                                (60, 60, 160), True))
        dets: list[Detection] = []
        for g in gts:
            v = g.pos - cam
            r = float(np.linalg.norm(v))
            if not (0.5 <= r <= cfg.max_range_m):
                continue
            if math.acos(float(np.clip(_unit(v) @ fwd, -1, 1))) > 0.5 * min(hfov, vfov):
                continue
            if not grid.line_of_sight(cam, g.pos, ignore_end_m=0.45):
                continue
            small = float(np.max(g.size)) < 0.3
            if rng.random() > (cfg.p_detect_small if small else cfg.p_detect):
                continue
            label = g.label
            u = rng.random()
            if u < cfg.label_synonym_p and label in SYNONYM_NOISE:
                label = str(rng.choice(SYNONYM_NOISE[label]))
            elif u < cfg.label_synonym_p + cfg.label_confuse_p and label in CONFUSE:
                label = CONFUSE[label]
            area = float(g.size[0] * g.size[2] + g.size[1] * g.size[2]) / 2.0
            points = int(np.clip(700.0 * area / (r * r) * rng.uniform(0.6, 1.2), 0, 900))
            score = float(np.clip(rng.normal(0.68 - 0.03 * r, 0.12), 0.12, 0.98))
            bearing_only = (points < 8 or r > cfg.bearing_range_m or (small and r > cfg.bearing_small_range_m)
                            or rng.random() < 0.03)
            ang = 0.009
            b = _unit(v + r * rng.normal(0.0, ang, 3))
            toward = -_unit(np.array([v[0], v[1], 0.0]))
            frac = rng.uniform(0.6, 1.0, 3)
            ext = g.size * frac + rng.normal(0, 0.03, 3)
            cen = g.pos + toward * (0.12 * max(g.size[0], g.size[1])) * rng.uniform(0.3, 1.0) \
                + rng.normal(0, 0.04 + 0.02 * r, 3)
            side = int(40 + 90 / max(r, 0.8))
            box = (300, 200, 300 + side, 200 + side)
            dets.append(Detection(
                label=label, score=score, centroid=np.full(3, np.nan) if bearing_only else cen,
                extent=np.zeros(3) if bearing_only else np.abs(ext), bearing=b, bearing_only=bearing_only,
                point_count=0 if bearing_only else points, crop=_crop(g.rgb, rng) if cfg.crops else b'',
                box_xyxy=box))
        # false positives
        if rng.random() < cfg.p_transient_fp:
            lab = str(rng.choice(['chair', 'bottle', 'tv', 'person', 'box']))
            p = np.array([rng.uniform(0.5, 11.5), rng.uniform(0.5, 7.5), rng.uniform(0.1, 1.0)])
            if grid.line_of_sight(cam, p) and 0.5 < np.linalg.norm(p - cam) < 5.0:
                dets.append(Detection(lab, float(rng.uniform(0.2, 0.45)), p, np.array([0.4, 0.4, 0.5]),
                                      _unit(p - cam), False, 60, crop=_crop((120, 120, 120), rng) if cfg.crops else b'',
                                      box_xyxy=(300, 200, 360, 260)))
        fp = np.array([cfg.flicker_pos[0], cfg.flicker_pos[1], 0.3])
        if rng.random() < cfg.p_flicker and 0.6 < np.linalg.norm(fp - cam) < 5.0 and \
                math.acos(float(np.clip(_unit(fp - cam) @ fwd, -1, 1))) < 0.5 * min(hfov, vfov):
            dets.append(Detection('bottle', float(rng.uniform(0.25, 0.45)), fp + rng.normal(0, 0.05, 3),
                                  np.array([0.08, 0.08, 0.25]), _unit(fp - cam), False, 30,
                                  crop=_crop((150, 150, 150), rng) if cfg.crops else b'', box_xyxy=(300, 200, 340, 240)))
        view = ViewGeometry(cam, fwd, hfov, vfov, cfg.max_range_m) if cfg.emit_view else None
        frames.append(Frame(stamp=t, camera_position=cam, detections=dets, view=view))
    return Scenario(grid, gt0, gt1, frames, poses, ptrack, cfg, 100.0 + move_k / cfg.rate_hz)


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(smap, gt: list[GtObject], tol_m: float = 1.0, include_dynamic: bool = False) -> dict:
    """Compare confirmed map objects with the final ground truth.

    recall: GT objects that have a compatible confirmed object within tol_m. position error: distance to the nearest
    compatible confirmed object (horizontal). duplicates: confirmed objects beyond the first that are compatible
    with the same GT object within tol_m. false_positives: confirmed objects with no compatible GT within tol_m.
    """
    objs = smap.visible_objects()
    objs = [o for o in objs if include_dynamic or not o.dynamic]
    gts = [g for g in gt if include_dynamic or not g.dynamic]
    errs, found, dup = [], 0, 0
    assigned: set[int] = set()
    for g in gts:
        cands = [(float(np.linalg.norm(o.position[:2] - g.pos[:2])), o) for o in objs
                 if smap.vocab.compat(smap.label(o), g.label) == 2]
        cands = [c for c in cands if c[0] <= tol_m]
        if cands:
            cands.sort(key=lambda c: c[0])
            found += 1
            errs.append(cands[0][0])
            dup += len(cands) - 1
            assigned.update(o.id for _, o in cands)
    fps = [o for o in objs if o.id not in assigned and
           not any(smap.vocab.compat(smap.label(o), g.label) == 2 and np.linalg.norm(o.position[:2] - g.pos[:2]) <= tol_m
                   for g in gts)]
    n = len(objs)
    return {
        'gt': len(gts), 'found': found, 'recall': found / max(len(gts), 1),
        'mean_err_m': float(np.mean(errs)) if errs else float('nan'),
        'median_err_m': float(np.median(errs)) if errs else float('nan'),
        'p90_err_m': float(np.percentile(errs, 90)) if errs else float('nan'),
        'duplicates': dup, 'duplicate_rate': dup / max(n, 1), 'false_positives': len(fps),
        'fp_labels': sorted(smap.label(o) for o in fps), 'map_objects': n,
    }


def person_error(smap, scenario: Scenario) -> float:
    """Distance between the mapped person and the true person at the last frame (nan if absent)."""
    ps = [o for o in smap.visible_objects(True) if smap.canon(o) == 'person']
    if not ps or not scenario.person_track:
        return float('nan')
    _, px, py = scenario.person_track[-1]
    return float(min(np.linalg.norm(o.position[:2] - np.array([px, py])) for o in ps))


def jpeg_bytes_to_array(b: bytes) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


__all__ = ['generate', 'evaluate', 'floorplan', 'objects', 'ScenarioConfig', 'Scenario', 'GtObject']
