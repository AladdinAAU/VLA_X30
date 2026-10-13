"""Navigation, localisation and mapping metrics. Pure functions over numpy arrays / plain dicts.

Conventions: world-frame positions in metres, yaw in radians, GT objects and map objects are dicts with
``label, x, y`` (and optionally ``sx, sy`` footprint size, ``name``, ``id``, ``bearing_only``).
Definitions and defaults are justified in docs/EVALUATION.md.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .geometry import SE2, align_umeyama_2d, align_yaw, path_length, point_to_box_distance, resample_path, wrap
from .grid import OccGrid


# --------------------------------------------------------------------------- labels and goal distance
def normalise_label(label: str) -> str:
    return " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())


def label_match(a: str, b: str, synonyms: dict[str, Sequence[str]] | None = None) -> bool:
    """Case-insensitive label equality, optionally through a synonym table {canonical: [alias, ...]}."""
    na, nb = normalise_label(a), normalise_label(b)
    if na == nb:
        return True
    for canon, aliases in (synonyms or {}).items():
        group = {normalise_label(canon)} | {normalise_label(x) for x in aliases}
        if na in group and nb in group:
            return True
    return False


def object_distance(xy: Sequence[float], obj: dict, mode: str = "surface") -> float:
    """Distance from a robot position to a GT object: 'centroid' or 'surface' (axis-aligned footprint box)."""
    if mode == "surface" and obj.get("sx") and obj.get("sy"):
        return point_to_box_distance(xy, (obj["x"], obj["y"]), (obj["sx"], obj["sy"]))
    return math.hypot(xy[0] - obj["x"], xy[1] - obj["y"])


def goal_distance_mode(goal: dict, default: str = "surface") -> str:
    """'surface' or 'centroid' for this goal: the episode's ``distance_to`` (bbox | centroid | centre) wins over the config default."""
    d = str(goal.get("distance_to", "")).lower()
    return "surface" if d in ("bbox", "surface") else "centroid" if d in ("centroid", "centre", "center") else default


def _is_instance(o: dict, name: str) -> bool:
    return name in (o.get("name"), f"{o['label']}_{o.get('id')}")


def goal_instances(goal: dict, gt_objects: Iterable[dict], synonyms=None) -> list[dict]:
    """GT objects that satisfy an object goal: any instance of the label, or only the listed ``instances`` when
    the episode sets ``any_instance: false``."""
    inst = [o for o in gt_objects if label_match(o["label"], goal["label"], synonyms)]
    if goal.get("any_instance") is False and goal.get("instances"):
        inst = [o for o in inst if any(_is_instance(o, n) for n in goal["instances"])]
    return inst


def find_named_instance(goal: dict, gt_objects: Iterable[dict]) -> dict | None:
    """The specific instance an episode names (``instance``, else the first of ``instances``), if the scene has it."""
    name = goal.get("instance") or (goal.get("instances") or [None])[0]
    if not name:
        return None
    return next((o for o in gt_objects if _is_instance(o, name)), None)


def distance_series(xy: np.ndarray, goal: dict, gt_objects: Sequence[dict], synonyms=None,
                    mode: str = "surface") -> np.ndarray:
    """Distance to goal for every trajectory sample (object: nearest satisfying instance)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if goal["type"] == "pose":
        return np.hypot(xy[:, 0] - goal["x"], xy[:, 1] - goal["y"])
    inst = goal_instances(goal, gt_objects, synonyms)
    if not inst:
        raise ValueError(f"no GT instance of label {goal['label']!r}")
    return np.min([[object_distance(p, o, mode) for p in xy] for o in inst], axis=0)


def instance_distance_series(xy: np.ndarray, goal: dict, gt_objects: Sequence[dict], mode: str = "surface"):
    """Distance series to the specifically named instance (None when the goal names no instance or it is unknown)."""
    inst = find_named_instance(goal, gt_objects)
    if inst is None:
        return None
    return np.array([object_distance(p, inst, mode) for p in np.asarray(xy, float).reshape(-1, 2)])


# --------------------------------------------------------------------------- navigation metrics
def path_length_deadband(xy, deadband: float = 0.02) -> float:
    """Path length of a sampled trajectory ignoring jitter smaller than ``deadband`` metres between kept samples."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return 0.0
    total, last = 0.0, xy[0]
    for p in xy[1:]:
        d = float(np.hypot(*(p - last)))
        if d >= deadband:
            total += d
            last = p
    return total


def spl(success: bool, shortest_m: float | None, path_m: float) -> float:
    """Success weighted by (clipped) Path Length: S * l / max(p, l). 0 if failure or no valid optimum."""
    if not success or shortest_m is None or shortest_m <= 0:
        return 0.0
    return float(shortest_m / max(path_m, shortest_m))


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic time warping cost (sum of Euclidean distances along the optimal warping path)."""
    n, m = len(a), len(b)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row, prev = acc[i], acc[i - 1]
        di = d[i - 1]
        for j in range(1, m + 1):
            row[j] = di[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
    return float(acc[n, m])


def ndtw(traj_xy, reference_xy, d_th: float = 3.0, spacing: float = 0.5, max_points: int = 400) -> float:
    """Normalised DTW (Ilharco et al. 2019): exp(-DTW(R, Q) / (|R| * d_th)).

    Both polylines are resampled to ``spacing`` metres first, so the score does not depend on the logging rate
    or on how sparse the reference waypoints are (documented deviation from R2R, where nodes are the samples)."""
    q = resample_path(traj_xy, spacing)
    r = resample_path(reference_xy, spacing)
    if len(q) > max_points or len(r) > max_points:
        s = max(path_length(q), path_length(r)) / max_points
        q, r = resample_path(q, s), resample_path(r, s)
    if len(q) == 0 or len(r) == 0:
        return 0.0
    return float(math.exp(-dtw(r, q) / (len(r) * d_th)))


def sdtw(success: bool, ndtw_value: float | None) -> float | None:
    return None if ndtw_value is None else (ndtw_value if success else 0.0)


# --------------------------------------------------------------------------- collisions and interventions
def footprint_points(length: float, width: float, step: float = 0.1) -> np.ndarray:
    """Sample points on the outline and inside of the robot footprint (body frame, x forward)."""
    xs = np.linspace(-length / 2, length / 2, max(2, int(round(length / step)) + 1))
    ys = np.linspace(-width / 2, width / 2, max(2, int(round(width / step)) + 1))
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def grid_collision_times(poses: np.ndarray, grid: OccGrid, length: float = 0.95, width: float = 0.45,
                         refractory_s: float = 1.0) -> list[float]:
    """Times at which the robot footprint overlaps an occupied cell (start of each contact run, runs closer than ``refractory_s`` merged).

    ``poses`` is (N, 4) = t, x, y, yaw in the world frame."""
    poses = np.asarray(poses, float)
    if len(poses) == 0:
        return []
    fp = footprint_points(length, width, step=max(grid.resolution, 0.05))
    hit = np.zeros(len(poses), bool)
    for i, (_, x, y, yaw) in enumerate(poses):
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * fp[:, 0] - s * fp[:, 1]
        wy = y + s * fp[:, 0] + c * fp[:, 1]
        rows = np.floor((wy - grid.origin[1]) / grid.resolution).astype(int)
        cols = np.floor((wx - grid.origin[0]) / grid.resolution).astype(int)
        inb = (rows >= 0) & (rows < grid.occ.shape[0]) & (cols >= 0) & (cols < grid.occ.shape[1])
        hit[i] = (~inb).any() or grid.occ[rows[inb], cols[inb]].any()
    starts = hit & ~np.concatenate([[False], hit[:-1]])
    return merge_events(poses[starts, 0], refractory_s)


def merge_events(times: Iterable[float], refractory_s: float) -> list[float]:
    """Collapse event timestamps closer than ``refractory_s`` to the previous kept event into one event."""
    out: list[float] = []
    for t in sorted(times):
        if not out or t - out[-1] >= refractory_s:
            out.append(float(t))
    return out


# --------------------------------------------------------------------------- localisation
def _pair_arrays(pairs: np.ndarray):
    p = np.asarray(pairs, float)
    return p[:, 1:3], p[:, 3], p[:, 4:6], p[:, 6]


def ate(pairs: np.ndarray, method: str = "yaw", min_pairs: int = 10) -> dict | None:
    """Absolute trajectory error of the SLAM/odometry pose against ground truth after SE(2) alignment.

    ``pairs`` is (N, 7): t, slam_x, slam_y, slam_yaw, gt_x, gt_y, gt_yaw (time-associated samples).
    method 'yaw'  : rotation = circular mean of the yaw differences, translation = mean residual (robust for short
                    or straight paths, stricter because rotation is not tuned to the positions);
    method 'se2'  : free least-squares rigid fit of the positions (Umeyama, 2D; optimistic, ill-conditioned on lines).
    Returns None when there are too few pairs."""
    pairs = np.asarray(pairs, float)
    if pairs.ndim != 2 or len(pairs) < min_pairs:
        return None
    mxy, myaw, gxy, gyaw = _pair_arrays(pairs)
    tf = align_yaw(mxy, myaw, gxy, gyaw) if method == "yaw" else align_umeyama_2d(mxy, gxy)
    err = np.linalg.norm(tf.apply(mxy) - gxy, axis=1)
    length = path_length(gxy)
    yaw_err = np.abs(wrap(myaw + tf.yaw - gyaw))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))), "ate_mean_m": float(err.mean()), "ate_max_m": float(err.max()),
        "ate_final_m": float(err[-1]), "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(yaw_err ** 2)))),
        "drift_pct": float(100.0 * np.sqrt(np.mean(err ** 2)) / max(length, 1e-6)), "path_len_m": length,
        "frame_yaw_offset_deg": float(np.degrees(tf.yaw)), "n_pairs": int(len(pairs)),
        "align": tf.as_dict(), "method": method,
    }


# --------------------------------------------------------------------------- semantic map
def semantic_map_metrics(map_objects: Sequence[dict], gt_objects: Sequence[dict], align: SE2 | None,
                         traj_xy: np.ndarray | None = None, match_radius: float = 1.0, cover_radius: float = 5.0,
                         synonyms=None, vocab: Sequence[str] | None = None,
                         ignore_labels: Sequence[str] = ()) -> dict:
    """Compare the final semantic map (map frame) with the GT objects (world frame).

    ``align`` maps map-frame positions to the world (from the trajectory, see ``ate``); None = identity.
    Matching is one-to-one, label-aware (synonyms allowed) and greedy by distance within ``match_radius``.
    * recall            matched GT / all considered GT objects
    * recall_covered    same, restricted to GT objects within ``cover_radius`` of the executed trajectory
                        (a robot that never visited a room cannot be expected to map it)
    * precision         matched map objects / map objects with a position
    * duplicate_rate    unmatched map objects of the right label lying within ``match_radius`` of a GT object that is
                        already matched, divided by map objects with a position (one object mapped twice)
    * pos_err_*         distance between matched pairs
    Objects with ``bearing_only`` have no position and are only counted (``n_bearing_only``)."""
    ignore = {normalise_label(x) for x in ignore_labels}
    vocab_n = {normalise_label(x) for x in vocab} if vocab else None

    def considered(o):
        lab = normalise_label(o["label"])
        return lab not in ignore and (vocab_n is None or lab in vocab_n or any(
            label_match(lab, v, synonyms) for v in vocab_n))

    gts = [o for o in gt_objects if considered(o)]
    pos_objs = [o for o in map_objects if not o.get("bearing_only") and considered(o)]
    n_bearing = sum(1 for o in map_objects if o.get("bearing_only"))
    if align is not None and pos_objs:
        pts = align.apply(np.array([[o["x"], o["y"]] for o in pos_objs]))
    else:
        pts = np.array([[o["x"], o["y"]] for o in pos_objs]).reshape(-1, 2)
    cand = []
    for i, o in enumerate(pos_objs):
        for j, g in enumerate(gts):
            if label_match(o["label"], g["label"], synonyms):
                d = math.hypot(pts[i][0] - g["x"], pts[i][1] - g["y"])
                if d <= match_radius:
                    cand.append((d, i, j))
    cand.sort()
    used_m, used_g, matches = set(), set(), []
    for d, i, j in cand:
        if i in used_m or j in used_g:
            continue
        used_m.add(i)
        used_g.add(j)
        matches.append((i, j, d))
    dup_objs = {i for d, i, j in cand if i not in used_m and j in used_g}
    covered = set(range(len(gts)))
    if traj_xy is not None and len(traj_xy):
        t = np.asarray(traj_xy, float).reshape(-1, 2)
        covered = {j for j, g in enumerate(gts) if np.min(np.hypot(t[:, 0] - g["x"], t[:, 1] - g["y"])) <= cover_radius}
    errs = [d for _, _, d in matches]
    n_gt, n_map = len(gts), len(pos_objs)
    matched_cov = sum(1 for _, j, _ in matches if j in covered)
    return {
        "n_gt": n_gt, "n_gt_covered": len(covered), "n_map": n_map, "n_bearing_only": n_bearing, "n_matched": len(matches),
        "recall": len(matches) / n_gt if n_gt else None,
        "recall_covered": matched_cov / len(covered) if covered else None,
        "precision": len(matches) / n_map if n_map else None,
        "duplicate_rate": len(dup_objs) / n_map if n_map else None,
        "pos_err_mean_m": float(np.mean(errs)) if errs else None,
        "pos_err_median_m": float(np.median(errs)) if errs else None,
        "match_radius_m": match_radius,
    }
