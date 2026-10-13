"""Room / corridor regions from the 2D occupancy grid, and rule-based region names (no ROS imports).

Method (similar in spirit to SysNav's wall-based room split): free space is eroded by the door clearance, so doorways
and narrow gaps disappear and the free space falls apart into connected cores; every free cell is then assigned to the
nearest core. Regions keep their ids across updates by overlap with the previous labelling. Names come from object
rules (a kitchen has a refrigerator), from the shape (long and narrow = corridor) or from the VLM annotator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import RoomsCfg
from .datatypes import Region
from .gridtools import GridView
from .semantic_map import SemanticMap


@dataclass
class RoomMap:
    """Region labelling of a grid: `labels` holds region ids on free cells, `near_*` the nearest region for any cell."""
    grid: GridView
    labels: np.ndarray                 # (H, W) int32, 0 = none
    near_labels: np.ndarray            # (H, W) int32 id of the nearest region cell
    near_dist_m: np.ndarray            # (H, W) float32 distance to it
    search_m: float
    next_id: int

    def label_at(self, x: float, y: float) -> int:
        """Region id at a world point; points on walls/furniture get the nearest region within the search radius."""
        c, r = self.grid.world_to_cell(x, y)
        if not self.grid.in_bounds(c, r):
            return 0
        if self.near_dist_m[r, c] <= self.search_m:
            return int(self.near_labels[r, c])
        return 0

    def mask(self, rid: int) -> np.ndarray:
        return self.labels == rid


def _nearest_label(zero_mask: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel the `values` entry and distance (cells) of the nearest pixel where zero_mask is True."""
    src = np.where(zero_mask, 0, 1).astype(np.uint8)
    dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[lab[zero_mask]] = values[zero_mask]
    return lut[lab], dist


def segment_rooms(grid: GridView, cfg: RoomsCfg, prev: RoomMap | None = None,
                  prev_regions: dict[int, Region] | None = None) -> tuple[RoomMap, dict[int, Region]]:
    res = grid.resolution
    free = grid.free_mask(cfg.free_threshold)
    empty = RoomMap(grid, np.zeros(free.shape, np.int32), np.zeros(free.shape, np.int32),
                    np.full(free.shape, 1e9, np.float32), cfg.object_room_search_m,
                    prev.next_id if prev else 1)
    if not free.any():
        return empty, {}
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res
    core = dist >= cfg.door_clearance_m
    n, core_lab = cv2.connectedComponents(core.astype(np.uint8), connectivity=4)
    keep = [i for i in range(1, n) if np.count_nonzero(core_lab == i) * res * res >= 0.1]
    if not keep:
        return empty, {}
    comp = np.zeros_like(core_lab)
    for new_id, i in enumerate(keep, start=1):
        comp[core_lab == i] = new_id
    near, d_core = _nearest_label(comp > 0, comp)
    assigned = np.where(free & (d_core * res <= cfg.assign_max_dist_m), near, 0).astype(np.int32)
    # drop regions that are too small, then renumber with ids stable w.r.t. the previous labelling
    ids = [i for i in np.unique(assigned) if i > 0 and np.count_nonzero(assigned == i) * res * res >= cfg.min_room_area_m2]
    next_id = prev.next_id if prev else 1
    mapping: dict[int, int] = {}
    taken: set[int] = set()
    if prev is not None and prev.labels.shape == assigned.shape:
        cand = []
        for i in ids:
            m = assigned == i
            prev_ids, counts = np.unique(prev.labels[m], return_counts=True)
            for pid, cnt in zip(prev_ids, counts):
                if pid > 0:
                    cand.append((int(cnt), int(i), int(pid)))
        for cnt, i, pid in sorted(cand, reverse=True):
            if i in mapping or pid in taken:
                continue
            if cnt >= 0.3 * min(np.count_nonzero(assigned == i), np.count_nonzero(prev.labels == pid)):
                mapping[i] = pid
                taken.add(pid)
    labels = np.zeros_like(assigned)
    for i in ids:
        if i not in mapping:
            mapping[i] = next_id
            next_id += 1
        labels[assigned == i] = mapping[i]
    region_ids = np.array(sorted(set(mapping.values())), np.int32)
    near_lab, d_reg = _nearest_label(labels > 0, labels) if region_ids.size else (labels, np.full(free.shape, 1e9))
    rooms = RoomMap(grid, labels, near_lab.astype(np.int32), (d_reg * res).astype(np.float32),
                    cfg.object_room_search_m, next_id)
    regions: dict[int, Region] = {}
    for rid in region_ids:
        rows, cols = np.nonzero(labels == rid)
        xs = grid.origin_x + (cols + 0.5) * res
        ys = grid.origin_y + (rows + 0.5) * res
        pts = np.stack([cols, rows], axis=1).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        width, length = sorted((w * res, h * res))
        old = (prev_regions or {}).get(int(rid))
        regions[int(rid)] = Region(
            id=int(rid), name=old.name if old else '', name_source=old.name_source if old else '',
            centroid=np.array([xs.mean(), ys.mean()]), bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
            area_m2=float(len(rows) * res * res), width_m=float(width), length_m=float(length),
            vlm_name_key=old.vlm_name_key if old else '')
    return rooms, regions


def region_label(r: Region) -> str:
    return r.name or f'area {r.id}'


def objects_by_region(smap: SemanticMap) -> dict[int, list]:
    out: dict[int, list] = {}
    for o in smap.visible_objects():
        rid = smap.room_id_at(o.position)
        if rid is not None:
            out.setdefault(rid, []).append(o)
    return out


def label_signature(smap: SemanticMap, objs: list) -> str:
    """Stable signature of the set of object labels in a region (used to cache VLM room names)."""
    return ','.join(sorted(f'{smap.canon(o)}' for o in objs))


def name_by_rules(smap: SemanticMap, cfg: RoomsCfg) -> None:
    """Fill region names that are not set by the VLM: object rules first, then corridor shape, else 'area N'."""
    rules = {rt: {smap.vocab.canon(k): float(w) for k, w in weights.items()} for rt, weights in cfg.rules.items()}
    with smap.lock:
        by_region = objects_by_region(smap)
        for rid, region in smap.regions.items():
            if region.name_source in ('vlm', 'user'):
                continue
            present = {smap.canon(o) for o in by_region.get(rid, [])}
            scores = {rt: sum(w for k, w in weights.items() if k in present) for rt, weights in rules.items()}
            best_type, best = max(sorted(scores.items()), key=lambda kv: kv[1], default=('', 0.0))
            if best >= cfg.min_score:
                region.name, region.name_source = best_type, 'rules'
            elif region.width_m <= cfg.corridor_max_width_m and \
                    region.length_m >= cfg.corridor_min_aspect * max(region.width_m, 0.1):
                region.name, region.name_source = 'corridor', 'shape'
            else:
                region.name, region.name_source = '', ''


def apply_user_rooms(smap: SemanticMap, grid: GridView, cfg: RoomsCfg, defs: list[dict]) -> None:
    """Use operator-defined rooms ({"name", "rect": [x0, y0, x1, y1]} in the map frame) instead of the automatic segmentation.
    Cells inside a rectangle get its id (1..N in list order); doorways and corridors stay unlabelled. Names are kept (name_source 'user')."""
    res = grid.resolution
    labels = np.zeros((grid.height, grid.width), np.int32)
    regions: dict[int, Region] = {}
    for i, d in enumerate(defs, start=1):
        x0, y0, x1, y1 = (float(v) for v in d['rect'])
        c0, r0 = grid.world_to_cell(min(x0, x1), min(y0, y1))
        c1, r1 = grid.world_to_cell(max(x0, x1), max(y0, y1))
        c0, c1 = max(c0, 0), min(c1 + 1, grid.width)
        r0, r1 = max(r0, 0), min(r1 + 1, grid.height)
        if c1 <= c0 or r1 <= r0:
            continue
        labels[r0:r1, c0:c1] = i
        xmin, ymin, xmax, ymax = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        regions[i] = Region(id=i, name=str(d['name']), name_source='user', centroid=np.array([(xmin + xmax) / 2, (ymin + ymax) / 2]),
                            bbox=(xmin, ymin, xmax, ymax), area_m2=(xmax - xmin) * (ymax - ymin),
                            width_m=min(xmax - xmin, ymax - ymin), length_m=max(xmax - xmin, ymax - ymin))
    near, d_reg = _nearest_label(labels > 0, labels)
    smap.set_rooms(RoomMap(grid, labels, near.astype(np.int32), (d_reg * res).astype(np.float32), cfg.object_room_search_m, len(defs) + 1), regions)


def update_rooms(smap: SemanticMap, grid: GridView, cfg: RoomsCfg) -> None:
    """Segment the grid, keep ids/names stable and refresh rule-based names."""
    rooms, regions = segment_rooms(grid, cfg, smap.room_map, smap.regions)
    smap.set_rooms(rooms, regions)
    name_by_rules(smap, cfg)


def distance_to_region_m(region: Region, x: float, y: float) -> float:
    return math.hypot(region.centroid[0] - x, region.centroid[1] - y)
