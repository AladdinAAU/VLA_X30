"""From server instances to 3D detections (no ROS imports): filtering, de-duplication, lifting, thermal statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .camera_model import CameraModel
from .config import Vocabulary
from .lidar_projector import Lift, ProjectedPoints, ProjectorParams, lift_mask, lift_mask_from_depth
from perception_server import rle
from perception_server.pipeline import box_iou
from .thermal import mask_pixels_to_thermal, thermal_stats


@dataclass
class LiftedDetection:
    label: str  # canonical label (aliases applied)
    prompt: str  # phrase the model matched
    score: float
    box_xyxy: tuple  # pixels in the working (sent) image
    mask_rle: dict
    mask: np.ndarray  # bool at working resolution
    bearing_only: bool
    point_count: int
    median_depth: float
    centroid: np.ndarray | None  # output frame
    extent: np.ndarray | None
    bearing_cam: np.ndarray  # camera frame
    depth_source: str
    reason: str = ''
    thermal_mean_c: float = float('nan')
    thermal_max_c: float = float('nan')
    extra: dict = field(default_factory=dict)


def filter_and_dedupe(
    instances: list[dict], vocab: Vocabulary, min_mask_px: int, ignore_regions: list, size_wh: tuple, dedupe_iou: float = 0.6
) -> list[tuple[dict, str, np.ndarray]]:
    """Apply per-class thresholds, mask size and ignore regions; merge aliases of the same object.

    Returns [(instance, canonical_label, mask)] sorted by score. Two instances with the same canonical label and
    overlapping boxes (e.g. 'chair' and 'office chair' on one chair) collapse into the higher-scoring one."""
    w, h = size_wh
    kept = []
    for inst in sorted(instances, key=lambda i: -i['score']):
        label = vocab.canonical(inst['label'])
        if inst['score'] < vocab.threshold(label):
            continue
        mask = rle.decode(inst['mask_rle'])
        if mask.shape != (h, w):
            raise ValueError(f'mask {mask.shape} does not match the working size {(h, w)}')
        area = int(mask.sum())
        if area < min_mask_px:
            continue
        ys, xs = np.nonzero(mask)
        mu, mv = xs.mean() / w, ys.mean() / h
        if any(r[0] <= mu <= r[2] and r[1] <= mv <= r[3] for r in ignore_regions):
            continue
        if any(k[1] == label and box_iou(k[0]['box'], inst['box']) > dedupe_iou for k in kept):
            continue
        kept.append((inst, label, mask))
    return kept


def lift_instances(
    instances: list[dict],
    vocab: Vocabulary,
    cam: CameraModel,
    proj: ProjectedPoints | None,
    params: ProjectorParams,
    T_out_cam: np.ndarray,
    min_mask_px: int,
    ignore_regions: list,
    depth_source: str = 'lidar',
    depth_img: np.ndarray | None = None,
    dedupe_iou: float = 0.6,
) -> list[LiftedDetection]:
    """`cam` is the camera model at the working (mask) resolution; `proj` the projected lidar points at that resolution."""
    out = []
    for inst, label, mask in filter_and_dedupe(
        instances, vocab, min_mask_px, ignore_regions, (cam.width, cam.height), dedupe_iou
    ):
        if depth_source == 'sim_depth' and depth_img is not None:
            lift = lift_mask_from_depth(depth_img, mask, cam, params, T_out_cam)
        elif depth_source == 'lidar' and proj is not None:
            lift = lift_mask(proj, mask, cam, params, T_out_cam)
        else:
            ys, xs = np.nonzero(mask)
            bearing = cam.rays(np.array([[xs.mean() + 0.5, ys.mean() + 0.5]]))[0]
            lift = Lift(True, 0, 0, float('nan'), None, None, bearing, np.zeros((0, 3)), 'no_depth_source')
        out.append(
            LiftedDetection(
                label,
                inst['label'],
                float(inst['score']),
                tuple(int(round(v)) for v in inst['box']),
                inst['mask_rle'],
                mask,
                lift.bearing_only,
                lift.n_used,
                lift.median_depth,
                lift.centroid,
                lift.extent,
                lift.bearing_cam,
                'none' if lift.bearing_only else depth_source,
                lift.reason,
            )
        )
    return out


def add_thermal(
    dets: list[LiftedDetection],
    cam: CameraModel,
    cam_th: CameraModel,
    T_th_cam: np.ndarray,
    temp_c: np.ndarray,
    default_depth_m: float,
) -> None:
    """Fill thermal_mean_c / thermal_max_c of each detection in place (temp_c: float32 degrees C at thermal resolution)."""
    for d in dets:
        depth = d.median_depth if np.isfinite(d.median_depth) else default_depth_m
        uv = mask_pixels_to_thermal(d.mask, cam, cam_th, T_th_cam, depth)
        d.thermal_mean_c, d.thermal_max_c = thermal_stats(temp_c, uv)


def mask_rle_json(rle_dict: dict) -> str:
    return json.dumps(rle_dict, separators=(',', ':'))
