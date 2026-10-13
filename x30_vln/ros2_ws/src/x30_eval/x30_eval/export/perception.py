"""Frames + labels for checking the perception side (detection quality), from recorded episodes.

Per saved camera frame the export writes the system's detections (``pseudo``: what /perception/detections said, matched by time)
and, when camera intrinsics and extrinsics were recorded (``scene_info['camera']``), the GT objects projected into the image
(``gt_projected``). Projection uses the robot GT pose and the object footprint boxes; it does NOT model occlusion, so a GT box can
be listed for an object hidden behind a wall: agreement numbers are an upper bound on recall and a lower bound on precision.
Held-out episodes are excluded. Outputs: annotations.jsonl, pseudo_labels_coco.json, agreement.json, images/.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..core.geometry import quat_to_matrix
from ..core.metrics import label_match
from ..core.recording import Recording
from ..core.runlog import RunDir
from ..core.runner import safe_name


def project_object_box(obj: dict, robot_xyz_yaw: tuple[float, float, float, float], cam: dict) -> dict | None:
    """Pixel box [x1, y1, x2, y2] of a GT object's 3D box in the camera image, or None if it is not in view.

    ``cam``: {K: [fx, fy, cx, cy], width, height, base_to_cam: {t: [x, y, z], q: [x, y, z, w]}} where base_to_cam is the pose of the
    optical frame in base_link. The robot is assumed level (GT yaw only)."""
    x, y, z, yaw = robot_xyz_yaw
    c, s = np.cos(yaw), np.sin(yaw)
    r_wb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    r_bc = quat_to_matrix(*cam["base_to_cam"]["q"])
    t_bc = np.array(cam["base_to_cam"]["t"], float)
    sx, sy, sz = obj.get("sx") or 0.3, obj.get("sy") or 0.3, obj.get("sz") or 0.3
    cz = obj.get("z", 0.0)
    corners = np.array([[obj["x"] + dx * sx / 2, obj["y"] + dy * sy / 2, cz + dz * sz / 2] for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    p_base = (corners - np.array([x, y, z])) @ r_wb                    # world -> base (R_wb^T applied to row vectors)
    p_cam = (p_base - t_bc) @ r_bc                                      # base -> optical frame
    if (p_cam[:, 2] <= 0.2).any():
        return None                                                     # partly behind the camera: skip rather than guess
    fx, fy, cx, cy = cam["K"]
    u, v = fx * p_cam[:, 0] / p_cam[:, 2] + cx, fy * p_cam[:, 1] / p_cam[:, 2] + cy
    box = [float(max(u.min(), 0)), float(max(v.min(), 0)), float(min(u.max(), cam["width"])), float(min(v.max(), cam["height"]))]
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    return {"box": box, "depth": float(p_cam[:, 2].mean())}


def iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def agreement(frames: list[dict], iou_thr: float = 0.5, synonyms=None) -> dict:
    """Per label recall (projected GT boxes found by a detection) and precision (detections explained by a GT box) at ``iou_thr``."""
    tp, fn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
    for f in frames:
        if f.get("gt_projected") is None:
            continue
        used = set()
        for g in f["gt_projected"]:
            hit = next((i for i, d in enumerate(f["pseudo"]) if i not in used and label_match(d["label"], g["label"], synonyms) and iou(d["box"], g["box"]) >= iou_thr), None)
            if hit is None:
                fn[g["label"]] += 1
            else:
                used.add(hit)
                tp[g["label"]] += 1
        for i, d in enumerate(f["pseudo"]):
            if i not in used:
                fp[d["label"]] += 1
    labels = sorted(set(tp) | set(fn) | set(fp))
    per = {lab: {"tp": tp[lab], "fn": fn[lab], "fp": fp[lab], "recall": tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else None,
               "precision": tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else None} for lab in labels}
    T, N, P = sum(tp.values()), sum(fn.values()), sum(fp.values())
    return {"iou": iou_thr, "per_label": per, "recall": T / (T + N) if T + N else None, "precision": T / (T + P) if T + P else None,
            "note": "GT projection ignores occlusion: recall is an upper bound, precision a lower bound"}


def export_perception(run_dirs: list[Path], out_dir: Path, *, match_s: float = 0.35, copy_images: bool = True, synonyms=None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    frames_out: list[dict] = []
    skipped = defaultdict(int)
    for path in run_dirs:
        rd = RunDir(path)
        for res in rd.results():
            if res.get("status", "done") != "done" or res.get("split") == "heldout":
                skipped["heldout" if res.get("split") == "heldout" else "not_done"] += 1
                continue
            tp = rd.tracks / f"{safe_name(res['episode_id'] + '#r' + str(res['repeat']))}.json.gz"
            if not tp.exists():
                skipped["no_track"] += 1
                continue
            rec = Recording.load(tp)
            cam = (rec.scene_info or {}).get("camera")
            det_t = np.array([d["t"] for d in rec.detections]) if rec.detections else np.array([])
            for i, fr in enumerate(rec.frames):
                src = Path(fr["path"])
                if not src.exists():
                    skipped["missing_image"] += 1
                    continue
                dst_rel = f"images/{rd.path.name}_{safe_name(res['episode_id'])}_r{res['repeat']}_{i:05d}{src.suffix}"
                if copy_images:
                    shutil.copyfile(src, out_dir / dst_rel)
                pseudo = []
                if len(det_t):
                    j = int(np.argmin(np.abs(det_t - fr["t"])))
                    if abs(det_t[j] - fr["t"]) <= match_s:
                        pseudo = [{"label": d["label"], "score": d["score"], "box": d["box"]} for d in rec.detections[j]["items"] if not d.get("bearing_only")]
                    else:
                        skipped["no_detection_message"] += 1
                gt = None
                if cam:
                    g = fr["gt"]
                    pose = (g[0], g[1], fr.get("gt_z", 0.0), g[2])
                    gt = []
                    for o in rec.gt_objects:
                        pr = project_object_box(o, pose, cam)
                        if pr:
                            gt.append({"label": o["label"], "name": o.get("name"), **pr})
                frames_out.append({"image": dst_rel, "run": rd.path.name, "episode": res["episode_id"], "repeat": res["repeat"], "scene": res["scene"],
                                   "t": fr["t"], "gt_pose": fr["gt"], "pseudo": pseudo, "gt_projected": gt})
    with open(out_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for r in frames_out:
            f.write(json.dumps(r) + "\n")
    cats = sorted({d["label"] for r in frames_out for d in r["pseudo"]})
    coco = {"images": [{"id": i, "file_name": r["image"]} for i, r in enumerate(frames_out)], "categories": [{"id": k, "name": c} for k, c in enumerate(cats)],
            "annotations": [{"id": n, "image_id": i, "category_id": cats.index(d["label"]), "bbox": [d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]],
                             "score": d["score"], "area": (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), "iscrowd": 0}
                            for n, (i, d) in enumerate((i, d) for i, r in enumerate(frames_out) for d in r["pseudo"])]}
    (out_dir / "pseudo_labels_coco.json").write_text(json.dumps(coco), encoding="utf-8")
    stats = {"frames": len(frames_out), "frames_with_detections": sum(1 for r in frames_out if r["pseudo"]),
             "frames_with_gt_projection": sum(1 for r in frames_out if r["gt_projected"] is not None), "skipped": dict(skipped),
             "agreement": agreement(frames_out, synonyms=synonyms)}
    (out_dir / "agreement.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
