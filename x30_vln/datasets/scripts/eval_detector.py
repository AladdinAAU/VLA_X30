"""Detector check on captured simulator frames.  python eval_detector.py <world> [<world> ...]
Needs results/worlds_check/<world>/{captured.json,frame_*.png} (sim_capture.py) and the perception server on 127.0.0.1:8100.
Per class recall of visible ground-truth objects (projected through the sim camera: 640x480, hfov 100, 0.5 m ahead of the body, +0.04 m)
and the fraction of detections without a matching same-class ground-truth object (conf >= CONF)."""
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_datasets import gridmap as gm  # noqa: E402
from x30_datasets import registry  # noqa: E402
from x30_datasets.paths import project_root, worlds_pkg_dir  # noqa: E402

VOCAB = ["chair", "sofa", "table", "bed", "tv", "lamp", "plant", "cabinet", "refrigerator", "sink", "toilet", "door", "window",
         "picture", "cushion", "desk", "shelf", "stool", "bench", "bicycle"]
CONF = 0.25
W, H = 640, 480
FX = (W / 2) / np.tan(np.radians(50.0))
URL = "http://127.0.0.1:8100/segment"


def project(box, cam):
    """Bounding box (x0,y0,z0,x1,y1,z1) -> image box [u0,v0,u1,v1] and depth of the centre, or None."""
    cx, cy, cz, yaw = cam
    pts = np.array([[x, y, z] for x in (box[0], box[3]) for y in (box[1], box[4]) for z in (box[2], box[5])])
    d = pts - [cx, cy, cz]
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = d[:, 0] * c + d[:, 1] * s
    left = -d[:, 0] * s + d[:, 1] * c
    up = d[:, 2]
    if (fwd < 0.15).any():
        m = fwd > 0.15
        if m.sum() < 2:
            return None
        fwd, left, up = fwd[m], left[m], up[m]
    u = W / 2 - FX * left / fwd
    v = H / 2 - FX * up / fwd
    cen = np.array([(box[0] + box[3]) / 2, (box[1] + box[4]) / 2]) - [cx, cy]
    depth = float(cen[0] * c + cen[1] * s)
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())], depth


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def run(world: str) -> dict:
    d = project_root() / "results" / "worlds_check" / world
    e = registry.world_entry(world)
    objs = [o for o in json.loads((worlds_pkg_dir() / e["objects_file"]).read_text()) if o["label"] in VOCAB]
    rec = json.loads((d / "captured.json").read_text())
    grid = gm.GridMap.load(worlds_pkg_dir() / e["map_file"])
    per = defaultdict(lambda: [0, 0])
    fp = defaultdict(int)
    n_det = n_fp = n_gt = 0
    labels_seen = defaultdict(int)
    for k, r in enumerate(rec):
        img = cv2.imread(str(d / r["frame"]))
        b64 = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()
        res = requests.post(URL, json={"image": b64, "vocab": VOCAB, "conf": CONF, "masks": False}, timeout=120).json()["instances"]
        cam = (r["x"] + 0.5 * np.cos(r["yaw"]), r["y"] + 0.5 * np.sin(r["yaw"]), r["z"] + 0.04, r["yaw"])
        vis = []
        allp = []  # every vocabulary object in front of the camera, whatever its size or occlusion: used to judge detections
        for o in objs:
            p = project(o["bbox"], cam)
            if p is not None and p[1] > 0.3:
                allp.append((o, [max(p[0][0], 0), max(p[0][1], 0), min(p[0][2], W), min(p[0][3], H)]))
        for o in objs:
            p = project(o["bbox"], cam)
            if p is None:
                continue
            (u0, v0, u1, v1), depth = p
            cu, cv_ = (u0 + u1) / 2, (v0 + v1) / 2
            u0c, v0c, u1c, v1c = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
            if not (0 <= cu < W and 0 <= cv_ < H and 0.6 <= depth <= 6.0 and (u1c - u0c) >= 24 and (v1c - v0c) >= 24):
                continue
            if (u1c - u0c) * (v1c - v0c) < 0.35 * (u1 - u0) * (v1 - v0):
                continue
            bb = o["bbox"]
            occl = False
            for t in np.linspace(0, 1, max(int(np.hypot(o["pose"]["x"] - cam[0], o["pose"]["y"] - cam[1]) / 0.05), 2)):
                px, py = cam[0] + (o["pose"]["x"] - cam[0]) * t, cam[1] + (o["pose"]["y"] - cam[1]) * t
                if bb[0] - 0.2 <= px <= bb[3] + 0.2 and bb[1] - 0.2 <= py <= bb[4] + 0.2:
                    break  # reached the object's own footprint: the rest of the ray is inside it
                iy, ix = grid.world_to_cell(px, py)
                if grid.in_bounds(iy, ix) and grid.occ[iy, ix]:
                    occl = True
                    break
            if occl:
                continue  # blocked in the 2D map by something else
            vis.append((o, [u0c, v0c, u1c, v1c]))
        for o, bx in vis:
            n_gt += 1
            per[o["label"]][1] += 1
            ok = any(i["label"] == o["label"] and (iou(i["box"], bx) >= 0.15 or (bx[0] <= (i["box"][0] + i["box"][2]) / 2 <= bx[2]
                     and bx[1] <= (i["box"][1] + i["box"][3]) / 2 <= bx[3])) for i in res)
            per[o["label"]][0] += int(ok)
        for i in res:
            n_det += 1
            labels_seen[i["label"]] += 1
            ok = any(o["label"] == i["label"] and (iou(i["box"], bx) >= 0.15 or (bx[0] <= (i["box"][0] + i["box"][2]) / 2 <= bx[2]
                     and bx[1] <= (i["box"][1] + i["box"][3]) / 2 <= bx[3])) for o, bx in allp)
            if not ok:
                n_fp += 1
                fp[i["label"]] += 1
            col = (0, 200, 0) if ok else (0, 0, 255)
            cv2.rectangle(img, tuple(int(v) for v in i["box"][:2]), tuple(int(v) for v in i["box"][2:]), col, 2)
            cv2.putText(img, f"{i['label']} {i['score']:.2f}", (int(i["box"][0]), int(i["box"][1]) - 4), 0, 0.5, col, 1)
        for o, bx in vis:
            cv2.rectangle(img, tuple(int(v) for v in bx[:2]), tuple(int(v) for v in bx[2:]), (255, 200, 0), 1)
        cv2.imwrite(str(d / f"overlay_{k:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    hit = sum(v[0] for v in per.values())
    out = {"world": world, "frames": len(rec), "gt_visible": n_gt, "gt_detected": hit, "recall": round(hit / max(n_gt, 1), 3),
           "detections": n_det, "false_positives": n_fp, "fp_rate": round(n_fp / max(n_det, 1), 3),
           "per_class_recall": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())}, "fp_by_label": dict(sorted(fp.items(), key=lambda kv: -kv[1])),
           "detected_labels": dict(sorted(labels_seen.items(), key=lambda kv: -kv[1]))}
    (d / "detector_eval.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for w in sys.argv[1:]:
        print(json.dumps(run(w))[:1500])
