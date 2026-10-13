#!/usr/bin/env python3
"""Top-down record of a known_map_bench run: occupancy grid, planned paths, waypoints and the /gt/pose trajectory -> PNG + GIF.
  python scripts/sim/make_nav_viz.py results/known_map/lab_01 docs/figures/navigation/known_map_lab_01
Needs numpy, matplotlib, pillow, pyyaml (host python). Paths are read from result.json (world-frame, same frame as the GT pose)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from PIL import Image
import yaml


def load(res_dir):
    r = json.load(open(os.path.join(res_dir, "result.json")))
    tr = np.load(os.path.join(res_dir, "trajectory.npy"))
    meta = yaml.safe_load(open(r["map_yaml"].replace("/x30/", os.path.join(os.path.dirname(__file__), "..", "..") + "/")))
    img = np.array(Image.open(os.path.join(os.path.dirname(r["map_yaml"].replace("/x30/", os.path.join(os.path.dirname(__file__), "..", "..") + "/")), meta["image"])).convert("L"), float) / 255
    occ = (img if meta.get("negate", 0) else 1 - img) >= meta.get("occupied_thresh", 0.65)
    h, w = occ.shape
    o, rs = meta["origin"], meta["resolution"]
    return r, tr, occ, [o[0], o[0] + w * rs, o[1], o[1] + h * rs]


def body(x, y, yaw, L=1.0, W=0.46):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def main(res_dir, out):
    r, tr, occ, ext = load(res_dir)
    s = r["summary"]
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=80)
    ax.imshow(~occ, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1.6, interpolation="nearest")
    first = {}
    for t, p in r["plans"]:
        if p.get("event") == "plan":
            first[tuple(np.round(p["goal_world"], 1))] = np.array(p["path"])
    for q in first.values():
        ax.plot(q[:, 0], q[:, 1], color="#1f77b4", lw=1.2, alpha=0.8, label="planned (A*, smoothed)" if q is list(first.values())[0] else None)
    for wp in r["waypoints"]:
        gx, gy = wp["goal_world"]
        ax.plot(gx, gy, "o", ms=9, mfc="#ffd92f" if wp["success"] else "#e41a1c", mec="k", zorder=5)
        ax.text(gx, gy, str(wp["i"] + 1), fontsize=7, ha="center", va="center", zorder=6)
    ax.plot(*r["start"][:2], "g^", ms=9, zorder=5, label="start")
    line, = ax.plot([], [], color="#d95f02", lw=1.6, label="trajectory (/gt/pose)")
    poly = Polygon(body(*tr[0, 2:5]), closed=True, fc="#d95f02", ec="k", alpha=0.8, zorder=7)
    ax.add_patch(poly)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, world)")
    ax.set_ylabel("y (m, world)")
    ax.set_title(f"{s['world']}: {s['reached']}/{s['waypoints']} reached, {s['travelled_total_m']} m (A* {s['astar_total_m']} m), "
                 f"{s['collisions']} collisions, RTF {s['rtf']}", fontsize=8)
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    line.set_data(tr[:, 2], tr[:, 3])
    poly.set_xy(body(*tr[-1, 2:5]))
    fig.savefig(out + ".png", dpi=60)
    idx = np.linspace(0, len(tr) - 1, min(110, len(tr))).astype(int)

    def upd(k):
        i = idx[k]
        line.set_data(tr[: i + 1, 2], tr[: i + 1, 3])
        poly.set_xy(body(*tr[i, 2:5]))
        return line, poly
    FuncAnimation(fig, upd, frames=len(idx)).save(out + ".gif", writer=PillowWriter(fps=12), dpi=60)
    print(out, os.path.getsize(out + ".png") // 1024, "KB png", os.path.getsize(out + ".gif") // 1024, "KB gif")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
