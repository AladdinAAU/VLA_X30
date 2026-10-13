"""Figures for docs/END_TO_END.md -> docs/figures/e2e/. Host python (matplotlib + PIL).  python scripts/e2e/make_figs.py"""
import base64
import io
import json
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results", "e2e")
OUT = os.path.join(ROOT, "docs", "figures", "e2e")
WORLD = os.path.join(ROOT, "ros2_ws", "src", "x30_worlds", "worlds", "replica_frl_apartment_0")
os.makedirs(OUT, exist_ok=True)
ORIGIN, MRES = (-2.04, -10.19), 0.05
EPS = ["ep1_sofa", "ep2_chair", "ep3_tv"]


def box(ax, x, y, w, h, text, fc, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, label="", off=(0, 0.12), color="#333", rad=0.0, fs=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=1, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, fontsize=fs, ha="center", color=color)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.7); ax.axis("off")
    for (x, y, w, h, t, c) in [(0.1, 4.3, 3.0, 3.4, "container e2e-sim (ROS 2 Jazzy, Gazebo Harmonic)", "#fff3d6"),
                               (3.4, 0.2, 6.9, 7.9, "container e2e-fw (ROS 2 Humble), ROS_DOMAIN_ID=80", "#e3f0ff"),
                               (10.6, 4.6, 2.3, 1.7, "x30-perception (GPU)", "#e8f7e4"), (10.6, 1.6, 2.3, 1.7, "x30-vlm (GPU)", "#f4e6f7")]:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec="#555", lw=1.5, ls="--"))
        ax.text(x + 0.1, y + h - 0.2, t, fontsize=8, weight="bold", va="top")
    box(ax, 0.3, 6.0, 2.6, 0.9, "Gazebo world\nreplica_frl_apartment_0\nkinematic X30", "#fff")
    box(ax, 0.3, 5.0, 2.6, 0.9, "sim bridges: camera 640x480,\n4 lidars merged, /gt/*, TF", "#fff")
    box(ax, 10.8, 5.0, 1.9, 0.8, "FastAPI :8100\nYOLOE-11l-seg", "#fff")
    box(ax, 10.8, 2.0, 1.9, 0.8, "llama.cpp :8000\nQwen3-VL-8B Q4", "#fff")
    box(ax, 3.7, 5.9, 2.3, 0.8, "known_map_nav\n+ waypoint_runner", "#fff")
    box(ax, 3.7, 4.7, 2.3, 0.8, "next_viewpoint\n(new, /nav/next_frontier)", "#fffbe0")
    box(ax, 6.5, 6.5, 2.7, 0.9, "perception_client\n(lidar lifting, vocabulary)", "#fff")
    box(ax, 6.5, 5.2, 2.7, 0.9, "semantic_mapper\n(instances, /semantic_map/*)", "#fff")
    box(ax, 3.7, 3.2, 2.3, 1.0, "skill_server\n/x30/skill/*, /x30/primitive", "#fff")
    box(ax, 3.7, 1.9, 2.3, 0.9, "mode_manager\n/x30/navigate, /x30/mode", "#fff")
    box(ax, 6.5, 3.2, 2.7, 1.0, "reasoner_agent\n/x30/agent/run (tool loop)", "#fff")
    box(ax, 6.5, 1.9, 2.7, 0.9, "instruction_follower\n(idle in this run)", "#eee")
    box(ax, 3.7, 0.5, 2.3, 0.9, "cmd_vel_mux + safety_guard\n/cmd_vel_nav -> /cmd_vel", "#fff")
    box(ax, 6.5, 0.5, 2.7, 0.9, "run_episode.py (client,\nrecorder, ground truth)", "#f0f0f0")
    # sim -> fw
    arrow(ax, (2.9, 5.45), (6.5, 6.9), "/camera/color/image_raw, camera_info, /lidar_points", (-0.2, 0.3))
    arrow(ax, (2.9, 5.2), (3.7, 6.2), "/tf, /gt/pose", (-0.1, 0.1))
    arrow(ax, (3.7, 0.9), (1.6, 5.0), "/cmd_vel", (-0.9, -1.4), rad=-0.1)
    arrow(ax, (8.6, 6.5), (8.6, 6.1), "/perception/detections", (0.95, 0.0))
    arrow(ax, (9.2, 6.9), (10.8, 5.6), "HTTP /segment (JPEG)", (0.2, 0.25))
    arrow(ax, (10.8, 5.2), (9.2, 6.7), "", rad=0.0)
    arrow(ax, (6.5, 5.65), (6.0, 5.3), "")
    arrow(ax, (6.5, 4.0), (6.0, 3.9), "/semantic_map/query,find", (0.0, 0.1))
    arrow(ax, (8.0, 5.2), (8.0, 4.2), "/semantic_map/*", (0.7, 0.0))
    arrow(ax, (3.7, 3.7), (3.7, 5.0), "/nav/next_frontier", (-0.85, 0.0), rad=0.3)
    arrow(ax, (6.25, 4.3), (6.0, 6.0), "navigate_to_pose", (0.0, 0.0), rad=0.0)
    arrow(ax, (4.3, 3.2), (4.3, 1.4), "/cmd_vel_nav, /cmd_vel_skill", (-0.9, -1.0), rad=0.0)
    arrow(ax, (6.5, 3.4), (6.0, 2.4), "/x30/agent/run (delegation)", (0.55, -0.2))
    arrow(ax, (6.5, 3.6), (6.0, 3.6), "/x30/skill/*", (0.25, -0.3))
    arrow(ax, (9.2, 3.6), (10.8, 2.5), "OpenAI chat + tools", (0.0, 0.2))
    arrow(ax, (6.5, 0.95), (6.0, 2.0), "/x30/navigate", (-0.45, -0.25))
    arrow(ax, (4.5, 2.8), (4.5, 1.9), "/x30/agent/run", (0.0, 0.0))
    ax.text(6.5, 8.45, "End-to-end episode: instruction -> /x30/navigate (object_nav) -> reasoner tool calls -> skills -> navigation -> sim", fontsize=9, ha="center")
    fig.savefig(os.path.join(OUT, "architecture.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def load_map():
    im = np.array(Image.open(os.path.join(WORLD, "map.png")).convert("L"))
    return im


def to_world(odom_xy, start):
    x0, y0, yaw = start
    c, s = math.cos(yaw), math.sin(yaw)
    return x0 + c * odom_xy[0] - s * odom_xy[1], y0 + s * odom_xy[0] + c * odom_xy[1]


def gt_objects(label):
    o = json.load(open(os.path.join(WORLD, "objects.json")))
    o = o["objects"] if isinstance(o, dict) else o
    return [(x["pose"]["x"], x["pose"]["y"], x["size"]) for x in o if x["label"] == label]


def draw_base(ax, ep):
    im = load_map()
    h, w = im.shape
    ax.imshow(im, cmap="gray", extent=(ORIGIN[0], ORIGIN[0] + w * MRES, ORIGIN[1], ORIGIN[1] + h * MRES), origin="upper")
    r = json.load(open(os.path.join(RES, ep, "result.json")))
    sm = json.load(open(os.path.join(RES, ep, "semmap.json")))
    for gx, gy, sz in gt_objects(r["label"]):
        ax.add_patch(plt.Rectangle((gx - sz[0] / 2, gy - sz[1] / 2), sz[0], sz[1], fc="none", ec="#d62728", lw=1.5, ls="--"))
    ax.plot([], [], color="#d62728", ls="--", label=f"GT {r['label']} (bbox)")
    for o in sm["objects"]:
        wx, wy = to_world(o["position"], r["start"])
        ax.plot(wx, wy, "o", ms=5, color="#1f77b4")
        ax.text(wx + 0.08, wy + 0.08, f"{o['label']} #{o['id']}", fontsize=6, color="#1f77b4")
    ax.plot([], [], "o", color="#1f77b4", label="semantic map objects")
    ax.set_aspect("equal"); ax.set_xlabel("x world [m]"); ax.set_ylabel("y world [m]")
    return r


def episode_plots():
    for ep in EPS:
        if not os.path.exists(os.path.join(RES, ep, "result.json")):
            continue
        tr = np.load(os.path.join(RES, ep, "trajectory.npy"))
        fig, ax = plt.subplots(figsize=(5.2, 7.6))
        r = draw_base(ax, ep)
        ax.plot(tr[:, 1], tr[:, 2], "-", color="#2ca02c", lw=1.6, label="robot path (/gt/pose)")
        ax.plot(*r["start"][:2], "s", color="#2ca02c", label="start"); ax.plot(*r["final"][:2], "*", ms=12, color="#ff7f0e", label="stop")
        ax.set_title(f"{ep}: {'SUCCESS' if r['success_gt'] else 'not successful'} ({r['dist_to_nearest_m']:.2f} m to nearest {r['label']} centre)", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        fig.savefig(os.path.join(OUT, f"{ep}_topdown.png"), dpi=90, bbox_inches="tight"); plt.close(fig)
        rend = json.load(open(os.path.join(RES, ep, "semmap_render.json")))
        if rend and rend.get("png_base64"):
            Image.open(io.BytesIO(base64.b64decode(rend["png_base64"]))).convert("RGB").save(os.path.join(OUT, f"{ep}_semantic_map.png"), optimize=True)
        for tag in ("start", "end"):
            src = os.path.join(RES, ep, f"cam_{tag}.jpg")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(OUT, f"{ep}_cam_{tag}.jpg"))
        fb = os.path.join(RES, ep)
        turns = sorted(f for f in os.listdir(fb) if f.startswith("cam_turn"))
        for f in turns[::max(len(turns) // 3, 1)][:3]:
            shutil.copy(os.path.join(fb, f), os.path.join(OUT, f"{ep}_{f}"))


def gif(ep="ep1_sofa", n_frames=60):
    tr = np.load(os.path.join(RES, ep, "trajectory.npy"))
    fbk = json.load(open(os.path.join(RES, ep, "feedback.json")))
    t0 = tr[0, 0]
    frames = []
    idx = np.linspace(1, len(tr) - 1, n_frames).astype(int)
    for i in idx:
        fig, ax = plt.subplots(figsize=(4.2, 6.2))
        r = draw_base(ax, ep)
        ax.plot(tr[:i + 1, 1], tr[:i + 1, 2], "-", color="#2ca02c", lw=1.6)
        x, y, yaw = tr[i, 1:4]
        ax.annotate("", xy=(x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw)), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color="#ff7f0e", lw=2))
        sim_t = tr[i, 0] - t0
        cur = [f for f in fbk if f["sim"] <= sim_t]
        sk = cur[-1]["skill"] if cur else "-"
        turn = cur[-1]["turn"] if cur else 0
        ax.set_title(f"{ep}  sim {sim_t:5.0f} s  turn {turn}  skill: {sk}", fontsize=7)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=60); plt.close(fig)
        frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(os.path.join(OUT, f"{ep}_topdown.gif"), save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)


if __name__ == "__main__":
    architecture()
    episode_plots()
    gif()
    for f in sorted(os.listdir(OUT)):
        print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")
