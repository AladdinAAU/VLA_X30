"""Top-down preview of a registered world (occupancy map + objects + rooms)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .gridmap import GridMap  # noqa: E402


def preview(world_dir: Path, out_png: Path | None = None, episode: dict | None = None, path_xy=None,
            robot_radius: float | None = None, labels: bool = True) -> Path:
    world_dir = Path(world_dir)
    grid = GridMap.load(world_dir / "map.yaml")
    h, w = grid.shape
    x0, y0 = grid.origin
    ext = (x0, x0 + w * grid.resolution, y0, y0 + h * grid.resolution)
    fig, ax = plt.subplots(figsize=(min(16, (ext[1] - ext[0]) / 3 + 2), min(12, (ext[3] - ext[2]) / 3 + 2)))
    ax.imshow(~grid.occ, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1.4)
    if robot_radius:  # reachable cells (robot centre) in green
        from . import gridmap as gm
        reach = gm.largest_component(grid.inflated(robot_radius))
        ov = np.zeros((*reach.shape, 4))
        ov[reach] = (0.2, 0.7, 0.3, 0.6)
        ax.imshow(ov, origin="lower", extent=ext)
    objs = json.loads((world_dir / "objects.json").read_text()) if (world_dir / "objects.json").exists() else []
    for o in objs:
        if o["label"] in ("rug", "window", "picture", "mirror", "whiteboard", "door"):
            continue
        b = o["bbox"]
        ax.add_patch(Rectangle((b[0], b[1]), b[3] - b[0], b[4] - b[1], fill=False, ec="tab:blue", lw=0.6))
        if labels:
            ax.text((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, o["label"][:6], fontsize=4, ha="center", va="center", color="tab:red")
    rooms_f = world_dir / "rooms.json"
    if rooms_f.exists():
        for r in json.loads(rooms_f.read_text())["rooms"]:
            ax.text(r["center"][0], r["center"][1], r["name"], fontsize=6, color="tab:green", ha="center", alpha=0.7)
    if path_xy is not None:
        ax.plot([p[0] for p in path_xy], [p[1] for p in path_xy], "m-", lw=1.5)
    if episode is not None:
        s = episode["start"]
        ax.plot(s["x"], s["y"], "go", ms=6)
        g = episode.get("goal", {})
        if "x" in g:
            ax.plot(g["x"], g["y"], "r*", ms=10)
    ax.set_aspect("equal")
    ax.set_title(world_dir.name, fontsize=8)
    out = Path(out_png) if out_png else world_dir / "preview.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
