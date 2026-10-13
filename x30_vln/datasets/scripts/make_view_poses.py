"""Robot poses that look at ground-truth objects, for the detector check.  python make_view_poses.py <world> [n] [seed]
Writes results/worlds_check/<world>/poses.json: reachable cell (r=0.35), yaw towards an object 1.5-4.5 m away with line of sight."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_datasets import gridmap as gm  # noqa: E402
from x30_datasets import registry  # noqa: E402
from x30_datasets.paths import project_root, worlds_pkg_dir  # noqa: E402

SKIP = {"rug", "clutter", "book", "kitchen_utensil", "clothes", "remote_control", "cup", "bowl", "plate", "towel"}

if __name__ == "__main__":
    wid = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    e = registry.world_entry(wid)
    g = gm.GridMap.load(worlds_pkg_dir() / e["map_file"])
    reach = gm.largest_component(g.inflated(0.35))
    objs = [o for o in json.loads((worlds_pkg_dir() / e["objects_file"]).read_text()) if o["label"] not in SKIP and o["size"][2] > 0.25]
    free = np.argwhere(reach)
    poses = []
    for _ in range(4000):
        if len(poses) >= n:
            break
        iy, ix = free[rng.integers(len(free))]
        x, y = g.cell_to_world(int(iy), int(ix))
        o = objs[rng.integers(len(objs))]
        dx, dy = o["pose"]["x"] - x, o["pose"]["y"] - y
        d = float(np.hypot(dx, dy))
        if not 1.5 <= d <= 4.5:
            continue
        a, b = g.world_to_cell(x, y), g.world_to_cell(o["pose"]["x"], o["pose"]["y"])
        k = np.linspace(0, 1, int(d / 0.05))
        cells = [(int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t)) for t in k[:-int(0.5 / 0.05) - 1]]
        if any(g.in_bounds(*c) and g.occ[c] for c in cells):
            continue
        poses.append({"x": round(x, 2), "y": round(y, 2), "yaw": round(float(np.arctan2(dy, dx)) + float(rng.normal(0, 0.25)), 3)})
    out = project_root() / "results" / "worlds_check" / wid
    out.mkdir(parents=True, exist_ok=True)
    (out / "poses.json").write_text(json.dumps(poses))
    print(wid, len(poses), "poses ->", out)
