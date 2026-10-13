"""Are the doors passable?  For each ground-truth door: is the robot-centre reachable set (largest component) present within 0.7 m of
the door centre at radius 0.25 / 0.35?   python scripts/door_check.py <world> ..."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_datasets import gridmap as gm  # noqa: E402
from x30_datasets import registry  # noqa: E402
from x30_datasets.paths import worlds_pkg_dir  # noqa: E402

for w in sys.argv[1:]:
    e = registry.world_entry(w)
    g = gm.GridMap.load(worlds_pkg_dir() / e["map_file"])
    doors = [o for o in json.loads((worlds_pkg_dir() / e["objects_file"]).read_text()) if o["label"] == "door"]
    row = {}
    for r in (0.25, 0.35):
        reach = gm.largest_component(g.inflated(r))
        n = 0
        for o in doors:
            iy, ix = g.world_to_cell(o["pose"]["x"], o["pose"]["y"])
            k = int(0.7 / g.resolution)
            sub = reach[max(iy - k, 0):iy + k + 1, max(ix - k, 0):ix + k + 1]
            n += int(sub.any())
        row[r] = f"{n}/{len(doors)}"
    print(w, "doors with reachable floor within 0.7 m (r=0.25, 0.35):", row)
