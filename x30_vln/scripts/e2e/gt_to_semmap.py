"""Ground-truth semantic map of a world (oracle map): every GT object becomes a confirmed (pinned) map object.
python scripts/e2e/gt_to_semmap.py --world aws_hospital  -> results/maps/<world>.gt.json (+ nothing else), WORLD frame.
Load it in an episode with run_episode.py --load-map results/maps/<world>.gt.json (converted to the map frame there). Rooms are NOT in the file:
they are set with the /semantic_map/set_rooms service from configs/rooms/<world>.yaml (run_episode.py --rooms)."""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "x30_semantic_map"))
from x30_semantic_map.config import load_config
from x30_semantic_map.semantic_map import MapObject, SemanticMap

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="aws_hospital")
ap.add_argument("--skip", nargs="*", default=["clutter"])
a = ap.parse_args()
objs = json.load(open(os.path.join(ROOT, "ros2_ws", "src", "x30_worlds", "worlds", a.world, "objects.json")))
objs = objs["objects"] if isinstance(objs, dict) else objs
smap = SemanticMap(load_config(""))
n = 0
for o in objs:
    if o["label"] in a.skip:
        continue
    x0, y0, z0, x1, y1, z1 = o["bbox"]
    m = MapObject(id=int(o["id"]) + 1, first_seen=0.0, last_seen=0.0)
    m.votes = {o["label"]: 1.0}
    m.n_obs, m.n_3d, m.score_sum, m.point_count = 5, 5, 4.75, 500
    m.position = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    m.size = np.array([x1 - x0, y1 - y0, z1 - z0])
    m.pinned = True
    smap.objects[m.id] = m
    n += 1
smap.next_id = max(smap.objects) + 1
out = os.path.join(ROOT, "results", "maps", a.world + ".gt.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
smap.save(out)
print(n, "objects ->", out)
