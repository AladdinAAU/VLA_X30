"""Sample a scan tour for the mapping phase from a world's 2D navigation map (host, needs numpy scipy pillow pyyaml).

python scripts/e2e/sample_tour.py --world replica_frl_apartment_0 --start 3.99 -1.51 [--clearance 0.3] [--spacing 2.0] [--max 10]
Writes results/maps/<world>.tour.json = {"waypoints": [[x, y], ...]} in the world frame: free cells connected to the start after
eroding by --clearance, farthest-point sampling with min --spacing, ordered greedily nearest-next.
"""
import argparse
import json
import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="replica_frl_apartment_0")
ap.add_argument("--start", nargs=2, type=float, required=True)
ap.add_argument("--clearance", type=float, default=0.3)
ap.add_argument("--spacing", type=float, default=2.0)
ap.add_argument("--max", type=int, default=10)
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
wd = os.path.join(root, "ros2_ws", "src", "x30_worlds", "worlds", a.world)
m = yaml.safe_load(open(os.path.join(wd, "map.yaml")))
res, (ox, oy) = m["resolution"], m["origin"][:2]
free = np.flipud(np.array(Image.open(os.path.join(wd, "map.png")).convert("L")) >= 230)
safe = ndi.binary_erosion(free, structure=np.ones((3, 3)), iterations=int(a.clearance / res))
lab, _ = ndi.label(safe)
l = lab[int((a.start[1] - oy) / res), int((a.start[0] - ox) / res)]
assert l > 0, "start not in safe free space"
ys, xs = np.nonzero(lab == l)
pts = np.c_[xs * res + ox, ys * res + oy]
sel = [np.array(a.start)]
while len(sel) <= a.max:
    d = np.min(np.linalg.norm(pts[:, None, :] - np.array(sel)[None], axis=2), axis=1)
    k = d.argmax()
    if d[k] < a.spacing:
        break
    sel.append(pts[k])
rest, cur, order = sel[1:], sel[0], []
while rest:
    j = int(np.argmin([np.linalg.norm(p - cur) for p in rest]))
    cur = rest.pop(j)
    order.append([round(float(cur[0]), 2), round(float(cur[1]), 2)])
os.makedirs(os.path.join(root, "results", "maps"), exist_ok=True)
json.dump({"world": a.world, "area_m2": round(len(pts) * res * res, 1), "waypoints": order},
          open(os.path.join(root, "results", "maps", a.world + ".tour.json"), "w"))
print(len(order), "waypoints, reachable area", round(len(pts) * res * res, 1), "m2:", order)
