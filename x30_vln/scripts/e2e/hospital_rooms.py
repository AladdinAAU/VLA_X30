"""Offline room segmentation of a world's 2D map with the semantic map's own algorithm + GT object contents per region.
python scripts/e2e/hospital_rooms.py --world aws_hospital   -> docs/figures/rooms/<world>_regions.png and results/maps/<world>.regions.json"""
import argparse, collections, json, os, sys
import cv2, numpy as np, yaml
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "x30_semantic_map"))
from x30_semantic_map.config import RoomsCfg
from x30_semantic_map.gridtools import GridView
from x30_semantic_map.rooms import segment_rooms

ap = argparse.ArgumentParser(); ap.add_argument("--world", default="aws_hospital"); a = ap.parse_args()
wd = os.path.join(ROOT, "ros2_ws", "src", "x30_worlds", "worlds", a.world)
m = yaml.safe_load(open(os.path.join(wd, "map.yaml"))); g = np.array(Image.open(os.path.join(wd, "map.png")).convert("L"))
occ = np.full(g.shape, -1, np.int8); occ[g >= 230] = 0; occ[g <= 50] = 100
grid = GridView(np.flipud(occ), m["resolution"], m["origin"][0], m["origin"][1])
rm, regions = segment_rooms(grid, RoomsCfg())
objs = json.load(open(os.path.join(wd, "objects.json"))); objs = objs["objects"] if isinstance(objs, dict) else objs
per = collections.defaultdict(list)
for o in objs:
    rid = rm.label_at(o["pose"]["x"], o["pose"]["y"])
    if rid: per[rid].append(o["label"])
out = []
for rid, r in sorted(regions.items()):
    c = collections.Counter(per[rid]).most_common(6)
    out.append({"id": rid, "centroid": [round(float(r.centroid[0]), 2), round(float(r.centroid[1]), 2)], "area_m2": round(r.area_m2, 1),
                "width_m": round(r.width_m, 1), "length_m": round(r.length_m, 1), "rule_name": r.name, "gt_objects": c})
os.makedirs(os.path.join(ROOT, "results", "maps"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "maps", a.world + ".regions.json"), "w"), indent=1)
img = np.full(rm.labels.shape + (3,), 255, np.uint8); rng = np.random.RandomState(3)
for rid in regions: img[rm.labels == rid] = rng.randint(60, 230, 3)
img[occ == 100] = 0
img = np.flipud(img).copy(); H = img.shape[0]
for rid, r in regions.items():
    c, rr = grid.world_to_cell(float(r.centroid[0]), float(r.centroid[1]))
    cv2.putText(img, str(rid), (int(c) - 6, H - 1 - int(rr)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
os.makedirs(os.path.join(ROOT, "docs", "figures", "rooms"), exist_ok=True)
cv2.imwrite(os.path.join(ROOT, "docs", "figures", "rooms", a.world + "_regions.png"), cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print(len(regions), "regions"); [print(r) for r in out]
