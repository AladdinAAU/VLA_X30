"""Markdown table of the real-scan worlds from registry.yaml.  python scripts/world_table.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_datasets import registry  # noqa: E402

REAL = ("replica", "replicacad", "habitat-test-scenes")
print("| world | source scene | extent m | visual tris | ceiling tris | collision boxes | reachable m2 (r) | GT objects | spawn |")
print("|---|---|---|---|---|---|---|---|---|")
for wid, e in registry.load()["worlds"].items():
    if e.get("source_dataset") not in REAL:
        continue
    s = e["stats"]
    sp = e["spawn"]
    print(f"| {wid} | {e.get('source_scene_id', '')} | {'x'.join(str(v) for v in s['extent_m'])} | {s.get('triangles_visual', '')} | "
          f"{s.get('triangles_ceiling', '')} | {s['collision_boxes']} | {s['navigable_area_m2']} ({e.get('robot_radius_m', 0.35)}) | "
          f"{s['objects']} | {sp['x']}, {sp['y']} |")
