"""Check the room plumbing: load the oracle map + set the named rooms for the current pose, print the Rooms part of the summary.
python3 /x30/scripts/e2e/check_rooms.py --world aws_hospital   (inside e2e-fw; the robot stays where it is)"""
import argparse, os, sys, time
import rclpy, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mapframe import _tf, convert_file
from run_episode import Rec

ap = argparse.ArgumentParser(); ap.add_argument("--world", default="aws_hospital"); a = ap.parse_args()
rclpy.init(); n = Rec()
for _ in range(50):
    rclpy.spin_once(n, timeout_sec=0.2)
    if n.pose: break
st = n.pose
print("pose (world):", [round(v, 2) for v in st])
convert_file(f"/x30/results/maps/{a.world}.gt.json", "/x30/results/maps/_check_load.json", st[:2], st[2], True)
print("reset", n.call("/semantic_map/reset", {}), "load", n.call("/semantic_map/load", {"path": "/x30/results/maps/_check_load.json"}))
defs = yaml.safe_load(open(f"/x30/configs/rooms/{a.world}.yaml"))["rooms"]
m = []
for d in defs:
    x0, y0, x1, y1 = d["rect"]; c = [_tf([x, y], st[:2], st[2], True) for x, y in ((x0, y0), (x1, y1))]
    m.append({"name": d["name"], "rect": [min(c[0][0], c[1][0]), min(c[0][1], c[1][1]), max(c[0][0], c[1][0]), max(c[0][1], c[1][1])]})
print("set_rooms", n.call("/semantic_map/set_rooms", {"rooms": m})); time.sleep(3)
r = n.call("/semantic_map/summary", {"pose": [0.0, 0.0, 0.0], "max_objects": 8})
print(r.get("text", r) if isinstance(r, dict) else r)
rclpy.shutdown()
