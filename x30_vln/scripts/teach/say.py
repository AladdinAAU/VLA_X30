"""Say something to the robot (inside e2e-fw):  python3 /x30/scripts/teach/say.py "This is Marcus." [--at x y (map frame)] [--world-xy x y (world frame, start 0.93 11.48)]
Prints the reply and the interpretation. Without a position the followed person (or the robot) is used."""
import argparse, json, sys, rclpy
from rclpy.node import Node
from x30_msgs.srv import Json
ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("--at", nargs=2, type=float); ap.add_argument("--world-xy", nargs=2, type=float)
ap.add_argument("--start", nargs=2, type=float, default=[0.93, 11.48]); a = ap.parse_args()
body = {"text": a.text}
if a.at: body.update(x=a.at[0], y=a.at[1])
if a.world_xy: body.update(x=a.world_xy[0] - a.start[0], y=a.world_xy[1] - a.start[1])
rclpy.init(); n = Node("say"); c = n.create_client(Json, "/teach/say"); c.wait_for_service(20)
f = c.call_async(Json.Request(request=json.dumps(body))); rclpy.spin_until_future_complete(n, f, timeout_sec=60)
r = json.loads(f.result().response)
print("REPLY:", r.get("reply") or "(no reply)")
print({k: r.get(k) for k in ("ok", "action", "position", "position_source", "latency_s", "error")})
print("interpretation:", r.get("interpretation")); print("context region:", (r.get("context") or {}).get("region"))
