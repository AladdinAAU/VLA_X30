"""Guided tour in the simulator: the robot follows a person who tells it the names of places (runs inside e2e-fw).

python3 /x30/scripts/teach/run_tour.py --name tour_1 [--timeout 150]
Stack: run_e2e.sh aws_hospital, scripts/follow/run_follow_stack.sh configs/follow/tour1.json, x30_teach node running (docs/TEACH_PLACES.md).

Like an operator: teleport the robot to the start, send the follow goal (armed), call /follow/lock while the nurse is in front, start the walk. The nurse walks
into Room C, Room E and a corridor spot (configs/follow/tour1/target.json). At each stop the runner waits until (a) the ground-truth person stands inside the
stop's rectangle for min_dwell_s and (b) the follow node reports state `follow` (the robot really tracks her), then says the stop's sentence with /teach/say (no
position given: the module uses the followed person's position, as it would for a real user). After the walk the places are read back, the map is saved
(map frame of the start pose) and scored against the ground-truth rooms:
  for each taught place: the ground-truth room that contains it (world frame), whether it is the intended room, the kind (room/spot), the owner, the region
  centroid inside the ground-truth room, the reply, the latency.
Writes results/teach/<name>/{tour.json, taught_map.json (also results/maps/<name>.json)}."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
import yaml
from rclpy.action import ActionClient
from std_msgs.msg import Empty
from x30_msgs.action import Navigate

sys.path.insert(0, "/x30/scripts/follow")
from run_follow_episode import Rec

STOPS = [  # (ground-truth room name or None for a corridor spot, centre hint for the corridor, sentence, minimum dwell [s])
    {"room": "Room C", "say": "This is Marcus.", "dwell": 3.0, "expect": {"kind": "room", "owner": "marcus", "name": "marcus"}},
    {"room": "Room E", "say": "This is Alexander.", "dwell": 3.0, "expect": {"kind": "room", "owner": "alexander", "name": "alexander"}},
    {"room": None, "rect": [3.7, -15.0, 5.2, -13.0], "say": "Here we deliver stuff.", "dwell": 3.0, "expect": {"kind": "spot", "owner": "", "name": "deliver"}},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tour_1")
    ap.add_argument("--start", nargs=3, type=float, default=[0.93, 11.48, 0.0])
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--rooms", default="/x30/configs/rooms/aws_hospital.yaml")
    a = ap.parse_args()
    out = f"/x30/results/teach/{a.name}"
    os.makedirs(out, exist_ok=True)
    rooms = {r["name"]: r["rect"] for r in yaml.safe_load(open(a.rooms))["rooms"]}
    for s in STOPS:
        s["rect"] = rooms[s["room"]] if s["room"] else s["rect"]
        s["done"], s["inside_since"], s["result"] = False, None, None
    ox, oy = a.start[0], a.start[1]
    rclpy.init()
    n = Rec()
    n.spin_for(2.0)
    n.call("/sim/set_robot_pose", {"x": a.start[0], "y": a.start[1], "yaw": a.start[2]}, 60.0)
    n.p_reset.publish(Empty())
    n.spin_for(5.0)
    n.call("/perception/vocabulary", {"op": "set", "labels": ["person", "chair", "bench", "door"]})
    n.call("/semantic_map/reset", {})
    cli = ActionClient(n, Navigate, "/x30/navigate")
    cli.wait_for_server(timeout_sec=30.0)
    goal = Navigate.Goal(mode="follow_person", instruction="Follow me, I will show you around.", episode_id=a.name, timeout_s=float(a.timeout),
                         params_json=json.dumps({"d_target": 2.5, "vmax": 0.6}))
    n.robot.clear(), n.person.clear(), n.people.clear(), n.status.clear()
    fut = cli.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=30.0)
    gh = fut.result()
    rfut = gh.get_result_async()
    wall0, lock = time.time(), None
    while lock is None and time.time() - wall0 < 40:
        n.spin_for(1.0)
        r = n.call("/follow/lock", {})
        if r and r.get("ok"):
            lock = r
    print("lock:", lock)
    if lock is None:
        raise SystemExit("lock failed")
    n.p_start.publish(Empty())
    log = []
    while not rfut.done() and time.time() - wall0 < a.timeout * 4 + 120:
        n.spin_for(0.3)
        if not n.person or not n.status:
            continue
        t, (px, py) = n.person[-1][0], n.person[-1][1:3]
        following = n.status[-1].get("state") == "follow" and t - n.status[-1]["t"] < 1.5
        for s in STOPS:
            if s["done"]:
                continue
            x0, y0, x1, y1 = s["rect"]
            if x0 <= px <= x1 and y0 <= py <= y1:
                s["inside_since"] = s["inside_since"] if s["inside_since"] is not None else t
            else:
                s["inside_since"] = None
            if s["inside_since"] is not None and t - s["inside_since"] >= s["dwell"] and following:
                s["done"] = True
                body = {"text": s["say"]}
                r = n.call("/teach/say", body, 60.0)
                s["result"] = r
                s["said_at"] = {"sim_t": t, "person_world": [px, py], "robot_status": n.status[-1]}
                print("SAID:", s["say"], "->", r.get("reply"), "| position", r.get("position"), r.get("position_source"))
        if all(s["done"] for s in STOPS):
            break
    gh.cancel_goal_async()
    n.spin_for(2.0)
    places = n.call("/semantic_map/places", {})["places"]
    print("save:", n.call("/semantic_map/save", {"path": f"/x30/results/maps/{a.name}.json"}))
    # ---- score against the ground truth (world frame = map + start)
    rep = []
    for s in STOPS:
        exp = s["expect"]
        r = (s.get("result") or {})
        applied = (r.get("applied") or {}).get("place") or {}
        p = next((q for q in places if q["name"].lower() == applied.get("name", "").lower()), None)
        row = {"said": s["say"], "intended_room": s["room"] or "corridor spot", "said_sim": bool(s["done"]), "reply": r.get("reply"), "latency_s": r.get("latency_s"),
               "position_source": r.get("position_source"), "person_world": (s.get("said_at") or {}).get("person_world")}
        if p:
            wx, wy = p["x"] + ox, p["y"] + oy
            gt_room = next((nm for nm, (x0, y0, x1, y1) in rooms.items() if x0 <= wx <= x1 and y0 <= wy <= y1), None)
            row.update(place=p["name"], kind=p["kind"], owner=p["owner"], function=p["function"], area_m2=p["area_m2"], world_xy=[round(wx, 2), round(wy, 2)],
                       gt_room_of_place=gt_room, in_intended_room=(gt_room == s["room"]) if s["room"] else (not gt_room),
                       kind_ok=p["kind"] == exp["kind"], owner_ok=exp["owner"] in (p["owner"] or "").lower(), name_ok=exp["name"] in p["name"].lower() or any(exp["name"] in (x or "").lower() for x in [p["function"]] + list(p["aliases"])))
        rep.append(row)
    json.dump({"name": a.name, "lock": lock, "stops": rep, "places": places}, open(out + "/tour.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
