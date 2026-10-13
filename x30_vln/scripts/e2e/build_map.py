"""Mapping phase (plan B): explore a world once, save the semantic map, score it against ground truth.

Runs inside the e2e-fw container (sim in the same ROS domain), after the launch sequence in docs/END_TO_END.md:
python3 /x30/scripts/e2e/build_map.py --world replica_frl_apartment_0 [--start x y yaw] [--labels a b c] [--tour file.json]
Default: scan tour (results/maps/<world>.tour.json from scripts/e2e/sample_tour.py): goto_pose + 360 deg scan at each waypoint.
--frontier: use the explore skill instead (unreliable on the small 2D maps of the scan worlds: unreachable frontiers).

Steps: teleport (optional) + restart known_map_nav, set the detector vocabulary, reset semantic map and viewpoint memory, call the
`explore` skill repeatedly (steps=3, scan at each frontier) until exhausted / max-calls / wall limit, save the map to
/x30/results/maps/<world>.json and write <world>.report.json (per-label recall of GT instances within --match-radius).
Episodes then load the map with run_episode.py --load-map (no reset), so only instruction -> object -> goal is exercised.
"""
import argparse
import json
import math
import os
import sys
import time

import rclpy
from rclpy.action import ActionClient
from x30_msgs.action import Skill

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mapframe import convert_file  # noqa: E402
from run_episode import WORLD_OBJECTS, Rec, sh  # noqa: E402

DEFAULT_LABELS = ["sofa", "chair", "tv", "table", "vase", "plant", "lamp", "cabinet", "bicycle", "refrigerator", "sink",
                  "bottle", "bowl", "cup", "cushion", "trash can", "shelf", "bed"]


def run_skill(n, cli, name, args, timeout):
    goal = Skill.Goal(name=name, args_json=json.dumps(args), timeout_s=float(timeout))
    fut = cli.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=30.0)
    gh = fut.result()
    rfut = gh.get_result_async()
    t0 = time.time()
    while not rfut.done():
        rclpy.spin_once(n, timeout_sec=0.2)
        if time.time() - t0 > timeout * 3 + 30:
            gh.cancel_goal_async()
            return None
    return rfut.result().result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="replica_frl_apartment_0")
    ap.add_argument("--start", nargs=3, type=float, default=None)
    ap.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    ap.add_argument("--max-calls", type=int, default=12)
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--tour", default=None)
    ap.add_argument("--wall-limit", type=float, default=1500.0)
    ap.add_argument("--match-radius", type=float, default=1.5)
    a = ap.parse_args()
    os.makedirs("/x30/results/maps", exist_ok=True)
    rclpy.init()
    n = Rec()
    for _ in range(50):
        rclpy.spin_once(n, timeout_sec=0.2)
        if n.pose and n.img is not None:
            break
    if a.start:
        x, y, yaw = a.start
        print("set_robot_pose", n.call("/sim/set_robot_pose", {"x": x, "y": y, "yaw": yaw}, 60.0))
        time.sleep(4)
    # restart known_map_nav so odom is re-aligned to the world (same as run_episode.py)
    sh("pkill -f '[k]nown_map_nav|[w]aypoint_runner|[n]ext_viewpoint'; sleep 3")
    import subprocess
    subprocess.Popen(["bash", "-lc", "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80; "
                      "ros2 launch x30_navigation known_map_nav.launch.py world:=%s relay:=false > /x30/results/e2e/logs/nav_map_%s.log 2>&1"
                      % (a.world, a.world)])
    time.sleep(12)
    print("vocab:", n.call("/perception/vocabulary", {"op": "set", "labels": a.labels}))
    print("reset:", n.call("/semantic_map/reset", {}), n.call("/nav/next_viewpoint/reset", {}))
    st = ActionClient(n, Skill, "/x30/skill/stop")  # a skill left running by a killed client would answer 'Busy'
    st.wait_for_server(timeout_sec=30.0)
    run_skill(n, st, "stop", {}, 20)
    wall0, log, exhausted = time.time(), [], False
    if a.frontier:
        cli = ActionClient(n, Skill, "/x30/skill/explore")
        cli.wait_for_server(timeout_sec=30.0)
        for i in range(a.max_calls):
            if time.time() - wall0 > a.wall_limit:
                break
            r = run_skill(n, cli, "explore", {"strategy": "nearest", "steps": 3, "scan": True}, 240)
            d = json.loads(r.data_json) if r and r.data_json else {}
            log.append({"call": i, "wall": time.time() - wall0, "ok": bool(r and r.success), "msg": (r.message if r else "no result")[:200],
                        "remaining": d.get("remaining_frontiers"), "exhausted": d.get("exhausted")})
            print(log[-1], flush=True)
            if d.get("exhausted"):
                exhausted = True
                break
    else:
        tour = json.load(open(a.tour or f"/x30/results/maps/{a.world}.tour.json"))["waypoints"]
        gp = ActionClient(n, Skill, "/x30/skill/goto_pose")
        sc = ActionClient(n, Skill, "/x30/skill/scan")
        gp.wait_for_server(timeout_sec=30.0)
        sc.wait_for_server(timeout_sec=30.0)
        x0, y0, yaw0 = a.start if a.start else (0.0, 0.0, 0.0)
        for i, (wx, wy) in enumerate(tour):  # tour is in the world frame; goto_pose wants the map frame (= robot start)
            x, y = (math.cos(yaw0) * (wx - x0) + math.sin(yaw0) * (wy - y0), -math.sin(yaw0) * (wx - x0) + math.cos(yaw0) * (wy - y0))
            if time.time() - wall0 > a.wall_limit:
                break
            r = run_skill(n, gp, "goto_pose", {"x": x, "y": y, "tolerance_m": 0.5}, 120)
            ok = bool(r and r.success)
            m2 = ""
            if True:  # scan after a near miss too: the robot is usually within 1 m of the waypoint and still sees the room
                r2 = run_skill(n, sc, "scan", {"sweep_deg": 360, "step_deg": 60, "dwell_s": 1.5}, 120)
                m2 = (r2.message if r2 else "no result")[:160]
            log.append({"wp": i, "xy_world": [wx, wy], "xy_map": [round(x, 2), round(y, 2)], "wall": time.time() - wall0, "reached": ok,
                        "goto": (r.message if r else "no result")[:120], "scan": m2})
            print(log[-1], flush=True)
    raw = f"/x30/results/maps/{a.world}.mapframe.json"
    print("save:", n.call("/semantic_map/save", {"path": raw}))
    st0 = a.start if a.start else (0.0, 0.0, 0.0)
    convert_file(raw, f"/x30/results/maps/{a.world}.json", st0[:2], st0[2], inverse=False)  # stored in the world frame
    sm = n.call("/semantic_map/query", {"limit": 500})
    for o in sm["objects"]:  # report positions in the world frame
        o["position"] = [st0[0] + math.cos(st0[2]) * o["position"][0] - math.sin(st0[2]) * o["position"][1],
                         st0[1] + math.sin(st0[2]) * o["position"][0] + math.cos(st0[2]) * o["position"][1]]
    objs = json.load(open(WORLD_OBJECTS.format(world=a.world)))
    objs = objs["objects"] if isinstance(objs, dict) else objs
    want = set(a.labels)
    per = {}
    for o in objs:
        if o["label"] not in want:
            continue
        hit = any(m["label"] == o["label"] and math.hypot(m["position"][0] - o["pose"]["x"], m["position"][1] - o["pose"]["y"]) <= a.match_radius
                  for m in sm["objects"])
        s = per.setdefault(o["label"], [0, 0])
        s[0] += int(hit)
        s[1] += 1
    tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
    rep = {"world": a.world, "labels": a.labels, "explore_calls": len(log), "exhausted": exhausted, "wall_s": time.time() - wall0,
           "map_objects": sm["count"], "gt_recall_by_label": {k: f"{v[0]}/{v[1]}" for k, v in per.items()},
           "gt_recall_total": f"{tot[0]}/{tot[1]}", "match_radius_m": a.match_radius, "log": log}
    json.dump(rep, open(f"/x30/results/maps/{a.world}.report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "log"}, indent=1))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
