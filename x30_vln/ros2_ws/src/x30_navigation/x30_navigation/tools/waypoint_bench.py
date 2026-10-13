"""waypoint_bench: random waypoints through /nav/follow_waypoints, scored against ground truth (/gt/pose, simulator map raster).

  ros2 run x30_navigation waypoint_bench --gt-map <map.yaml> --goals 12 --out wp.json [--radius 0.4] [--seed 0]

Waypoints are chosen in the world frame on free, reachable cells (clearance --goal-clearance), at least --min-dist of path
distance from the previous waypoint, converted to the map frame (map origin = robot pose at start). One service call per waypoint.
Success = GT distance to the waypoint below --radius; efficiency = shortest GT path / travelled GT path; collisions = GT clearance
below --contact-dist (one per approach). Needs the same GT conventions as goal_benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import rclpy
from scipy import ndimage
from x30_msgs.srv import Json

from ..geometry import matrix_to_rpy
from .goal_benchmark import Bench, geodesic_field, load_gt_map


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-map", required=True)
    ap.add_argument("--goals", type=int, default=12)
    ap.add_argument("--min-dist", type=float, default=3.0)
    ap.add_argument("--max-dist", type=float, default=12.0)
    ap.add_argument("--goal-clearance", type=float, default=0.6)
    ap.add_argument("--robot-radius", type=float, default=0.3)
    ap.add_argument("--contact-dist", type=float, default=0.15)
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--map-origin", default="", help="world x,y,yaw of the map frame origin (robot pose when the odometry was reset); default: first GT pose")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="waypoint_bench.json")
    args = ap.parse_args(argv)
    res = 0.1
    blocked, origin = load_gt_map(args.gt_map, res)
    dist = ndimage.distance_transform_edt(~blocked) * res
    free_path, free_goal = dist > args.robot_radius, dist > args.goal_clearance
    rng = np.random.default_rng(args.seed + 100)

    rclpy.init()
    node = Bench()
    while node.gt is None:
        rclpy.spin_once(node, timeout_sec=0.2)
    cli = node.create_client(Json, "/nav/follow_waypoints")
    if not cli.wait_for_service(timeout_sec=120.0):
        raise SystemExit("/nav/follow_waypoints not available")
    start = node.gt.copy()
    if args.map_origin:
        ox, oy, oyaw = (float(v) for v in args.map_origin.split(","))
        start[:2, 3] = (ox, oy)
        start[:3, :3] = [[math.cos(oyaw), -math.sin(oyaw), 0], [math.sin(oyaw), math.cos(oyaw), 0], [0, 0, 1]]
    node.clearance = (dist, origin, res, args.contact_dist)
    yaw0 = matrix_to_rpy(start[:3, :3])[2]
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    d0, idx0 = geodesic_field(free_path, res, start[:2, 3], origin)
    reach = np.isfinite(d0[idx0.clip(0)]) & (idx0 >= 0)
    rows = []
    for i in range(args.goals):
        node.spin_for(0.5)
        here = node.gt[:2, 3].copy()
        d, idx = geodesic_field(free_path, res, here, origin)
        ys, xs = np.nonzero(free_goal & reach)
        dd = d[idx[ys, xs]]
        ok = np.nonzero(np.isfinite(dd) & (dd >= args.min_dist) & (dd <= args.max_dist))[0]
        if len(ok) == 0:
            break
        k = int(rng.choice(ok))
        gw = np.array([xs[k] * res + origin[0] + res / 2, ys[k] * res + origin[1] + res / 2])
        shortest = float(dd[k])
        dw = gw - start[:2, 3]
        gm = (c0 * dw[0] + s0 * dw[1], -s0 * dw[0] + c0 * dw[1])
        col0, node.path_len = node.collisions, 0.0
        node.last_xy = node.gt[:2, 3].copy()
        fut = cli.call_async(Json.Request(request=json.dumps({"waypoints": [[gm[0], gm[1]]], "radius": args.radius, "timeout_s": args.timeout})))
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < args.timeout + 30:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.spin_for(0.5)
        r = json.loads(fut.result().response)["results"][0] if fut.done() else {"status": "no_answer"}
        err = float(np.linalg.norm(node.gt[:2, 3] - gw))
        row = {"i": i, "goal_world": [round(float(gw[0]), 2), round(float(gw[1]), 2)], "status": r.get("status"), "runner_error_m": r.get("final_error_m"),
               "gt_error_m": round(err, 3), "success": bool(err < args.radius), "shortest_m": round(shortest, 2), "travelled_m": round(node.path_len, 2),
               "efficiency": round(min(1.0, shortest / max(node.path_len, 1e-6)), 3) if err < args.radius else None,
               "collisions": node.collisions - col0, "duration_s": round(time.monotonic() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in rows if r["success"]]
    summary = {"goals": len(rows), "success": len(ok), "success_rate": round(len(ok) / max(1, len(rows)), 3),
               "mean_efficiency": round(float(np.mean([r["efficiency"] for r in ok])), 3) if ok else None,
               "collisions": sum(r["collisions"] for r in rows), "mean_duration_s": round(float(np.mean([r["duration_s"] for r in rows])), 1) if rows else None}
    print(json.dumps(summary))
    json.dump({"summary": summary, "goals": rows}, open(args.out, "w"), indent=2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
