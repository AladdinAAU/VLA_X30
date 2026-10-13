"""known_map_bench: 12 (default) random waypoints in order through /nav/follow_waypoints on the known-map backend, scored with /gt/pose.

Waypoints are chosen on the world grid: reachable with the widest clearance tier, >= min-dist apart (A* length), clearance >= goal-clearance.
They are sent in ONE request (odom frame = robot start frame). Logs: trajectory (sim stamp, wall, x, y, yaw), planned paths from
/nav/known_map_info, per waypoint success (GT distance < radius), path length vs A* length (previous waypoint -> waypoint on the same map),
duration (sim and wall), collisions (oriented 1.0 x 0.46 body rectangle overlapping an occupied cell of the 0.05 m world raster; rising edges), real-time factor.
  ros2 run x30_navigation known_map_bench --world lab_01 --out /x30/results/known_map/lab_01
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from x30_msgs.srv import Json

from ..known_map import KnownMap, footprint_contact, load_world_grid, path_length, resolve_map
from .. import known_map_nav as _k


class Rec(Node):
    def __init__(self):
        super().__init__("known_map_bench")
        self.traj, self.plans, self.sim_t = [], [], 0.0
        self.create_subscription(PoseStamped, "/gt/pose", self.on_gt, 50)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "sim_t", m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self.create_subscription(String, "/nav/known_map_info", lambda m: self.plans.append((self.sim_t, json.loads(m.data))), 50)

    def on_gt(self, m):
        p = m.pose
        self.traj.append((self.sim_t, time.monotonic(), p.position.x, p.position.y, _k.yaw_of(p.orientation)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--registry", default="/x30/datasets/registry.yaml")
    ap.add_argument("--worlds-root", default="/x30/ros2_ws/src/x30_worlds")
    ap.add_argument("--goals", type=int, default=12)
    ap.add_argument("--min-dist", type=float, default=2.5)
    ap.add_argument("--max-dist", type=float, default=9.0)
    ap.add_argument("--goal-clearance", type=float, default=0.7)
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--timeout", type=float, default=600.0, help="wall seconds per waypoint")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ypath, _ = resolve_map(a.world, a.registry, a.worlds_root)
    fine, forg = load_world_grid(ypath, 0.05)
    blocked, origin = load_world_grid(ypath, 0.1)
    km = KnownMap(blocked, origin, 0.1)

    rclpy.init()
    node = Rec()
    while len(node.traj) < 5:
        rclpy.spin_once(node, timeout_sec=0.2)
    cli = node.create_client(Json, "/nav/follow_waypoints")
    if not cli.wait_for_service(timeout_sec=120.0):
        raise SystemExit("/nav/follow_waypoints not available")
    time.sleep(1.0)
    rclpy.spin_once(node, timeout_sec=0.2)
    x0, y0, yaw0 = node.traj[-1][2:5]
    reach = km.reachable_from([x0, y0], km.pp.radii[0]) & (km.dist >= a.goal_clearance)
    rng = np.random.default_rng(a.seed)
    cur, goals = np.array([x0, y0]), []
    for _ in range(a.goals):
        ys, xs = np.nonzero(reach)
        cand = rng.permutation(len(ys))[:300]
        pick = None
        for k in cand:
            g = km.xy((xs[k], ys[k]))
            r = km.plan(cur, g)
            if r is not None and a.min_dist <= path_length(r[0]) <= a.max_dist:
                pick = (g, path_length(r[0]))
                break
        if pick is None:
            break
        goals.append(pick)
        cur = pick[0]
    c, s = math.cos(yaw0), math.sin(yaw0)
    odom = [[c * (g[0] - x0) + s * (g[1] - y0), -s * (g[0] - x0) + c * (g[1] - y0)] for g, _ in goals]
    print(f"{len(goals)} waypoints, start {x0:.2f},{y0:.2f},{yaw0:.2f}", flush=True)
    t_wall0, t_sim0, i0 = time.monotonic(), node.sim_t, len(node.traj)
    fut = cli.call_async(Json.Request(request=json.dumps({"waypoints": odom, "radius": a.radius, "timeout_s": a.timeout})))
    while not fut.done() and time.monotonic() - t_wall0 < a.timeout * len(goals) + 60:
        rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(0.5)
    rclpy.spin_once(node, timeout_sec=0.2)
    wall, sim = time.monotonic() - t_wall0, node.sim_t - t_sim0
    resp = json.loads(fut.result().response) if fut.done() else {"results": []}
    tr = np.array(node.traj[i0:])
    # collisions: rising edges of body-rectangle contact with the fine world raster
    contact = np.array([footprint_contact(fine, forg, 0.05, (r[2], r[3], r[4])) for r in tr[:: 2]])
    collisions = int(np.sum(contact[1:] & ~contact[:-1]) + (1 if contact[:1].any() else 0))
    # split the trajectory by waypoint with the cumulative wall durations reported by the runner
    rows, res, t_edge = [], resp.get("results", []), t_wall0
    for i, (g, ideal) in enumerate(goals):
        t_next = t_edge + (res[i]["duration_s"] if i < len(res) else 0.0)
        j0 = int(np.searchsorted(tr[:, 1], t_edge)); j1 = max(j0, min(len(tr) - 1, int(np.searchsorted(tr[:, 1], t_next))))
        rows.append({"i": i, "goal_world": np.round(g, 2).tolist(), "astar_m": round(ideal, 2), "j0": j0, "j1": j1, "runner": res[i] if i < len(res) else None})
        t_edge = t_next
    # per waypoint path length / time for the reached ones; failed ones get the remainder up to the next reached waypoint start
    for i, r in enumerate(rows):
        seg = tr[r["j0"]: r["j1"] + 1]
        r["success"] = bool(np.linalg.norm(seg[-1, 2:4] - np.array(r["goal_world"])) < a.radius) if len(seg) else False
        r["travelled_m"] = round(float(np.linalg.norm(np.diff(seg[:, 2:4], axis=0), axis=1).sum()), 2) if len(seg) > 1 else 0.0
        r["sim_s"] = round(float(seg[-1, 0] - seg[0, 0]), 1) if len(seg) else 0.0
        r["wall_s"] = round(float(seg[-1, 1] - seg[0, 1]), 1) if len(seg) else 0.0
        r["efficiency"] = round(min(1.0, r["astar_m"] / max(r["travelled_m"], 1e-6)), 3) if r["success"] else None
    ok = [r for r in rows if r["success"]]
    summary = {"world": a.world, "waypoints": len(goals), "reached": len(ok), "success_rate": round(len(ok) / max(1, len(goals)), 3),
               "astar_total_m": round(sum(g[1] for g in goals), 1), "travelled_total_m": round(float(np.linalg.norm(np.diff(tr[:, 2:4], axis=0), axis=1).sum()), 1),
               "sim_s": round(sim, 1), "wall_s": round(wall, 1), "rtf": round(sim / max(wall, 1e-6), 2), "collisions": collisions,
               "mean_efficiency": round(float(np.mean([r["efficiency"] for r in ok])), 3) if ok else None}
    print(json.dumps(summary), flush=True)
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k not in ("j0", "j1")}), flush=True)
    np.save(os.path.join(a.out, "trajectory.npy"), tr)
    json.dump({"summary": summary, "waypoints": rows, "start": [x0, y0, yaw0], "plans": node.plans, "map_yaml": ypath},
              open(os.path.join(a.out, "result.json"), "w"), indent=1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
