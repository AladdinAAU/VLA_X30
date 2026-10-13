"""goal_benchmark: send N random NavigateToPose goals and score them against ground truth.

  ros2 run x30_navigation goal_benchmark --world rooms --seed 0 --goals 20 --out result.json [--explore-first]

With --gt-map <map.yaml> (map_server yaml of a simulator world, e.g. x30_worlds/worlds/lab_01/map.yaml) the ground-truth raster
comes from that map instead of a stub world, and collisions are counted from the ground-truth pose (clearance to an occupied
cell below --contact-dist; one count per approach).
Ground truth comes from /gt/pose (frame `world`) and from a ground-truth occupancy raster of the world (the nav_stub_robot
presets of tools/stub_world.py). Goals are sampled in the world frame at least `--min-dist` of path distance from the
start pose of each run; success = GT distance to the goal below `--radius` when Nav2 reports success. Path efficiency =
shortest GT path (inflated raster) / travelled GT path. Collisions come from /stub/collisions (stub) when present.
The map frame is assumed to start at the robot's first GT pose (what every odometry source here does).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy import ndimage
import json as _json
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from ..frontiers import _graph
from ..geometry import matrix_to_rpy, pose_msg_to_matrix
from .stub_world import make_world
from scipy.sparse.csgraph import dijkstra


class Bench(Node):
    def __init__(self):
        super().__init__("goal_benchmark")
        self.gt: np.ndarray | None = None
        self.collisions = 0
        self.clearance = None            # (distance to occupied [m], origin, resolution, threshold): GT-based collision count
        self.in_contact = False
        self.create_subscription(PoseStamped, "/gt/pose", self.on_gt, 10)
        self.create_subscription(Int32, "/stub/collisions", self.on_col, 10)
        self.map: OccupancyGrid | None = None
        self.status: dict = {}
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map", lambda m: setattr(self, "map", m), qos)
        self.create_subscription(String, "/nav/explore_status", lambda m: setattr(self, "status", _json.loads(m.data)), 5)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.tf_buffer = Buffer()
        self.tfl = TransformListener(self.tf_buffer, self)
        self.path_len = 0.0
        self.last_xy: np.ndarray | None = None

    def on_gt(self, msg: PoseStamped) -> None:
        m = pose_msg_to_matrix(msg.pose)
        self.gt = m
        xy = m[:2, 3]
        if self.clearance is not None:
            dist, org, r, thr = self.clearance
            ix, iy = ((xy - org) / r).astype(int)
            near = 0 <= iy < dist.shape[0] and 0 <= ix < dist.shape[1] and dist[iy, ix] < thr
            if near and not self.in_contact:
                self.collisions += 1
            self.in_contact = bool(near)
        if self.last_xy is not None:
            self.path_len += float(np.linalg.norm(xy - self.last_xy))
        self.last_xy = xy.copy()

    def on_col(self, msg: Int32) -> None:
        self.collisions = msg.data

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def load_gt_map(path: str, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Blocked raster (True = occupied, row 0 = lowest y) and lower-left origin of a map_server yaml, resampled to `res`."""
    import os
    import yaml
    from PIL import Image
    meta = yaml.safe_load(open(path))
    img = np.array(Image.open(os.path.join(os.path.dirname(path), meta["image"])).convert("L"), dtype=float) / 255.0
    occ = img if meta.get("negate", 0) else 1.0 - img
    blocked = occ[::-1] >= meta.get("occupied_thresh", 0.65)
    k = max(1, int(round(res / meta["resolution"])))
    h, w = (blocked.shape[0] // k) * k, (blocked.shape[1] // k) * k
    return blocked[:h, :w].reshape(h // k, k, w // k, k).any(axis=(1, 3)), np.array(meta["origin"][:2], dtype=float)


def geodesic_field(img_free: np.ndarray, res: float, src_xy, origin) -> tuple[np.ndarray, np.ndarray]:
    mat, idx = _graph(img_free)
    c = np.floor((np.asarray(src_xy) - origin) / res).astype(int)
    d = dijkstra(mat, directed=False, indices=int(idx[c[1], c[0]])) * res
    return d, idx


def coverage(node: "Bench", free: np.ndarray, res: float, origin, start: np.ndarray) -> float:
    """Fraction of ground-truth traversable cells (world raster) that /map marks as free."""
    m = node.map
    if m is None:
        return 0.0
    yaw0 = matrix_to_rpy(start[:3, :3])[2]
    c, s = math.cos(yaw0), math.sin(yaw0)
    ys, xs = np.nonzero(free)
    wx, wy = xs * res + origin[0] - start[0, 3], ys * res + origin[1] - start[1, 3]
    mx, my = c * wx + s * wy, -s * wx + c * wy       # world -> map frame
    r = m.info.resolution
    ix, iy = ((mx - m.info.origin.position.x) / r).astype(int), ((my - m.info.origin.position.y) / r).astype(int)
    ok = (ix >= 0) & (ix < m.info.width) & (iy >= 0) & (iy < m.info.height)
    data = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
    known = np.zeros(len(xs), bool)
    known[ok] = data[iy[ok], ix[ok]] == 0
    return float(known.mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore-first", action="store_true", help="run the autonomous frontier loop to completion before the goals")
    ap.add_argument("--explore-timeout", type=float, default=900.0)
    ap.add_argument("--explore-log", default="")
    ap.add_argument("--world", default="rooms")
    ap.add_argument("--gt-map", default="", help="map_server yaml of the simulator world (replaces the stub world raster)")
    ap.add_argument("--contact-dist", type=float, default=0.15, help="GT clearance counted as a collision (--gt-map only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goals", type=int, default=20)
    ap.add_argument("--min-dist", type=float, default=3.0)
    ap.add_argument("--goal-clearance", type=float, default=0.6, help="goals keep this distance from obstacles (circumscribed radius 0.56 plus margin: the final yaw turn must not touch a wall)")
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--robot-radius", type=float, default=0.3, help="inflation of the GT raster for shortest paths and goal sampling")
    ap.add_argument("--out", default="goal_benchmark.json")
    args = ap.parse_args(argv)

    res = 0.1
    if args.gt_map:
        blocked, origin = load_gt_map(args.gt_map, res)
    else:
        blocked, origin = make_world(args.world, args.seed).raster(res)
    dist = ndimage.distance_transform_edt(~blocked) * res
    free = dist > args.robot_radius
    rng = np.random.default_rng(args.seed + 100)

    rclpy.init()
    node = Bench()
    while node.gt is None:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not node.nav.wait_for_server(timeout_sec=120.0):
        raise SystemExit("navigate_to_pose action server not available")
    start = node.gt.copy()                      # map frame = robot pose at start
    if args.gt_map:
        node.clearance = (dist, origin, res, args.contact_dist)
    d0, idx0 = geodesic_field(free, res, start[:2, 3], origin)
    free &= np.isfinite(d0[idx0.clip(0)]) & (idx0 >= 0)      # only what the robot can reach counts for coverage and goals
    yaw0 = matrix_to_rpy(start[:3, :3])[2]
    c0, s0 = math.cos(yaw0), math.sin(yaw0)

    def world_to_map(xy, yaw):
        d = np.asarray(xy) - start[:2, 3]
        return (c0 * d[0] + s0 * d[1], -s0 * d[0] + c0 * d[1], yaw - yaw0)

    explore_log = []
    if args.explore_first:
        cli = node.create_client(Trigger, "/nav/explore/start")
        cli.wait_for_service(timeout_sec=30.0)
        rclpy.spin_until_future_complete(node, cli.call_async(Trigger.Request()), timeout_sec=10.0)
        t_exp = time.monotonic()
        last = 0.0
        while time.monotonic() - t_exp < args.explore_timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.monotonic() - last >= 10.0:
                last = time.monotonic()
                cov = coverage(node, free, res, origin, start)
                st = node.status
                explore_log.append({"t_s": round(last - t_exp, 1), "gt_coverage": round(cov, 4), "estimated_fraction": st.get("explored_fraction"),
                                    "frontiers": st.get("frontier_count"), "goals_sent": st.get("sent"), "failed": st.get("failed"),
                                    "path_m": round(node.path_len, 1), "state": st.get("state")})
                print(json.dumps(explore_log[-1]), flush=True)
                if st.get("state") == "complete":
                    break
        node.path_len = 0.0
        if args.explore_log:
            with open(args.explore_log, "w") as f:
                json.dump(explore_log, f, indent=2)

    results = []
    for i in range(args.goals):
        node.spin_for(0.5)
        here = node.gt[:2, 3].copy()
        d, idx = geodesic_field(free, res, here, origin)
        ys, xs = np.nonzero(free)
        dd = d[idx[ys, xs]]
        ok = np.isfinite(dd) & (dd > args.min_dist) & (dd < 14.0) & (dist[ys, xs] > args.goal_clearance)
        if not ok.any():
            break
        k = rng.choice(np.nonzero(ok)[0])
        goal_w = np.array([xs[k] * res + origin[0], ys[k] * res + origin[1]])
        shortest = float(dd[k])
        gx, gy, gyaw = world_to_map(goal_w, rng.uniform(-math.pi, math.pi))
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = gx, gy
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(gyaw / 2), math.cos(gyaw / 2)
        node.path_len = 0.0
        col0 = node.collisions
        t0 = time.monotonic()
        fut = node.nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
        handle = fut.result()
        status = "rejected"
        if handle is not None and handle.accepted:
            res_fut = handle.get_result_async()
            while not res_fut.done() and time.monotonic() - t0 < args.timeout:
                rclpy.spin_once(node, timeout_sec=0.1)
            if res_fut.done():
                status = {4: "succeeded", 5: "canceled", 6: "aborted"}.get(res_fut.result().status, str(res_fut.result().status))
            else:
                handle.cancel_goal_async()
                node.spin_for(1.0)
                status = "timeout"
        dur = time.monotonic() - t0
        final = float(np.linalg.norm(node.gt[:2, 3] - goal_w))
        success = status == "succeeded" and final < args.radius
        eff = shortest / max(node.path_len, shortest) if success else 0.0
        rec = {"i": i, "status": status, "success": bool(success), "final_dist_m": round(final, 3), "shortest_m": round(shortest, 2),
               "path_m": round(node.path_len, 2), "efficiency": round(eff, 3), "duration_s": round(dur, 1),
               "collisions": node.collisions - col0, "from": here.round(2).tolist(), "goal": goal_w.round(2).tolist()}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    n = len(results)
    succ = [r for r in results if r["success"]]
    summary = {"goals": n, "success_rate": len(succ) / max(n, 1), "mean_efficiency_successful": float(np.mean([r["efficiency"] for r in succ])) if succ else 0.0,
               "mean_duration_s": float(np.mean([r["duration_s"] for r in results])) if results else 0.0,
               "total_collisions": int(sum(r["collisions"] for r in results)),
               "world": args.world, "seed": args.seed}
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "goals": results}, f, indent=2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
