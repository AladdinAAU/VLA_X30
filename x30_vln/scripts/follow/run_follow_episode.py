"""One follow_person episode in the simulator (runs inside e2e-fw; sim + person_mover + follow_person + modes running, see docs/FOLLOW_PERSON.md).

python3 /x30/scripts/follow/run_follow_episode.py --name crowd_1 [--start 0.93 11.48 0.0] [--timeout 95] [--d-target 2.5] [--vmax 0.6]

Operator flow (like a user would do it): teleport the robot to --start (world frame; known_map_nav's map frame is the start pose, map = world - start), reset the
people, send the Navigate goal (mode follow_person, the node waits ARMED), then call the service /follow/lock while the target stands in front of the
robot (retried every second until a person is confirmed). After a successful lock the people start walking (/sim/person/start).
Scoring against ground truth (/gt/person = the target, /gt/people = everybody, if a crowd runs):
  tracked_frac        share of the walking time in which the node's reported target is within 1.0 m of the true target
  wrong_person_frac   share of the follow state in which the estimated target is nearer (within 1.2 m) to another person than to the target
  identity_switches   changes of the nearest person to the estimate while following
  ambiguous_s/events  time and number of times the node held still because the target was ambiguous (node metrics)
  dist_err_*, within_0p5_frac, min_sep_target_m, min_sep_any_other_m (closest any non-target person came to the robot, whoever moved) and min_sep_other_robot_closing_m (the same, only while the robot's own motion closes the distance: scripted walkers ignore the robot, so only this one can be the robot's fault), floor_violation_s (target closer than 2 m),
  lost_events, guard_stopped_s, est_range_bias_m, path lengths
  success = tracked_frac >= 0.8 and wrong_person_frac <= 0.05 and min_sep_target_m >= 1.8 and min_sep_other_robot_closing_m >= 0.5 (first version: min_sep_any_other_m, changed after run crowd_rec3, see docs) and end distance <= 6 m
Writes results/follow/<name>/{result.json, series.json}."""
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Empty, String
from x30_msgs.action import Navigate
from x30_msgs.msg import SafetyStatus
from x30_msgs.srv import Json


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class Rec(Node):
    def __init__(self):
        super().__init__("follow_recorder", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.robot, self.person, self.people, self.status, self.safety = [], [], [], [], []
        self.create_subscription(PoseStamped, "/gt/pose", lambda m: self.robot.append((stamp(m.header), m.pose.position.x, m.pose.position.y,
                                 2 * math.atan2(m.pose.orientation.z, m.pose.orientation.w))), 50)
        self.create_subscription(PoseStamped, "/gt/person", lambda m: self.person.append((stamp(m.header), m.pose.position.x, m.pose.position.y)), 50)
        self.create_subscription(String, "/gt/people", self.on_people, 50)
        self.create_subscription(String, "/follow/status", self.on_status, 50)
        self.create_subscription(SafetyStatus, "/x30/safety", lambda m: self.safety.append((stamp(m.header), bool(m.stopped), float(m.scale), m.reason)), 50)
        self.p_start = self.create_publisher(Empty, "/sim/person/start", 1)
        self.p_reset = self.create_publisher(Empty, "/sim/person/reset", 1)

    def on_people(self, m):
        try:
            self.people.append(json.loads(m.data))
        except ValueError:
            pass

    def on_status(self, m):
        try:
            self.status.append(json.loads(m.data))
        except ValueError:
            pass

    def call(self, name, req, timeout=30.0):
        cli = self.create_client(Json, name)
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {name} not available")
        fut = cli.call_async(Json.Request(request=json.dumps(req)))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return json.loads(fut.result().response) if fut.result() else None

    def spin_for(self, s):
        t0 = time.time()
        while time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", nargs=3, type=float, default=[0.93, 11.48, 0.0])
    ap.add_argument("--timeout", type=float, default=95.0)
    ap.add_argument("--d-target", type=float, default=2.5)
    ap.add_argument("--vmax", type=float, default=0.6, help="robot forward speed cap [m/s]; kinematic sim 0.6, real robot trial limit 0.4")
    ap.add_argument("--lock-wait", type=float, default=40.0, help="wall seconds to retry the lock command")
    ap.add_argument("--no-planner", action="store_true", help="planner_follow false: old behaviour (drive straight at the person, stop at the wall)")
    ap.add_argument("--instruction", default="Follow the person in front of you.")
    a = ap.parse_args()
    out = f"/x30/results/follow/{a.name}"
    os.makedirs(out, exist_ok=True)
    rclpy.init()
    n = Rec()
    n.spin_for(2.0)
    print("set_robot_pose", n.call("/sim/set_robot_pose", {"x": a.start[0], "y": a.start[1], "yaw": a.start[2]}, 60.0))
    n.p_reset.publish(Empty())
    n.spin_for(5.0)
    print("vocabulary", n.call("/perception/vocabulary", {"op": "set", "labels": ["person", "chair", "bench", "door"]}))
    cli = ActionClient(n, Navigate, "/x30/navigate")
    cli.wait_for_server(timeout_sec=30.0)
    goal = Navigate.Goal(mode="follow_person", instruction=a.instruction, episode_id=a.name, timeout_s=float(a.timeout),
                         params_json=json.dumps({"d_target": a.d_target, "vmax": a.vmax, **({"planner_follow": False} if a.no_planner else {})}))
    n.robot.clear(), n.person.clear(), n.people.clear(), n.status.clear(), n.safety.clear()
    wall0 = time.time()
    fut = cli.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=30.0)
    gh = fut.result()
    print("goal accepted:", gh.accepted)
    rfut = gh.get_result_async()
    # ---- operator: lock on the person in front
    lock, started_t = None, None
    while lock is None and time.time() - wall0 < a.lock_wait and not rfut.done():
        n.spin_for(1.0)
        r = n.call("/follow/lock", {})
        if r and r.get("ok"):
            lock = r
    print("lock:", lock)
    if lock is not None:
        n.p_start.publish(Empty())
        n.spin_for(0.5)
        started_t = n.status[-1]["t"] if n.status else (n.robot[-1][0] if n.robot else 0.0)
        print("people started at sim t=%.1f" % started_t)
    while not rfut.done():
        n.spin_for(0.2)
        if time.time() - wall0 > a.timeout * 4 + 120:
            gh.cancel_goal_async()
            break
    res = rfut.result().result if rfut.done() else None
    n.spin_for(1.0)

    # ---- scoring against ground truth (world frame; map = world - start for yaw 0)
    R, Pp = np.array(n.robot), np.array(n.person)
    ox, oy = a.start[0], a.start[1]
    summary = {"name": a.name, "d_target": a.d_target, "start": a.start, "timeout_s": a.timeout, "lock": lock,
               "stop_reason": res.stop_reason if res else "no_result", "node_message": res.message if res else "", "wall_s": time.time() - wall0}
    walk = Pp[Pp[:, 0] >= started_t] if (len(Pp) and started_t is not None) else Pp[:0]
    if len(R) > 2 and len(walk) > 2:
        rx, ry = np.interp(walk[:, 0], R[:, 0], R[:, 1]), np.interp(walk[:, 0], R[:, 0], R[:, 2])
        ryaw = np.interp(walk[:, 0], R[:, 0], np.unwrap(R[:, 3]))
        dist = np.hypot(walk[:, 1] - rx, walk[:, 2] - ry)
        bearing = (np.degrees(np.arctan2(walk[:, 2] - ry, walk[:, 1] - rx) - ryaw) + 180) % 360 - 180
        st = [s for s in n.status if s.get("state") == "follow" and "x" in s]
        tracked, bias = np.zeros(len(walk), bool), []
        if st:
            sx = np.array([[s["t"], s["x"] + ox, s["y"] + oy] for s in st])
            for i, (t, px, py) in enumerate(walk):
                k = int(np.argmin(np.abs(sx[:, 0] - t)))
                tracked[i] = abs(sx[k, 0] - t) < 1.0 and math.hypot(sx[k, 1] - px, sx[k, 2] - py) < 1.0
                if tracked[i]:
                    bias.append(math.hypot(sx[k, 1] - rx[i], sx[k, 2] - ry[i]) - dist[i])
        dt = float(np.median(np.diff(walk[:, 0]))) if len(walk) > 2 else 0.05
        wrong, switches, last_name, min_any, min_closing = 0, 0, None, math.inf, math.inf          # identity: nearest person (any) to the estimate while following
        if n.people and st:
            pt = np.array([p["t"] for p in n.people])
            for s in st:
                if s["t"] < walk[0, 0]:
                    continue
                k = int(np.argmin(np.abs(pt - s["t"])))
                people = n.people[k]["people"]
                names = list(people)
                d = [math.hypot(people[nm][0] - (s["x"] + ox), people[nm][1] - (s["y"] + oy)) for nm in names]
                j = int(np.argmin(d))
                if d[j] > 1.2:
                    continue
                if names[j] != "target":
                    wrong += 1
                if last_name is not None and names[j] != last_name:
                    switches += 1
                last_name = names[j]
            for p in n.people:
                if p["t"] < walk[0, 0]:
                    continue
                kk = int(np.argmin(np.abs(R[:, 0] - p["t"])))
                k0, k1 = max(0, kk - 5), min(len(R) - 1, kk + 5)
                dtv = R[k1, 0] - R[k0, 0]
                vr = ((R[k1, 1] - R[k0, 1]) / dtv, (R[k1, 2] - R[k0, 2]) / dtv) if dtv > 1e-3 else (0.0, 0.0)
                for nm, v in p["people"].items():
                    if nm == "target":
                        continue
                    dx, dy = v[0] - R[kk, 1], v[1] - R[kk, 2]
                    d = math.hypot(dx, dy)
                    min_any = min(min_any, d)
                    if (vr[0] * dx + vr[1] * dy) / max(d, 1e-6) > 0.1:        # the robot's own motion is closing the distance
                        min_closing = min(min_closing, d)
        n_follow = sum(1 for s in st if s["t"] >= walk[0, 0])
        lost = sum(1 for a_, b_ in zip(n.status[:-1], n.status[1:]) if a_.get("state") != "lost" and b_.get("state") == "lost")
        g_s = 0.0
        stopped = [(s[0], s[1]) for s in n.safety]
        for (t0_, s0), (t1_, _) in zip(stopped[:-1], stopped[1:]):
            if s0 and walk[0, 0] <= t0_ <= walk[-1, 0]:
                g_s += t1_ - t0_
        err = np.abs(dist - a.d_target)
        pl = lambda xs, ys: float(np.hypot(np.diff(xs), np.diff(ys)).sum())
        mnode = {}
        try:
            mnode = json.loads(res.message[res.message.index("{"):]) if res and "{" in res.message else {}
        except ValueError:
            pass
        summary.update({
            "walk_duration_s": float(walk[-1, 0] - walk[0, 0]), "tracked_frac": float(tracked.mean()),
            "wrong_person_frac": (wrong / n_follow) if (n_follow and n.people) else None, "identity_switches": switches if n.people else None,
            "ambiguous_s": mnode.get("ambiguous_s"), "ambiguous_events": mnode.get("ambiguous_events"),
            "dist_max_m": float(dist.max()), "frac_dist_gt_3p5": float((dist > 3.5).mean()), "frac_dist_gt_5": float((dist > 5.0).mean()),
            "dist_mean_m": float(dist.mean()), "dist_err_mean_m": float(err.mean()), "dist_err_median_m": float(np.median(err)),
            "within_0p5_frac": float((err <= 0.5).mean()), "min_sep_target_m": float(dist.min()), "min_sep_any_other_m": min_any if math.isfinite(min_any) else None, "min_sep_other_robot_closing_m": min_closing if math.isfinite(min_closing) else None,
            "floor_violation_s": float((dist < 2.0).sum() * dt), "end_dist_m": float(dist[-1]), "bearing_abs_mean_deg": float(np.abs(bearing).mean()),
            "est_range_bias_m": float(np.mean(bias)) if bias else None, "lost_events_status": lost, "guard_stopped_s": g_s,
            "person_path_m": pl(walk[:, 1], walk[:, 2]), "robot_path_m": pl(R[:, 1], R[:, 2])})
        summary["success"] = bool(summary["tracked_frac"] >= 0.8 and (summary["wrong_person_frac"] or 0.0) <= 0.05 and summary["min_sep_target_m"] >= 1.8
                                  and (summary["min_sep_other_robot_closing_m"] is None or summary["min_sep_other_robot_closing_m"] >= 0.5) and summary["end_dist_m"] <= 6.0)
        json.dump({"t": walk[:, 0].tolist(), "dist": dist.tolist(), "bearing_deg": bearing.tolist(), "tracked": tracked.tolist(),
                   "person": walk[:, 1:].tolist(), "robot": np.c_[rx, ry, ryaw].tolist()}, open(out + "/series.json", "w"))
    else:
        summary["success"] = False
        summary["error"] = "no ground-truth walking samples" + ("" if lock else " (lock command never succeeded)")
    json.dump(summary, open(out + "/result.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
