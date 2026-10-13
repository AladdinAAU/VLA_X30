"""One object_nav episode through /x30/navigate (runs inside the e2e-fw container; sim in the same ROS domain).

python3 /x30/scripts/e2e/run_episode.py --name ep1 --instruction "Find the sofa and stop next to it." --label sofa \
        --start 3.99 -1.51 0.0 [--timeout 900] [--reset]

--start x y yaw (world frame): teleports the robot with /sim/set_robot_pose, restarts known_map_nav (it aligns odom to world once),
resets the semantic map and the seen-set of next_viewpoint, then sends the Navigate goal. Records into results/e2e/<name>/:
trajectory.npy (t_sim, x, y, yaw, world), cam_*.jpg (camera at every agent turn change), feedback.json, result.json, semmap.json.
Success = distance(final /gt/pose, nearest GT instance of --label) <= 1.5 m and the agent ended with stop_reason finished.
"""
import argparse
import json
import math
import os
import subprocess
import time

import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from x30_msgs.action import Navigate
from x30_msgs.srv import Json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORLD_OBJECTS = "/x30/ros2_ws/src/x30_worlds/worlds/{world}/objects.json"


class Rec(Node):
    def __init__(self):
        super().__init__("e2e_recorder", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.pose, self.img, self.traj, self.fb = None, None, [], []
        self.create_subscription(PoseStamped, "/gt/pose", self.on_pose, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_img, 1)

    def on_pose(self, m):
        q = m.pose.orientation
        self.pose = (m.pose.position.x, m.pose.position.y, 2.0 * math.atan2(q.z, q.w))
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if not self.traj or t - self.traj[-1][0] > 0.2:
            self.traj.append((t, *self.pose))

    def on_img(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        self.img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else a

    def call(self, name, req, timeout=20.0):
        cli = self.create_client(Json, name)
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {name} not available")
        fut = cli.call_async(Json.Request(request=json.dumps(req)))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return json.loads(fut.result().response) if fut.result() else None

    def save_cam(self, out, tag):
        if self.img is not None:
            cv2.imwrite(os.path.join(out, f"cam_{tag}.jpg"), self.img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--start", nargs=3, type=float, default=None)
    ap.add_argument("--world", default="replica_frl_apartment_0")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--rooms", default=None, help="configs/rooms/<world>.yaml: named rooms (world-frame rectangles) set in the semantic map")
    ap.add_argument("--target-room", default=None, help="room task: success = final /gt/pose inside this room's rectangle (e.g. 'Room C')")
    ap.add_argument("--policy-name", default=None, help="agent policy yaml in configs/agent (e.g. locate_nav: the model only reports where the target is)")
    ap.add_argument("--load-map-raw", default=None, help="saved semantic map already in the map frame of this start pose (e.g. taught by scripts/teach/run_tour.py): loaded as is")
    ap.add_argument("--rooms-score-only", action="store_true", help="use --rooms/--target-room only for scoring (the semantic map keeps its own rooms/places)")
    ap.add_argument("--load-map", default=None, help="saved semantic map (from build_map.py); loaded instead of resetting (plan B)")
    a = ap.parse_args()
    out = f"/x30/results/e2e/{a.name}"
    os.makedirs(out, exist_ok=True)
    rclpy.init()
    n = Rec()
    for _ in range(50):
        rclpy.spin_once(n, timeout_sec=0.2)
        if n.pose and n.img is not None:
            break
    if a.start:
        x, y, yaw = a.start
        print("set_robot_pose", n.call("/sim/set_robot_pose", {"x": x, "y": y, "yaw": yaw}, 60.0))
        t0 = time.time()
        while time.time() - t0 < 4:
            rclpy.spin_once(n, timeout_sec=0.2)
        # restart known_map_nav so that T_world_odom is re-aligned
        sh("pkill -f '[k]nown_map_nav|[w]aypoint_runner|[n]ext_viewpoint'; sleep 3")
        subprocess.Popen(["bash", "-lc", "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80; "
                          "ros2 launch x30_navigation known_map_nav.launch.py world:=%s relay:=false > /x30/results/e2e/logs/nav_%s.log 2>&1" % (a.world, a.name)])
        time.sleep(12)
    if a.load_map_raw:
        n.call("/semantic_map/reset", {})
        print("load semantic map (raw):", n.call("/semantic_map/load", {"path": a.load_map_raw}), "viewpoints:", n.call("/nav/next_viewpoint/reset", {}))
    elif a.load_map:
        n.call("/semantic_map/reset", {})
        from mapframe import convert_file  # saved maps are in the world frame; known_map_nav's map frame is the start pose
        st = a.start if a.start else (n.pose[0], n.pose[1], n.pose[2])
        tmp = f"/x30/results/maps/_load_{a.name}.json"
        convert_file(a.load_map, tmp, st[:2], st[2], inverse=True)
        print("load semantic map:", n.call("/semantic_map/load", {"path": tmp}), "viewpoints:", n.call("/nav/next_viewpoint/reset", {}))
    else:
        print("reset semantic map:", n.call("/semantic_map/reset", {}), "viewpoints:", n.call("/nav/next_viewpoint/reset", {}))
    room_rect = None
    if a.rooms:
        import yaml
        from mapframe import _tf
        st = a.start if a.start else (n.pose[0], n.pose[1], n.pose[2])
        if abs(math.atan2(math.sin(st[2]), math.cos(st[2]))) > 0.05:
            raise SystemExit("--rooms needs a start yaw of 0 (map-frame rooms are axis-aligned rectangles)")
        defs = yaml.safe_load(open(a.rooms))["rooms"]
        mdefs = []
        for d in defs:  # world rectangle -> map frame (map = start pose); both corners rotated, axis-aligned hull (exact for yaw 0)
            x0, y0, x1, y1 = d["rect"]
            c = [_tf([x, y], st[:2], st[2], True) for x, y in ((x0, y0), (x1, y1))]
            mdefs.append({"name": d["name"], "rect": [min(c[0][0], c[1][0]), min(c[0][1], c[1][1]), max(c[0][0], c[1][0]), max(c[0][1], c[1][1])]})
            if a.target_room and d["name"] == a.target_room:
                room_rect = d["rect"]
        if not a.rooms_score_only:
            print("set_rooms:", n.call("/semantic_map/set_rooms", {"rooms": mdefs}))
            time.sleep(2)
    if a.target_room and room_rect is None:
        raise SystemExit("target room not found in --rooms file")
    cli = ActionClient(n, Navigate, "/x30/navigate")
    cli.wait_for_server(timeout_sec=30.0)
    start_pose = n.pose
    goal = Navigate.Goal(mode="object_nav", instruction=a.instruction, target_label=a.label, episode_id=a.name,
                         timeout_s=float(a.timeout), params_json=json.dumps({"trace_dir": out + "/traces", **({"policy_name": a.policy_name} if a.policy_name else {}),
                                                     **({"policy": {"guard": {"require_verify": False}}} if a.target_room and not a.policy_name else {})}))
    wall0, sim0 = time.time(), n.traj[-1][0] if n.traj else 0.0
    last_turn = [-1]

    def on_fb(msg):
        f = msg.feedback
        n.fb.append({"wall": time.time() - wall0, "sim": (n.traj[-1][0] - sim0) if n.traj else 0, "status": f.status,
                     "skill": f.current_skill, "dist": float(f.distance_to_goal), "turn": int(f.agent_turn)})
        if f.agent_turn != last_turn[0]:
            last_turn[0] = f.agent_turn
            n.save_cam(out, f"turn{f.agent_turn:02d}")

    n.save_cam(out, "start")
    fut = cli.send_goal_async(goal, feedback_callback=on_fb)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=30.0)
    gh = fut.result()
    print("goal accepted:", gh.accepted)
    rfut = gh.get_result_async()
    while not rfut.done():
        rclpy.spin_once(n, timeout_sec=0.2)
        if time.time() - wall0 > a.timeout * 3 + 60:
            gh.cancel_goal_async()
            break
    res = rfut.result().result if rfut.done() else None
    wall, sim = time.time() - wall0, n.traj[-1][0] - sim0
    n.save_cam(out, "end")
    if a.target_room:
        x0, y0, x1, y1 = room_rect
        px, py = n.pose[0], n.pose[1]
        dr = math.hypot(max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1))
        out_res = {"name": a.name, "instruction": a.instruction, "target_room": a.target_room, "start": start_pose, "final": n.pose,
                   "dist_to_room_m": dr, "inside": dr == 0.0, "stop_reason": res.stop_reason if res else "no_result",
                   "agent_success": bool(res.success) if res else False, "message": res.message if res else "",
                   "path_length_m": float(res.path_length_m) if res else None, "turns": int(res.agent_turns) if res else None,
                   "wall_s": wall, "sim_s": sim, "success_gt": bool(res and res.stop_reason == "finished" and dr == 0.0)}
        json.dump(out_res, open(out + "/result.json", "w"), indent=1)
        json.dump(n.fb, open(out + "/feedback.json", "w"), indent=1)
        np.save(out + "/trajectory.npy", np.array(n.traj))
        print(json.dumps(out_res, indent=1))
        rclpy.shutdown()
        return
    objs = json.load(open(WORLD_OBJECTS.format(world=a.world)))
    objs = objs["objects"] if isinstance(objs, dict) else objs
    inst = [(o["id"], o["pose"]["x"], o["pose"]["y"]) for o in objs if o["label"] == a.label]
    dists = [(math.hypot(n.pose[0] - x, n.pose[1] - y), i) for i, x, y in inst]
    dmin, iid = min(dists)
    out_res = {"name": a.name, "instruction": a.instruction, "label": a.label, "start": start_pose, "final": n.pose,
               "nearest_instance": iid, "dist_to_nearest_m": dmin, "radius_m": a.radius,
               "stop_reason": res.stop_reason if res else "no_result", "agent_success": bool(res.success) if res else False,
               "message": res.message if res else "", "path_length_m": float(res.path_length_m) if res else None,
               "turns": int(res.agent_turns) if res else None, "wall_s": wall, "sim_s": sim,
               "success_gt": bool(res and res.stop_reason == "finished" and dmin <= a.radius)}
    json.dump(out_res, open(out + "/result.json", "w"), indent=1)
    json.dump(n.fb, open(out + "/feedback.json", "w"), indent=1)
    np.save(out + "/trajectory.npy", np.array(n.traj))
    json.dump(n.call("/semantic_map/query", {"limit": 500}), open(out + "/semmap.json", "w"))
    sm = n.call("/semantic_map/render", {})
    json.dump(sm, open(out + "/semmap_render.json", "w"))
    print(json.dumps(out_res, indent=1))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
