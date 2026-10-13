"""Environment backed by the real ROS 2 graph (sim, replay-with-ground-truth or any stack that publishes /gt/*).

Runs an rclpy node in a background executor thread; the runner thread blocks on futures and the node clock.
Contract used (INTERFACES.md): /sim/set_robot_pose, /semantic_map/reset (+ optional mapping resets), /x30/ready,
/x30/navigate, /gt/pose, /gt/objects, /gt/scene_info, /x30/safety, /semantic_map/objects, TF map->base_link.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..core.episodes import Episode
from ..core.geometry import SE2, quat_from_yaw, yaw_from_quat
from ..core.infra import InfraError
from ..core.recording import Recording, TaskResult
from .base import Env


class RosEnv(Env):
    name = "ros"

    def __init__(self, cfg: dict, use_sim_time: bool = True, node_name: str = "x30_eval_runner"):
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.cfg = cfg
        self.node = rclpy.create_node(node_name, parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)])
        self.cb = ReentrantCallbackGroup()
        self.lock = threading.RLock()
        self._clients: dict[str, object] = {}
        tp = cfg["topics"]
        best = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Bool, String
        from x30_msgs.action import Navigate
        from x30_msgs.msg import SafetyStatus, SemanticObjectArray
        from x30_msgs.srv import Json
        from rclpy.action import ActionClient
        from tf2_ros import Buffer, TransformListener
        self._Json, self._Navigate = Json, Navigate
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.nav_client = ActionClient(self.node, Navigate, tp["navigate_action"], callback_group=self.cb)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel_nav", 10)

        def sub(msg, topic, fn, qos):
            return self.node.create_subscription(msg, topic, fn, qos, callback_group=self.cb)

        self._subs = [
            sub(PoseStamped, tp["gt_pose"], self._on_gt, best),
            sub(Bool, tp["ready"], self._on_ready, latched),
            sub(SafetyStatus, tp["safety"], self._on_safety, best),
            sub(SemanticObjectArray, tp["map_objects"], self._on_map, latched),
            sub(SemanticObjectArray, tp["gt_objects"], self._on_gt_objects, latched),
            sub(String, tp["gt_scene_info"], self._on_scene_info, latched),
            sub(Bool, tp["estop"], self._on_estop, QoSProfile(depth=10)),
            sub(String, tp["mode"], self._on_mode, latched),
            sub(Twist, tp["teleop"], self._on_teleop, QoSProfile(depth=10)),
            sub(Odometry, tp["odom_active"], self._on_odom, best),
        ]
        self._subs.append(sub(String, tp["contacts"], self._on_contact, QoSProfile(depth=50)))   # optional topic
        self._frames_enabled = bool(cfg["run"]["frames"]["enabled"])
        if self._frames_enabled:
            from sensor_msgs.msg import Image
            from x30_msgs.msg import Detection3DArray
            from sensor_msgs.msg import CameraInfo
            self._subs.append(sub(Image, tp["camera"], self._on_image, best))
            self._subs.append(sub(CameraInfo, tp["camera_info"], self._on_camera_info, latched))
            self._subs.append(sub(Detection3DArray, tp["detections"], self._on_detections, best))

        self.ex = MultiThreadedExecutor(num_threads=4)
        self.ex.add_node(self.node)
        self._spin = threading.Thread(target=self.ex.spin, daemon=True)
        self._spin.start()

        self._gt = None            # latest (wall_recv, clock_t, x, y, yaw)
        self._odom = None
        self._ready = None
        self._gt_objects: list[dict] = []
        self._map_objects: list[dict] = []
        self._scene_info: dict = {}
        self._mode = None
        self._recording = False
        self.rec = Recording()
        self.t0 = 0.0
        self.map_to_world = SE2()
        self._goal_handle = None
        self._result_future = None
        self._early_result: TaskResult | None = None
        self._notes: list[str] = []
        self._camera_info: dict | None = None
        self._frames_dir = ""
        self._last_teleop = -1e9
        self._last_frame_t = -1e9
        self._timer = self.node.create_timer(1.0 / cfg["run"]["record_hz"], self._on_timer, callback_group=self.cb)

    # ------------------------------------------------------------------ clock and waiting
    def _clock(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def now(self) -> float:
        return self._clock() - self.t0

    def advance(self, dt: float) -> None:
        start, wall0 = self._clock(), time.monotonic()
        while self._clock() - start < dt and time.monotonic() - wall0 < 5.0 * dt + 1.0:
            time.sleep(0.02)

    def _wait(self, predicate, timeout_wall: float, what: str, kind: str = "infrastructure") -> None:
        t0 = time.monotonic()
        while not predicate():
            if time.monotonic() - t0 > timeout_wall:
                raise InfraError(f"timeout waiting for {what} ({timeout_wall:.0f}s)", kind)
            time.sleep(0.05)

    # ------------------------------------------------------------------ callbacks
    def _on_gt(self, m) -> None:
        q = m.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        with self.lock:
            self._gt = (time.monotonic(), self._clock(), m.pose.position.x, m.pose.position.y, yaw, m.pose.position.z)
            if self._recording:
                self.rec.gt.append([self._clock() - self.t0, m.pose.position.x, m.pose.position.y, yaw])

    def _on_odom(self, m) -> None:
        q = m.pose.pose.orientation
        with self.lock:
            self._odom = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quat(q.x, q.y, q.z, q.w))

    def _on_ready(self, m) -> None:
        self._ready = bool(m.data)

    def _on_safety(self, m) -> None:
        if self._recording:
            with self.lock:
                self.rec.safety.append({"t": self.now(), "stopped": bool(m.stopped), "limiting": bool(m.limiting),
                                        "reason": m.reason, "min_dist": float(m.min_obstacle_distance)})

    @staticmethod
    def _objects(msg) -> list[dict]:
        out = []
        for o in msg.objects:
            out.append({"id": int(o.id), "label": o.label, "name": o.caption or f"{o.label}_{o.id}", "x": o.position.x,
                        "y": o.position.y, "z": o.position.z, "sx": o.size.x, "sy": o.size.y, "sz": o.size.z,
                        "confidence": float(o.confidence), "observations": int(o.observations),
                        "bearing_only": bool(o.bearing_only), "room": o.room})
        return out

    def _on_map(self, m) -> None:
        self._map_objects = self._objects(m)

    def _on_gt_objects(self, m) -> None:
        self._gt_objects = self._objects(m)

    def _on_scene_info(self, m) -> None:
        try:
            self._scene_info = json.loads(m.data)
        except json.JSONDecodeError:
            self._scene_info = {"raw": m.data}

    def _on_estop(self, m) -> None:
        if self._recording and m.data:
            self.rec.interventions.append({"t": self.now(), "kind": "estop"})

    def _on_mode(self, m) -> None:
        if self._recording and self._mode is not None and m.data != self._mode:
            self.rec.interventions.append({"t": self.now(), "kind": f"mode_change:{m.data}"})
        self._mode = m.data

    def _on_teleop(self, m) -> None:
        if self._recording and (abs(m.linear.x) + abs(m.linear.y) + abs(m.angular.z)) > 1e-3:
            t = self.now()
            if t - self._last_teleop > 1.0:                       # one intervention per teleop burst
                self.rec.interventions.append({"t": t, "kind": "teleop"})
            self._last_teleop = t

    def _on_contact(self, m) -> None:
        with self.lock:
            self.rec.contacts_available = True
            if self._recording:
                self.rec.contacts.append(self.now())

    def _on_image(self, m) -> None:
        fr = self.cfg["run"]["frames"]
        if not self._recording or self._clock() - self._last_frame_t < 1.0 / fr["hz"] or self._gt is None:
            return
        self._last_frame_t = self._clock()
        try:
            from PIL import Image as PILImage
            arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
            img = PILImage.fromarray(arr if m.encoding.startswith("rgb") else arr[..., ::-1])
            if img.width > fr["max_width"]:
                img = img.resize((fr["max_width"], int(img.height * fr["max_width"] / img.width)))
            d = Path(self._frames_dir)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{len(self.rec.frames):05d}.jpg"
            img.save(path, quality=fr["jpeg_quality"])
        except Exception as exc:                                  # a failed frame must never break an episode
            self.node.get_logger().warn(f"frame not saved: {exc}", throttle_duration_sec=30.0)
            return
        with self.lock:
            g = self._gt
            self.rec.frames.append({"t": self.now(), "path": str(path), "stamp": m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                                    "gt": [g[2], g[3], g[4]], "gt_z": g[5]})

    def _on_camera_info(self, m) -> None:
        self._camera_info = {"K": [m.k[0], m.k[4], m.k[2], m.k[5]], "width": int(m.width), "height": int(m.height)}

    def _camera_extrinsics(self) -> dict | None:
        """Camera description for the perception export: intrinsics plus the pose of the optical frame in base_link."""
        from rclpy.time import Time
        if self._camera_info is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(self.cfg["topics"]["base_frame"], self.cfg["topics"]["camera_frame"], Time())
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return {**self._camera_info, "base_to_cam": {"t": [t.x, t.y, t.z], "q": [q.x, q.y, q.z, q.w]}}

    def _on_detections(self, m) -> None:
        if not self._recording:
            return
        items = [{"label": d.label, "score": float(d.score), "box": list(d.box_xyxy), "point_count": int(d.point_count),
                  "bearing_only": bool(d.bearing_only)} for d in m.detections]
        self.rec.detections.append({"t": self.now(), "image_stamp": m.image_stamp.sec + 1e-9 * m.image_stamp.nanosec, "items": items})

    def _lookup_map_base(self) -> SE2 | None:
        from rclpy.time import Time
        try:
            tf = self.tf_buffer.lookup_transform(self.cfg["topics"]["slam_frame"], self.cfg["topics"]["base_frame"], Time())
        except Exception:
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        return SE2(t.x, t.y, yaw_from_quat(q.x, q.y, q.z, q.w))

    def _on_timer(self) -> None:
        if not self._recording or self._gt is None:
            return
        _, _, gx, gy, gyaw, _ = self._gt
        t = self.now()
        s = self._lookup_map_base()
        with self.lock:
            if s is not None:
                self.rec.pairs.append([t, s.x, s.y, s.yaw, gx, gy, gyaw])
            if self._odom is not None:
                self.rec.odom_pairs.append([t, *self._odom, gx, gy, gyaw])

    # ------------------------------------------------------------------ services
    def _call_json(self, name: str, request: dict, timeout: float = 15.0, wait: float = 3.0, optional: bool = False):
        cli = self._clients.get(name) or self.node.create_client(self._Json, name, callback_group=self.cb)
        self._clients[name] = cli
        if not cli.wait_for_service(timeout_sec=wait):
            if optional:
                return None
            raise InfraError(f"service {name} not available", "service")
        req = self._Json.Request()
        req.request = json.dumps(request)
        fut = cli.call_async(req)
        self._wait(fut.done, timeout, f"response of {name}", "service")
        resp = fut.result()
        try:
            body = json.loads(resp.response) if resp.response else {}
        except json.JSONDecodeError:
            body = {"raw": resp.response}
        return bool(resp.ok), body

    # ------------------------------------------------------------------ Env interface
    def check_health(self) -> None:
        cs = self.cfg["infra"]["containers"]
        if cs and shutil.which("docker"):
            for c in cs:
                out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", c], capture_output=True, text=True)
                if out.stdout.strip() != "true":
                    raise InfraError(f"container {c} is not running", "container")
        self._wait(lambda: self._gt is not None and time.monotonic() - self._gt[0] < 5.0, 15.0,
                   f"{self.cfg['topics']['gt_pose']} (sim not running?)", "sim")
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            raise InfraError(f"action server {self.cfg['topics']['navigate_action']} not available", "service")

    def reset_episode(self, ep: Episode, start: dict, seed: int, cfg: dict) -> None:
        rs = cfg["reset"]
        self._cancel_goal()
        if rs["strategy"] == "relaunch":
            if not rs["relaunch_command"]:
                raise InfraError("reset.strategy=relaunch needs reset.relaunch_command", "config")
            subprocess.run(rs["relaunch_command"], shell=True, check=False, timeout=600)
            self.advance(rs["relaunch_wait_s"])
        ok, body = self._call_json(rs["pose_service"], {"x": start["x"], "y": start["y"], "z": start.get("z", 0.0), "yaw": start["yaw"]},
                                   timeout=rs["pose_timeout_s"])
        if not ok or body.get("success") is False:
            raise InfraError(f"{rs['pose_service']} refused the reset: {body}", "sim")
        notes = []
        if rs["strategy"] == "services":
            for svc in rs["services"]:
                out = self._call_json(svc["name"], svc.get("request", {}), optional=svc.get("optional", False))
                if out is None:
                    notes.append(f"optional reset service {svc['name']} missing: mapping state NOT reset")
                elif not out[0]:
                    raise InfraError(f"{svc['name']} failed: {out[1]}", "service")
        self.advance(rs["settle_s"])
        self._wait(lambda: self._ready is True, cfg["run"]["ready_timeout_s"], f"{cfg['topics']['ready']} = true", "robot")
        self._wait(lambda: self._gt is not None and math.hypot(self._gt[2] - start["x"], self._gt[3] - start["y"]) < 0.75,
                   15.0, "the teleported GT pose", "sim")
        self.map_to_world = self._measure_map_to_world(notes)
        self._notes = notes
        self._map_objects = []

    def _measure_map_to_world(self, notes: list[str]) -> SE2:
        t0 = time.monotonic()
        s = self._lookup_map_base()
        while s is None and time.monotonic() - t0 < 10.0:
            time.sleep(0.2)
            s = self._lookup_map_base()
        if s is None:
            notes.append("TF map->base_link unavailable at start: map->world assumed identity")
            return SE2()
        _, _, gx, gy, gyaw, _ = self._gt
        return SE2(gx, gy, gyaw).compose(s.inverse())

    def begin(self, ep: Episode, mode: str, params: dict, trace_path: str | None) -> None:
        from geometry_msgs.msg import PoseStamped
        self._frames_dir = params.get("frames_dir", "")
        goal = self._Navigate.Goal()
        goal.mode, goal.instruction, goal.target_label = mode, ep.instruction, ep.target_label
        goal.episode_id, goal.timeout_s = params.get("trace_id", ep.id), float(ep.time_limit_s)   # unique per repeat: names the agent trace
        goal.params_json = json.dumps(params)
        gp = ep.goal
        if mode == "goal_pose":
            gx, gy = gp.get("x"), gp.get("y")
            if gx is None:
                raise ValueError(f"episode {ep.id}: goal_pose mode needs goal.x/goal.y (object goals only have a label)")
            local = self.map_to_world.inverse().compose(SE2(gx, gy, gp.get("yaw", 0.0)))
            pose = PoseStamped()
            pose.header.frame_id = self.cfg["topics"]["slam_frame"]
            pose.pose.position.x, pose.pose.position.y = local.x, local.y
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quat_from_yaw(local.yaw)
            goal.goal = pose
        self.rec = Recording(map_to_world=self.map_to_world.as_dict(), notes=list(self._notes))
        self.rec.gt_objects = [dict(o) for o in self._gt_objects]
        self.rec.scene_info = dict(self._scene_info)
        if self._frames_enabled:
            cam = self._camera_extrinsics()
            if cam:
                self.rec.scene_info["camera"] = cam
        self._early_result = None
        self._result_future = None
        self.t0 = self._clock()
        self._recording = True
        fut = self.nav_client.send_goal_async(goal, feedback_callback=self._on_feedback)
        try:
            self._wait(fut.done, 15.0, "goal acceptance", "service")
        except InfraError:
            self._recording = False
            raise
        self._goal_handle = fut.result()
        if not self._goal_handle.accepted:
            self._early_result = TaskResult(success=False, stop_reason="error", message="goal rejected by /x30/navigate")
            self._goal_handle = None
            return
        self._result_future = self._goal_handle.get_result_async()

    def _on_feedback(self, fb_msg) -> None:
        fb = fb_msg.feedback
        if self._recording:
            self.rec.feedback.append({"t": self.now(), "status": fb.status, "skill": fb.current_skill,
                                      "dist": float(fb.distance_to_goal), "turn": int(fb.agent_turn)})

    def task_result(self) -> TaskResult | None:
        if self._early_result is not None:
            return self._early_result
        if self._result_future is None or not self._result_future.done():
            return None
        wrapped = self._result_future.result()
        r = wrapped.result
        # action status 4 = SUCCEEDED; aborted/canceled goals still carry a result message
        return TaskResult(success=bool(r.success), stop_reason=r.stop_reason or "error", message=r.message,
                          path_length_m=float(r.path_length_m), duration_s=float(r.duration_s), agent_turns=int(r.agent_turns),
                          stopped_by_system=True)

    def _cancel_goal(self) -> None:
        gh, self._goal_handle = self._goal_handle, None
        if gh is not None:
            fut = gh.cancel_goal_async()
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < 3.0:
                time.sleep(0.05)
        from geometry_msgs.msg import Twist
        for _ in range(5):
            self.cmd_pub.publish(Twist())
            time.sleep(0.05)

    def cancel(self) -> None:
        self._cancel_goal()

    def finish_recording(self) -> Recording:
        self._recording = False
        time.sleep(1.2)                                           # let the 1 Hz map topic deliver its final state
        with self.lock:
            self.rec.map_objects = [dict(o) for o in self._map_objects]
            self.rec.gt_objects = [dict(o) for o in self._gt_objects] or self.rec.gt_objects
            if self._scene_info:
                self.rec.scene_info = {**self._scene_info, **({"camera": self.rec.scene_info["camera"]} if "camera" in self.rec.scene_info else {})}
        self._goal_handle = None
        return self.rec

    def recover(self, cfg: dict) -> None:
        if cfg["reset"]["strategy"] == "relaunch" and cfg["reset"]["relaunch_command"]:
            subprocess.run(cfg["reset"]["relaunch_command"], shell=True, check=False, timeout=600)
        self._wait(lambda: self._gt is not None and time.monotonic() - self._gt[0] < 5.0, 120.0, "ground-truth pose after recovery", "sim")

    def world_info(self) -> dict:
        return dict(self._scene_info)

    def close(self) -> None:
        try:
            self.ex.shutdown()
            self.node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass
