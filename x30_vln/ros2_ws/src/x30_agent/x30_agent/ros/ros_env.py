"""AgentEnv on ROS 2: skills via /x30/skill/<name>, the semantic map via its Json services and topics, pose via TF.

All blocking calls wait on threading.Event objects set from future callbacks, so they are safe in the worker thread of a
MultiThreadedExecutor and never spin the executor themselves."""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from x30_msgs.action import Skill
from x30_msgs.srv import Json

from x30_common.images import array_from_image_msg

from ..core.types import AgentEnv, Pose2D, SkillResult


def quat_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class RosEnv(AgentEnv):
    def __init__(self, node: Node, map_frame: str = "map", base_frame: str = "base_link", camera_topic: str = "/camera/color/image_raw",
                 annotated_topic: str = "/semantic_map/annotated_map", frame_buffer: int = 40):
        self.node = node
        self.map_frame, self.base_frame = map_frame, base_frame
        self.cb = ReentrantCallbackGroup()
        self._skill_clients: dict[str, ActionClient] = {}
        self._json_clients: dict[str, object] = {}
        self._frames: deque = deque(maxlen=frame_buffer)       # (stamp_s, ndarray)
        self._map_img = None
        self._lock = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(Image, camera_topic, self._on_camera, qos, callback_group=self.cb)
        node.create_subscription(Image, annotated_topic, self._on_map, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL), callback_group=self.cb)
        self._feedback_cb = None
        self._last_frame_t = 0.0

    # ---------------------------------------------------------------- topics
    def _on_camera(self, msg: Image) -> None:
        now = time.time()
        if now - self._last_frame_t < 0.5:          # keep ~2 frames/s, enough for history sampling
            return
        self._last_frame_t = now
        try:
            arr = array_from_image_msg(msg)
        except ValueError as ex:
            self.node.get_logger().warn(f"camera frame skipped: {ex}", throttle_duration_sec=30.0)
            return
        with self._lock:
            self._frames.append((now, arr))

    def _on_map(self, msg: Image) -> None:
        try:
            self._map_img = array_from_image_msg(msg)
        except ValueError:
            pass

    def camera_image(self):
        with self._lock:
            return self._frames[-1][1] if self._frames else None

    def recent_frames(self, n: int, spacing_s: float = 1.0) -> list:
        with self._lock:
            fr = list(self._frames)
        out = []
        for t, a in reversed(fr):
            if not out or out[-1][0] - t >= spacing_s:
                out.append((t, a))
            if len(out) >= n:
                break
        return [a for _, a in reversed(out)]

    def map_image(self):
        return self._map_img

    # ---------------------------------------------------------------- state
    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def pose(self) -> Pose2D | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return Pose2D(t.x, t.y, quat_yaw(q))

    def set_feedback(self, fn) -> None:
        self._feedback_cb = fn

    def feedback(self, status: str, skill: str = "", distance: float = float("nan"), turn: int = 0) -> None:
        if self._feedback_cb:
            try:
                self._feedback_cb(status, skill, distance, turn)
            except Exception:
                pass

    # ---------------------------------------------------------------- semantic map
    def _json_call(self, service: str, payload: dict, timeout_s: float = 8.0) -> dict:
        cli = self._json_clients.get(service)
        if cli is None:
            cli = self.node.create_client(Json, service, callback_group=self.cb)
            self._json_clients[service] = cli
        if not cli.wait_for_service(timeout_sec=1.5):
            return {"ok": False, "error": f"service {service} not available"}
        ev = threading.Event()
        fut = cli.call_async(Json.Request(request=json.dumps(payload)))
        fut.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout_s):
            return {"ok": False, "error": f"timeout calling {service}"}
        try:
            r = fut.result()
            body = json.loads(r.response) if r.response else {}
        except Exception as ex:
            return {"ok": False, "error": f"bad reply from {service}: {ex}"}
        if not isinstance(body, dict):
            body = {"value": body}
        body.setdefault("ok", bool(r.ok))
        return body

    def map_summary(self, pose, max_objects: int = 30) -> str:
        req = {"max_objects": max_objects}
        if pose is not None:
            req["pose"] = [pose.x, pose.y, pose.yaw]
        r = self._json_call("/semantic_map/summary", req)
        if not r.get("ok", True) and r.get("error"):
            return f"(map summary unavailable: {r['error']})"
        return str(r.get("text") or r.get("summary") or r.get("value") or "")

    def object_info(self, object_id: int):
        r = self._json_call("/semantic_map/query", {"limit": 500}, 5.0)    # query has no id filter
        for o in r.get("objects", []) or []:
            if int(o.get("id", -1)) == int(object_id):
                pos, size = o.get("position") or [0, 0, 0], o.get("size") or [0, 0, 0]
                return {"id": o["id"], "label": o.get("label", ""), "x": float(pos[0]), "y": float(pos[1]), "size": [float(size[0]), float(size[1])],
                        "bearing_only": bool(o.get("bearing_only", False))}
        return None

    def query_map(self, op: str, **kw) -> dict:
        if op == "list":
            r = self._json_call("/semantic_map/query", {"label": kw.get("label", ""), "limit": 10})
        elif op in ("get", "relations"):
            r = self._json_call("/semantic_map/relations", {"id": int(kw.get("id", -1))})
        elif op == "describe":
            r = self._json_call("/semantic_map/find", {"text": kw.get("text", "")}, 20.0)
        else:
            return {"ok": False, "error": f"unknown op {op}"}
        if r.get("ok", True) and "text" not in r:
            objs = r.get("objects")
            if isinstance(objs, list):
                r["text"] = "; ".join(f"#{o.get('id')} {o.get('label')} at ({o.get('position', [0, 0])[0]:.1f}, {o.get('position', [0, 0])[1]:.1f})" for o in objs[:8]) or "no match"
                r.pop("objects", None)
        return r

    # ---------------------------------------------------------------- skills
    def _client(self, name: str) -> ActionClient:
        c = self._skill_clients.get(name)
        if c is None:
            c = ActionClient(self.node, Skill, f"/x30/skill/{name}", callback_group=self.cb)
            self._skill_clients[name] = c
        return c

    def run_skill(self, name: str, args: dict, timeout_s: float, cancel: threading.Event) -> SkillResult:
        t0 = time.time()
        cli = self._client(name)
        if not cli.wait_for_server(timeout_sec=3.0):
            return SkillResult(False, f"skill server for '{name}' is not available", error="no_server")
        goal = Skill.Goal()
        goal.name, goal.args_json, goal.timeout_s = name, json.dumps(args), float(timeout_s)
        sent = threading.Event()
        sfut = cli.send_goal_async(goal)
        sfut.add_done_callback(lambda _f: sent.set())
        while not sent.wait(0.1):
            if cancel.is_set() or time.time() - t0 > 15:
                return SkillResult(False, "cancelled" if cancel.is_set() else "skill goal was not accepted in time", cancelled=cancel.is_set(), error="no_accept")
        handle = sfut.result()
        if not handle.accepted:
            return SkillResult(False, f"skill '{name}' was rejected (another motion skill is running?)", error="rejected")
        done = threading.Event()
        rfut = handle.get_result_async()
        rfut.add_done_callback(lambda _f: done.set())
        cancelled = False
        while not done.wait(0.1):
            if cancel.is_set() and not cancelled:
                cancelled = True
                handle.cancel_goal_async()
                if name != "stop":
                    self._send_stop()
            if cancelled and time.time() - t0 > timeout_s + 5:
                break
            if time.time() - t0 > timeout_s + 10:
                handle.cancel_goal_async()
                return SkillResult(False, f"{name} did not finish within {timeout_s:.0f} s", error="timeout", duration_s=time.time() - t0)
        if not done.is_set():
            return SkillResult(False, "cancelled", cancelled=True, duration_s=time.time() - t0)
        try:
            res = rfut.result().result
            data = json.loads(res.data_json) if res.data_json else {}
            if not isinstance(data, dict):
                data = {"value": data}
            return SkillResult(bool(res.success), res.message, data, time.time() - t0, cancelled=cancelled)
        except Exception as ex:
            return SkillResult(False, f"bad skill result: {ex}", error="bad_result", cancelled=cancelled)

    def _send_stop(self) -> None:
        cli = self._client("stop")
        if cli.wait_for_server(timeout_sec=0.5):
            cli.send_goal_async(Skill.Goal(name="stop", args_json="{}", timeout_s=5.0))
