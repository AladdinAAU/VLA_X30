#!/usr/bin/env python3
"""FAKE robot side for the ROS integration check of the agent nodes (run inside the x30-framework container).

One process that wraps tests/fake_world.py as ROS 2 interfaces with the names and JSON shapes of the real system:
  * action servers /x30/skill/<name> (x30_msgs/action/Skill) for every skill of configs/skills.yaml; motion skills are exclusive,
    `stop` and `verify` are not; `stop` cancels the running skill; every skill takes --skill-delay seconds and honours cancel.
  * services /semantic_map/{summary,query,find,relations} (x30_msgs/srv/Json) in the format of x30_semantic_map.
  * TF map->base_link, /camera/color/image_raw (rgb8), /semantic_map/annotated_map (transient local).
  * /fake/reset and /fake/state (Json) for the test client: new world + the mock model for the chosen mode; ground truth.
  * an OpenAI-compatible mock model server (rule-based mocks of tests/mock_models.py) on --port, so the agent nodes run without a GPU.
It exercises the plumbing only; it says nothing about the real model or robot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1]), str(HERE.parents[2] / "x30_common")]

import rclpy  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402
from x30_msgs.action import Skill  # noqa: E402
from x30_msgs.srv import Json  # noqa: E402

from fake_world import FakeWorld, ground_truth_success  # noqa: E402
from mock_models import FollowMock, ObjNavMock  # noqa: E402
from x30_common.testing.fake_openai import FakeOpenAIServer, text_reply  # noqa: E402

from x30_agent.core.skills import load_registry  # noqa: E402

FREE_SKILLS = ("stop", "verify")


def to_image_msg(img, stamp, frame: str) -> Image:
    import numpy as np
    a = np.asarray(img.convert("RGB"), dtype=np.uint8)
    m = Image()
    m.header.stamp, m.header.frame_id = stamp, frame
    m.height, m.width, m.encoding, m.step = a.shape[0], a.shape[1], "rgb8", a.shape[1] * 3
    m.data = a.tobytes()
    return m


class FakeStack(Node):
    def __init__(self, a):
        super().__init__("fake_stack")
        self.args = a
        self.lock = threading.RLock()
        self.world = FakeWorld(seed=a.seed)
        self.target = None
        self.mode = "object_nav"
        self.skill_delay_s = a.skill_delay
        self.model_delay_s = 0.0
        self.motion_busy = False
        self.skill_events: set = set()
        self.cb = ReentrantCallbackGroup()
        self.registry = load_registry(Path(a.configs) / "skills.yaml")
        self.servers = []
        for name in self.registry:
            self.servers.append(ActionServer(
                self, Skill, f"/x30/skill/{name}", execute_callback=lambda h, n=name: self._execute(n, h),
                goal_callback=lambda g, n=name: self._on_goal(n, g), cancel_callback=lambda _h: CancelResponse.ACCEPT,
                callback_group=self.cb))
        for srv_name, fn in (("summary", self._summary), ("query", self._query), ("find", self._find), ("relations", self._relations)):
            self.create_service(Json, f"/semantic_map/{srv_name}", self._json(fn), callback_group=self.cb)
        self.create_service(Json, "/fake/reset", self._json(self._reset), callback_group=self.cb)
        self.create_service(Json, "/fake/state", self._json(self._state), callback_group=self.cb)
        self.tf = TransformBroadcaster(self)
        self.cam_pub = self.create_publisher(Image, "/camera/color/image_raw", QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.map_pub = self.create_publisher(Image, "/semantic_map/annotated_map", QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(0.05, self._tf)
        self.create_timer(0.25, self._camera)
        self.create_timer(1.0, self._map)
        self.model = FakeOpenAIServer(self._model_handler, port=a.port).start()
        self._reset({"seed": a.seed, "mode": "object_nav"})
        self.get_logger().info(f"fake stack up: skills={list(self.registry)}, mock model {self.model.base_url}")

    # ---------------------------------------------------------------- publishers
    def _tf(self):
        with self.lock:
            p = self.world.pos
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = self.get_clock().now().to_msg(), "map", "base_link"
        t.transform.translation.x, t.transform.translation.y = float(p.x), float(p.y)
        t.transform.rotation.z, t.transform.rotation.w = math.sin(p.yaw / 2), math.cos(p.yaw / 2)
        self.tf.sendTransform(t)

    def _camera(self):
        with self.lock:
            img = self.world.camera_image()
        self.cam_pub.publish(to_image_msg(img, self.get_clock().now().to_msg(), "camera_link"))

    def _map(self):
        with self.lock:
            img = self.world.map_image()
        self.map_pub.publish(to_image_msg(img, self.get_clock().now().to_msg(), "map"))

    # ---------------------------------------------------------------- skills
    def _on_goal(self, name, goal):
        with self.lock:
            if name not in FREE_SKILLS and self.motion_busy:
                self.get_logger().info(f"rejecting {name}: a motion skill is running")
                return GoalResponse.REJECT
            if name not in FREE_SKILLS:
                self.motion_busy = True
        return GoalResponse.ACCEPT

    def _execute(self, name, handle):
        req = handle.request
        res = Skill.Result()
        ev = threading.Event()
        with self.lock:
            self.skill_events.add(ev)
        try:
            try:
                args = json.loads(req.args_json) if req.args_json else {}
                if not isinstance(args, dict):
                    raise ValueError("args must be an object")
            except ValueError as ex:
                res.success, res.message, res.data_json = False, f"invalid arguments: {ex}", "{}"
                handle.succeed()
                return res
            if name == "stop":
                with self.lock:
                    for e in list(self.skill_events):
                        e.set()
            t_end = time.time() + (self.skill_delay_s if name != "stop" else 0.0)
            while time.time() < t_end:
                if handle.is_cancel_requested or ev.is_set():
                    res.success, res.message, res.data_json = False, "cancelled", "{}"
                    handle.canceled()
                    return res
                time.sleep(0.05)
            with self.lock:
                r = self.world.run_skill(name, args, float(req.timeout_s or 30), ev if name != "stop" else threading.Event())
            res.success, res.message, res.data_json = bool(r.success), r.message, json.dumps(r.data)
            self._tf()                      # the world teleports; publish the new pose before the result (a real robot's TF is continuous)
            time.sleep(0.1)
            handle.succeed()
            return res
        finally:
            with self.lock:
                self.skill_events.discard(ev)
                if name not in FREE_SKILLS:
                    self.motion_busy = False

    # ---------------------------------------------------------------- semantic map services (format of x30_semantic_map)
    @staticmethod
    def _json(fn):
        def cb(req, resp):
            try:
                body = json.loads(req.request) if req.request else {}
                resp.response, resp.ok = json.dumps(fn(body)), True
            except Exception as ex:
                resp.response, resp.ok = json.dumps({"error": f"{type(ex).__name__}: {ex}"}), False
            return resp
        return cb

    def _desc(self, o):
        p = self.world.pos
        dx, dy = o.x - p.x, o.y - p.y
        return {"id": o.id, "label": o.label, "status": "confirmed", "confidence": 0.8, "position": [round(o.x, 3), round(o.y, 3), 0.4],
                "size": [o.size, o.size, 0.8], "caption": o.color, "attributes": [o.color], "room": o.room, "bearing_only": False,
                "range_m": round(math.hypot(dx, dy), 2),
                "bearing_deg": round(math.degrees((math.atan2(dy, dx) - p.yaw + math.pi) % (2 * math.pi) - math.pi), 1)}

    def _summary(self, b):
        with self.lock:
            return {"text": self.world.map_summary(None, int(b.get("max_objects") or 30))}

    def _query(self, b):
        with self.lock:
            lab = str(b.get("label") or "").lower()
            objs = [o for o in self.world.objects if o.seen and (not lab or lab in o.label)]
            objs.sort(key=lambda o: math.hypot(o.x - self.world.pos.x, o.y - self.world.pos.y))
            objs = objs[: int(b.get("limit", 10))]
            p = self.world.pos
            return {"count": len(objs), "objects": [self._desc(o) for o in objs], "robot_pose": [p.x, p.y, p.yaw]}

    def _find(self, b):
        text = str(b.get("text") or "").lower()
        with self.lock:
            hits = [o for o in self.world.objects if o.seen and o.label in text and (o.color in text or not any(c in text for c in ("red", "blue", "green", "white", "black")))]
            return {"text": "; ".join(self.world._obj_line(o) for o in hits[:5]) or "no match", "ranked": [self._desc(o) for o in hits[:5]], "vlm_calls": 0}

    def _relations(self, b):
        with self.lock:
            o = next((o for o in self.world.objects if o.id == int(b["id"]) and o.seen), None)
            if o is None:
                raise ValueError("unknown id")
            return {"id": o.id, "label": o.label, "position": [o.x, o.y, 0.4], "room": o.room, "relations": [], "between": [],
                    "text": f"{o.label} #{o.id} in the {o.room}"}

    # ---------------------------------------------------------------- test control
    def _reset(self, b):
        with self.lock:
            self.world = FakeWorld(seed=int(b.get("seed", 0)), noise=float(b.get("noise", 0.0)))
            self.mode = b.get("mode", "object_nav")
            self.skill_delay_s = float(b.get("skill_delay_s", self.args.skill_delay))
            self.model_delay_s = float(b.get("model_delay_s", 0.0))
            instr, self.target = self.world.make_task()
            if self.mode == "velocity_follow":
                instr = f"Walk to the {self.target.label} and stop next to it."
                self.model.handler = self._delayed(FollowMock(self.world, self.target, seed=int(b.get("seed", 0))))
            else:
                self.model.handler = self._delayed(ObjNavMock(seed=int(b.get("seed", 0)), **({} if b.get("mock_noise") else dict(p_malformed=0, p_text=0, p_nocall=0, p_repeat=0, p_prefinish=0))))
            self.model.replies.clear()
            self.motion_busy = False
            return {"instruction": instr, "target_label": self.target.label, "target": {"id": self.target.id, "x": self.target.x, "y": self.target.y},
                    "start": [self.world.pos.x, self.world.pos.y, self.world.pos.yaw]}

    def _delayed(self, inner):
        def h(req, idx):
            if self.model_delay_s:
                time.sleep(self.model_delay_s)
            return inner(req, idx)
        return h

    def _model_handler(self, req, idx):
        return text_reply("no scenario loaded")

    def _state(self, b):
        with self.lock:
            p = self.world.pos
            t = self.target
            return {"pose": [p.x, p.y, p.yaw], "dist_to_target": math.hypot(t.x - p.x, t.y - p.y), "success": ground_truth_success(self.world, t),
                    "skill_calls": [c[0] for c in self.world.skill_calls], "n_skill_calls": len(self.world.skill_calls)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="/x30/configs")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skill-delay", type=float, default=0.3)
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = FakeStack(a)
    ex = MultiThreadedExecutor(num_threads=16)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
