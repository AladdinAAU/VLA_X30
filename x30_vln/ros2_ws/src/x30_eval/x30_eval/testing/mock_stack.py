"""MOCK of the robot-facing graph, for testing RosEnv and the runner node without Gazebo. Not a simulator.

Provides the contract topics/services/actions the evaluator uses: /sim/set_robot_pose, /semantic_map/reset, /x30/ready,
/gt/pose, /gt/objects, /gt/scene_info, /semantic_map/objects, /x30/safety, TF map->base_link and the /x30/navigate action.
The "map" frame is deliberately rotated/shifted against the world (its origin is the pose at the last reset) so the
evaluator's frame alignment is exercised. Behaviour per goal: goal_pose drives straight to the pose; object_nav drives to
``standoff`` metres from the nearest object of the label; label "hang" never finishes; label "boom" fails with a connection
error (an infrastructure failure for the runner). Run:  ros2 run x30_eval x30_eval_mock_stack
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster
from x30_msgs.action import Navigate
from x30_msgs.msg import SafetyStatus, SemanticObject, SemanticObjectArray
from x30_msgs.srv import Json

from ..core.geometry import SE2, quat_from_yaw

OBJECTS = [("chair", 1, 3.0, 1.5), ("chair", 2, 9.0, 6.0), ("plant", 4, 10.0, 1.5)]


class MockStack(Node):
    def __init__(self):
        super().__init__("x30_mock_stack")
        cb = ReentrantCallbackGroup()
        self.speed = self.declare_parameter("speed", 1.5).value
        self.standoff = self.declare_parameter("standoff_m", 1.0).value
        self.pose = SE2(0.0, 0.0, 0.0)
        self.map_origin = SE2(0.0, 0.0, 0.0)
        self.map_objects: dict[int, tuple[str, float, float]] = {}
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_gt = self.create_publisher(PoseStamped, "/gt/pose", 10)
        self.pub_ready = self.create_publisher(Bool, "/x30/ready", latched)
        self.pub_gtobj = self.create_publisher(SemanticObjectArray, "/gt/objects", latched)
        self.pub_info = self.create_publisher(String, "/gt/scene_info", latched)
        self.pub_map = self.create_publisher(SemanticObjectArray, "/semantic_map/objects", latched)
        self.pub_safe = self.create_publisher(SafetyStatus, "/x30/safety", 10)
        self.tf = TransformBroadcaster(self)
        self.create_service(Json, "/sim/set_robot_pose", self.on_set_pose, callback_group=cb)
        self.create_service(Json, "/semantic_map/reset", self.on_reset, callback_group=cb)
        self.action = ActionServer(self, Navigate, "/x30/navigate", self.execute, goal_callback=lambda g: GoalResponse.ACCEPT,
                                   cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=cb)
        self.pub_ready.publish(Bool(data=True))
        self.pub_info.publish(String(data=json.dumps({"name": "mock_world"})))
        self.pub_gtobj.publish(self._objects([(n, i, x, y) for n, i, x, y in OBJECTS], name_caption=True))
        self.create_timer(0.05, self.tick, callback_group=cb)
        self.create_timer(1.0, self.publish_map, callback_group=cb)

    @staticmethod
    def _objects(items, name_caption=False) -> SemanticObjectArray:
        msg = SemanticObjectArray()
        for label, oid, x, y in items:
            o = SemanticObject()
            o.id, o.label, o.confidence, o.observations = oid, label, 0.9, 3
            o.position.x, o.position.y, o.position.z = x, y, 0.4
            o.size.x = o.size.y = 0.5
            o.size.z = 0.8
            o.caption = f"{label}_{oid}" if name_caption else ""
            msg.objects.append(o)
        return msg

    def on_set_pose(self, req, resp):
        d = json.loads(req.request)
        self.pose = SE2(d["x"], d["y"], d.get("yaw", 0.0))
        self.map_origin = SE2(self.pose.x + 0.5, self.pose.y - 0.3, self.pose.yaw + 0.4)   # map frame offset on purpose
        resp.ok, resp.response = True, "{}"
        return resp

    def on_reset(self, req, resp):
        self.map_objects.clear()
        resp.ok, resp.response = True, "{}"
        return resp

    def tick(self):
        now = self.get_clock().now().to_msg()
        m = PoseStamped()
        m.header.stamp, m.header.frame_id = now, "world"
        m.pose.position.x, m.pose.position.y = self.pose.x, self.pose.y
        m.pose.orientation.z, m.pose.orientation.w = quat_from_yaw(self.pose.yaw)[2:]
        self.pub_gt.publish(m)
        local = self.map_origin.inverse().compose(self.pose)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "map", "base_link"
        t.transform.translation.x, t.transform.translation.y = local.x, local.y
        t.transform.rotation.z, t.transform.rotation.w = quat_from_yaw(local.yaw)[2:]
        self.tf.sendTransform(t)
        for label, oid, x, y in OBJECTS:
            if math.hypot(x - self.pose.x, y - self.pose.y) < 5.0 and oid not in self.map_objects:
                p = self.map_origin.inverse().apply([x, y])
                self.map_objects[oid] = (label, float(p[0]), float(p[1]))
        s = SafetyStatus()
        s.header.stamp, s.min_obstacle_distance = now, 9.0
        self.pub_safe.publish(s)

    def publish_map(self):
        self.pub_map.publish(self._objects([(lab, oid, x, y) for oid, (lab, x, y) in self.map_objects.items()]))

    def execute(self, gh):
        goal, result = gh.request, Navigate.Result()
        params = json.loads(goal.params_json or "{}")
        trace = params.get("trace_file")
        if trace:
            Path(trace).parent.mkdir(parents=True, exist_ok=True)
            with open(trace, "w", encoding="utf-8") as f:
                for k in (1, 2):
                    f.write(json.dumps({"kind": "turn", "turn": k, "tool": "goto_pose", "args": {"k": k}, "parse_ok": True, "success": True,
                                        "latency_s": 0.5, "prompt_tokens": 1000, "completion_tokens": 20}) + "\n")
        if goal.target_label == "boom":
            gh.abort()
            result.success, result.stop_reason, result.message = False, "error", "LLM error: Connection refused (http://vlm:8000)"
            return result
        if goal.mode == "goal_pose":
            local = SE2(goal.goal.pose.position.x, goal.goal.pose.position.y, 0.0)
            target = self.map_origin.compose(local)
            tx, ty = target.x, target.y
        else:
            objs = [(math.hypot(x - self.pose.x, y - self.pose.y), x, y) for lab, _, x, y in OBJECTS if lab == goal.target_label]
            if not objs:
                gh.abort()
                result.success, result.stop_reason, result.message = False, "aborted", "unknown label"
                return result
            _, ox, oy = min(objs)
            d = math.hypot(ox - self.pose.x, oy - self.pose.y)
            f = max(d - self.standoff, 0.0) / max(d, 1e-6)
            tx, ty = self.pose.x + f * (ox - self.pose.x), self.pose.y + f * (oy - self.pose.y)
        t0, dist0 = time.monotonic(), 0.0
        while rclpy.ok():
            if gh.is_cancel_requested:
                gh.canceled()
                result.success, result.stop_reason = False, "aborted"
                return result
            if time.monotonic() - t0 > goal.timeout_s:
                gh.abort()
                result.success, result.stop_reason, result.message = False, "timeout", "time limit"
                return result
            dx, dy = tx - self.pose.x, ty - self.pose.y
            d = math.hypot(dx, dy)
            if goal.instruction != "hang" and d < 0.05:
                break
            if goal.instruction != "hang":
                step = min(self.speed * 0.05, d)
                self.pose = SE2(self.pose.x + step * dx / d, self.pose.y + step * dy / d, math.atan2(dy, dx))
                dist0 += step
            fb = Navigate.Feedback()
            fb.status, fb.current_skill, fb.distance_to_goal, fb.agent_turn = "driving", "goto_pose", float(d), 1
            gh.publish_feedback(fb)
            time.sleep(0.05)
        gh.succeed()
        result.success, result.stop_reason, result.message = True, "finished", "arrived"
        result.path_length_m, result.duration_s, result.agent_turns = float(dist0), float(time.monotonic() - t0), 2
        return result


def main():
    rclpy.init()
    node = MockStack()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
