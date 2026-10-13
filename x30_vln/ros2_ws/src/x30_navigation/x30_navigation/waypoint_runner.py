"""waypoint_runner: drives a list of waypoints through the `navigate_to_pose` action (Nav2 or direct_navigator) and reports each one.

Publishes JSON progress on /nav/waypoint_progress (state driving|done, index, total, reached).
Service /nav/follow_waypoints (x30_msgs/srv/Json), frame `map`:
  request  {"waypoints": [[x, y, yaw?], ...], "radius": 0.4, "timeout_s": 120, "stop_on_failure": false}
  response {"ok": bool (all reached), "results": [{"index", "goal": [x, y], "reached": bool, "status", "final_error_m", "duration_s"}], "reached": n}
A waypoint counts as reached when the action succeeded and the final distance (TF map -> base_link) is below `radius`.
"""
from __future__ import annotations

import json
import math
import time

import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from x30_msgs.srv import Json

STATUS = {4: "succeeded", 5: "canceled", 6: "aborted"}


class WaypointRunner(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_runner")
        self.declare_parameter("default_radius", 0.4)
        self.declare_parameter("default_timeout_s", 120.0)
        self.declare_parameter("planner_backend", "nav2")   # informational: the backend is chosen at launch (navigation.launch.py planner_backend:=)
        self.cb = ReentrantCallbackGroup()
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose", callback_group=self.cb)
        self.tf_buffer = Buffer()
        self.tfl = TransformListener(self.tf_buffer, self)
        self.progress = self.create_publisher(String, "/nav/waypoint_progress", 10)
        self.create_service(Json, "/nav/follow_waypoints", self.on_request, callback_group=self.cb)

    def position(self):
        try:
            v = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time()).transform.translation
            return v.x, v.y
        except Exception:
            return None

    def publish(self, d: dict) -> None:
        self.progress.publish(String(data=json.dumps(d)))

    def wait(self, fut, timeout):
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.05)
        return fut.done()

    def go(self, wp, radius, timeout) -> dict:
        x, y = float(wp[0]), float(wp[1])
        yaw = float(wp[2]) if len(wp) > 2 else 0.0
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        t0, status = time.monotonic(), "rejected"
        if self.nav.wait_for_server(timeout_sec=5.0) and self.wait(sf := self.nav.send_goal_async(goal), 10.0) and sf.result().accepted:
            h = sf.result()
            rf = h.get_result_async()
            if self.wait(rf, timeout):
                status = STATUS.get(rf.result().status, str(rf.result().status))
            else:
                h.cancel_goal_async()
                time.sleep(1.0)
                status = "timeout"
        pos = self.position()
        err = math.hypot(pos[0] - x, pos[1] - y) if pos else float("nan")
        return {"goal": [x, y], "reached": bool(status == "succeeded" and err < radius), "status": status,
                "final_error_m": round(err, 3), "duration_s": round(time.monotonic() - t0, 1)}

    def on_request(self, req, resp):
        try:
            r = json.loads(req.request or "{}")
            wps = r["waypoints"]
        except (ValueError, KeyError, TypeError) as exc:
            resp.ok, resp.response = False, json.dumps({"ok": False, "error": f"bad request: {exc}"})
            return resp
        radius = float(r.get("radius", self.get_parameter("default_radius").value))
        timeout = float(r.get("timeout_s", self.get_parameter("default_timeout_s").value))
        results = []
        for i, wp in enumerate(wps):
            self.publish({"state": "driving", "index": i, "total": len(wps), "goal": [float(wp[0]), float(wp[1])], "reached": sum(1 for q in results if q["reached"])})
            res = self.go(wp, radius, timeout)
            res["index"] = i
            results.append(res)
            self.get_logger().info(f"waypoint {i}: {res['status']}, error {res['final_error_m']} m")
            if not res["reached"] and r.get("stop_on_failure", False):
                break
        n = sum(1 for x in results if x["reached"])
        self.publish({"state": "done", "index": len(results), "total": len(wps), "reached": n})
        resp.ok = n == len(wps)
        resp.response = json.dumps({"ok": resp.ok, "results": results, "reached": n, "backend": self.get_parameter("planner_backend").value})
        return resp


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointRunner()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
