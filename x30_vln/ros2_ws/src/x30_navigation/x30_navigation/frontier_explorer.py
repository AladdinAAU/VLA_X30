"""frontier_explorer: frontier service for skills (/nav/next_frontier) and an optional autonomous explore loop.

Service /nav/next_frontier (x30_msgs/srv/Json)
  request  {"strategy": "nearest|largest|informative|biased", "bias": [x, y], "exclude": [[x, y], ...]}
  response {"ok": bool, "found": bool, "goal": [x, y, yaw], "goal_pose": {"x","y","yaw"}, "info_gain": m^2,
            "frontier_count": n, "remaining": n, "explored_fraction": 0..1, "distance_m": path distance,
            "size": cells, "reason": str}
  Goals are reachable free cells at least `robot_radius` from obstacles, in the map frame, yaw towards the unknown.
Services /nav/explore/start and /nav/explore/stop (std_srvs/Trigger) run the autonomous loop: pick a frontier, send
NavigateToPose, blacklist goals that fail or time out, re-plan when a frontier gets explored on the way, stop
when nothing reachable is left. Progress is published as JSON on /nav/explore_status.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from x30_msgs.srv import Json

from .frontiers import FrontierParams, FrontierResult, find_frontiers, select


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        d = self.declare_parameter
        d("map_topic", "/map")
        d("map_frame", "map")
        d("base_frame", "base_link")
        d("autonomous", False)
        d("strategy", "informative")
        d("loop_period", 1.0)
        d("goal_timeout", 150.0)            # s per goal before it is blacklisted
        d("replan_check_period", 3.0)       # s; cancel the goal when its frontier was explored on the way
        d("max_consecutive_empty", 3)       # empty answers in a row before the exploration is declared complete
        d("blacklist_radius", 0.8)
        for k, v in vars(FrontierParams()).items():
            d("frontier." + k, v)
        g = self.get_parameter
        self.map_frame, self.base_frame = g("map_frame").value, g("base_frame").value
        self.strategy = g("strategy").value
        self.goal_timeout = float(g("goal_timeout").value)
        self.replan_period = float(g("replan_check_period").value)
        self.max_empty = int(g("max_consecutive_empty").value)
        self.blacklist_radius = float(g("blacklist_radius").value)
        fp = FrontierParams()
        for k, v in vars(fp).items():
            setattr(fp, k, type(v)(g("frontier." + k).value))
        self.fp = fp

        self.map: OccupancyGrid | None = None
        self.grid: np.ndarray | None = None
        self.cache_key = None
        self.cache: FrontierResult | None = None
        self.blacklist: list[tuple[float, float]] = []
        self.running = bool(g("autonomous").value)
        self.state = "idle"
        self.current_goal: tuple[float, float, float] | None = None
        self.goal_handle = None
        self.goal_start = 0.0
        self.last_replan_check = 0.0
        self.empty_count = 0
        self.stats = {"sent": 0, "succeeded": 0, "failed": 0, "blacklisted": 0}
        self.start_time = time.monotonic()
        self.history: list[dict] = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, g("map_topic").value, self.on_map, qos)
        self.create_service(Json, "/nav/next_frontier", self.on_next_frontier)
        self.create_service(Trigger, "/nav/explore/start", self.on_start)
        self.create_service(Trigger, "/nav/explore/stop", self.on_stop)
        self.status_pub = self.create_publisher(String, "/nav/explore_status", 5)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.create_timer(float(g("loop_period").value), self.loop)

    # ---- map and frontiers --------------------------------------------------------------------------------------
    def on_map(self, msg: OccupancyGrid) -> None:
        self.map = msg
        self.grid = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

    def robot_xy(self) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None
        return np.array([tf.transform.translation.x, tf.transform.translation.y])

    def frontiers(self) -> FrontierResult | None:
        if self.map is None or self.grid is None:
            return None
        xy = self.robot_xy()
        if xy is None:
            return None
        key = (self.map.header.stamp.sec, self.map.header.stamp.nanosec, round(xy[0], 1), round(xy[1], 1))
        if key != self.cache_key:
            info = self.map.info
            origin = np.array([info.origin.position.x, info.origin.position.y])
            self.cache = find_frontiers(self.grid, info.resolution, origin, xy, self.fp)
            self.cache_key = key
        return self.cache

    @staticmethod
    def _answer(ok: bool, **kw) -> str:
        return json.dumps({"ok": ok, **kw})

    def on_next_frontier(self, req: Json.Request, resp: Json.Response) -> Json.Response:
        try:
            q = json.loads(req.request) if req.request.strip() else {}
        except json.JSONDecodeError as exc:
            resp.ok, resp.response = False, self._answer(False, found=False, reason=f"bad json: {exc}")
            return resp
        res = self.frontiers()
        if res is None:
            resp.ok = False
            resp.response = self._answer(False, found=False, reason="no map or no robot pose yet")
            return resp
        exclude = [tuple(e[:2]) for e in q.get("exclude", [])] + self.blacklist
        fr = select(res.frontiers, q.get("strategy", self.strategy), q.get("bias"), exclude, self.fp.exclude_radius)
        remaining = sum(1 for f in res.frontiers
                        if all(math.hypot(f.goal[0] - e[0], f.goal[1] - e[1]) > self.fp.exclude_radius for e in exclude))
        common = dict(frontier_count=len(res.frontiers), remaining=remaining,
                      explored_fraction=round(res.explored_fraction, 4))
        if fr is None:
            resp.ok = True
            resp.response = self._answer(True, found=False, goal=None, info_gain=0.0,
                                         reason="exhausted" if not res.frontiers else "all frontiers excluded", **common)
            return resp
        resp.ok = True
        resp.response = self._answer(True, found=True, goal=[fr.goal[0], fr.goal[1], fr.goal[2]],
                                     goal_pose={"x": fr.goal[0], "y": fr.goal[1], "yaw": fr.goal[2]},
                                     info_gain=round(fr.info_gain, 2), distance_m=round(fr.distance_m, 2),
                                     size=fr.size, reason="", **common)
        return resp

    # ---- autonomous loop ----------------------------------------------------------------------------------------
    def on_start(self, _req, resp):
        self.running, self.state, self.empty_count = True, "idle", 0
        resp.success, resp.message = True, "exploration started"
        return resp

    def on_stop(self, _req, resp):
        self.running = False
        self.cancel_goal()
        self.state = "stopped"
        resp.success, resp.message = True, "exploration stopped"
        return resp

    def cancel_goal(self) -> None:
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        self.current_goal = None

    def loop(self) -> None:
        self.publish_status()
        if not self.running:
            return
        if self.state == "navigating":
            now = time.monotonic()
            if now - self.goal_start > self.goal_timeout:
                self.get_logger().warn("goal timed out, blacklisting it")
                self.fail_current("timeout")
            elif now - self.last_replan_check > self.replan_period:
                self.last_replan_check = now
                self.check_goal_still_a_frontier()
            return
        if self.state == "complete":
            return
        res = self.frontiers()
        if res is None:
            return
        fr = select(res.frontiers, self.strategy, None, self.blacklist, self.fp.exclude_radius)
        if fr is None:
            self.empty_count += 1
            if self.empty_count >= self.max_empty:
                self.state = "complete"
                self.get_logger().info(f"exploration complete (explored fraction {res.explored_fraction:.2f}, "
                                       f"{len(self.blacklist)} blacklisted goals)")
            return
        self.empty_count = 0
        xy = self.robot_xy()
        heading = math.atan2(fr.goal[1] - xy[1], fr.goal[0] - xy[0]) if xy is not None else fr.goal[2]
        self.send_goal((fr.goal[0], fr.goal[1], heading))   # arrive facing the direction of travel: no final turn near walls

    def send_goal(self, goal) -> None:
        if not self.nav.server_is_ready():
            self.get_logger().warn("navigate_to_pose not available yet", throttle_duration_sec=5.0)
            return
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = goal[0], goal[1]
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(goal[2] / 2.0), math.cos(goal[2] / 2.0)
        self.current_goal = goal
        self.state = "navigating"
        self.goal_start = self.last_replan_check = time.monotonic()
        self.stats["sent"] += 1
        self.get_logger().info(f"frontier goal ({goal[0]:.2f}, {goal[1]:.2f})")
        fut = self.nav.send_goal_async(NavigateToPose.Goal(pose=pose))
        fut.add_done_callback(self.on_goal_response)

    def on_goal_response(self, fut) -> None:
        handle = fut.result()
        if not handle.accepted:
            self.fail_current("rejected")
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.on_result)

    def on_result(self, fut) -> None:
        status = fut.result().status
        if self.current_goal is None:
            return  # cancelled on purpose
        if status == 4:  # SUCCEEDED
            self.stats["succeeded"] += 1
            self.blacklist.append(self.current_goal[:2])  # visited: its frontier is explored now, never revisit the spot
            self.finish_goal()
        else:
            self.fail_current(f"status {status}")

    def fail_current(self, why: str) -> None:
        if self.current_goal is not None:
            self.blacklist.append(self.current_goal[:2])
            self.stats["failed"] += 1
            self.stats["blacklisted"] += 1
            self.history.append({"t": round(time.monotonic() - self.start_time, 1), "goal": list(self.current_goal[:2]), "result": why})
        self.cancel_goal()
        self.state = "idle"

    def finish_goal(self) -> None:
        self.goal_handle = None
        self.current_goal = None
        self.state = "idle"

    def check_goal_still_a_frontier(self) -> None:
        res = self.frontiers()
        if res is None or self.current_goal is None:
            return
        gx, gy = self.current_goal[:2]
        if not any(math.hypot(f.centroid[0] - gx, f.centroid[1] - gy) < 2.0 or math.hypot(f.goal[0] - gx, f.goal[1] - gy) < 2.0
                   for f in res.frontiers):
            self.get_logger().info("goal frontier was explored on the way: re-planning")
            self.cancel_goal()
            self.state = "idle"

    def publish_status(self) -> None:
        res = self.cache
        msg = String()
        msg.data = json.dumps({
            "state": self.state if self.running else ("stopped" if self.state != "idle" else "idle"),
            "running": self.running, "elapsed_s": round(time.monotonic() - self.start_time, 1),
            "frontier_count": len(res.frontiers) if res else None,
            "explored_fraction": round(res.explored_fraction, 4) if res else None,
            "known_free_area_m2": round(res.known_free_area, 1) if res else None,
            "goal": list(self.current_goal[:2]) if self.current_goal else None,
            "blacklist": len(self.blacklist), **self.stats})
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
