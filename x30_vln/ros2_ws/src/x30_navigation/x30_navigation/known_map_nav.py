"""known_map_nav: navigation on the KNOWN ground-truth occupancy grid of the world (no SLAM, no Nav2, no mapping).

* loads the world grid (registry world id -> map.yaml, or `map_yaml`), publishes it latched on /map (frame `map`, inflated by `map_inflation`)
* frame alignment: the simulator odometry (TF odom->base_link) starts at 0 at the spawn pose while the grid is in the world frame.
  T_world_odom is computed once from /gt/pose and TF odom->base_link (`align:=gt`), or from the registry spawn pose (`align:=spawn`).
  The `map` frame IS the odom frame (static identity TF map->odom is published); the grid is published with the origin / yaw that put it there.
  Waypoints of /nav/follow_waypoints are therefore given in odom coordinates (robot start = 0,0).
* serves the `navigate_to_pose` action (same as direct_navigator / Nav2), so waypoint_runner and the skills work unchanged (planner_backend:=known_map)
* A* (8-connected, footprint clearance tiers, smoothing) + pure pursuit; slows down with the distance to obstacles (from the grid, static world);
  rotation in place only in free space; replans periodically and when wedged / blocked (temporary blocker, back-up recovery).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from .geometry import wrap_angle
from .known_map import CtrlParams, KnownMap, PlanParams, control, load_world_grid, path_length, resolve_map


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class KnownMapNav(Node):
    def __init__(self) -> None:
        super().__init__("known_map_nav")
        d = self.declare_parameter
        d("world", "lab_01")
        d("map_yaml", "")                                  # overrides the registry lookup
        d("registry", "/x30/datasets/registry.yaml")
        d("worlds_root", "/x30/ros2_ws/src/x30_worlds")
        d("grid_resolution", 0.1)
        d("align", "gt")                                   # gt: /gt/pose + TF odom->base_link; spawn: registry spawn pose
        d("cmd_topic", "/cmd_vel_nav")
        d("map_frame", "map")
        d("odom_frame", "odom")
        d("base_frame", "base_link")
        d("publish_map_odom_tf", True)
        d("rate_hz", 10.0)
        d("goal_radius", 0.25)
        d("goal_snap_m", 3.0)                              # unreachable goal: move it to the nearest reachable cell within this distance
        d("body_half_length", 0.5)                         # X30 kinematic body 1.0 x 0.46 m
        d("body_half_width", 0.23)
        d("margin", 0.05)                                  # planner clearance tier 1 = half-length + margin; tier 3 = half-width + margin + 0.04
        d("map_inflation", 0.32)                           # inflation of the published /map [m]
        d("replan_period_s", 3.0)
        d("wedge_window_s", 4.0)
        d("backup_m", 0.35)
        d("stuck_timeout_s", 30.0)                         # sim seconds without 0.15 m of progress: abort the goal
        d("timeout_s", 240.0)                              # sim seconds per goal
        for k, v in vars(CtrlParams()).items():
            d("ctrl." + k, v)
        g = self.get_parameter
        self.P = {k: g(k).value for k in ("cmd_topic", "map_frame", "odom_frame", "base_frame", "rate_hz", "goal_radius", "backup_m",
                                          "replan_period_s", "wedge_window_s", "stuck_timeout_s", "timeout_s")}
        self.cp = CtrlParams(**{k: type(v)(g("ctrl." + k).value) for k, v in vars(CtrlParams()).items()})
        hl, hw, mg = g("body_half_length").value, g("body_half_width").value, g("margin").value
        self.cp.rot_clearance = hl
        yaml_path, spawn = g("map_yaml").value, {}
        if g("world").value:
            ypath, spawn = resolve_map(g("world").value, g("registry").value, g("worlds_root").value)
            yaml_path = yaml_path or ypath
        blocked, origin = load_world_grid(yaml_path, g("grid_resolution").value)
        self.km = KnownMap(blocked, origin, g("grid_resolution").value, PlanParams(radii=(hl + mg, 0.42, hw + mg + 0.04)))
        self.spawn = spawn
        self.get_logger().info(f"known map {yaml_path}: {blocked.shape[1]}x{blocked.shape[0]} cells, tiers {self.km.pp.radii}")
        self.T = None                                      # world <- odom: (tx, ty, theta)
        self.gt: PoseStamped | None = None
        self.map_inflation = g("map_inflation").value
        self.tf_buffer = Buffer()
        self.tfl = TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseStamped, "/gt/pose", lambda m: setattr(self, "gt", m), 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", qos)
        self.pub = self.create_publisher(Twist, self.P["cmd_topic"], 10)
        self.info_pub = self.create_publisher(String, "/nav/known_map_info", 10)
        self.static_tf = StaticTransformBroadcaster(self)
        # rclpy rejects cancel requests unless a cancel_callback is given: without it skill timeouts could not stop a goal, goal loops
        # leaked (still publishing /cmd_vel_nav) and filled the 4 executor threads, so later goals got no answer. A new goal pre-empts the old one.
        self.goal_gen = 0
        self.server = ActionServer(self, NavigateToPose, "navigate_to_pose", self.execute, goal_callback=self.on_goal,
                                   cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=ReentrantCallbackGroup())
        self.create_timer(1.0, self.try_align, callback_group=ReentrantCallbackGroup())

    # ---- frames -------------------------------------------------------------------------------------------------------------
    def odom_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.P["odom_frame"], self.P["base_frame"], rclpy.time.Time())
        except Exception:
            return None
        v, q = t.transform.translation, t.transform.rotation
        return v.x, v.y, yaw_of(q)

    def try_align(self) -> None:
        if self.T is not None:
            return
        op = self.odom_pose()
        if op is None:
            return
        if self.get_parameter("align").value == "gt":
            if self.gt is None:
                return
            wp = (self.gt.pose.position.x, self.gt.pose.position.y, yaw_of(self.gt.pose.orientation))
        else:
            if not self.spawn:
                return
            wp = (self.spawn["x"], self.spawn["y"], self.spawn.get("yaw", 0.0))
        th = wrap_angle(wp[2] - op[2])                     # world <- odom = world_pose * odom_pose^-1
        c, s = math.cos(th), math.sin(th)
        self.T = (wp[0] - (c * op[0] - s * op[1]), wp[1] - (s * op[0] + c * op[1]), th)
        self.get_logger().info(f"aligned: world<-odom t=({self.T[0]:.3f},{self.T[1]:.3f}) yaw={self.T[2]:.3f}")
        if self.get_parameter("publish_map_odom_tf").value:
            m = TransformStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id, m.child_frame_id = self.P["map_frame"], self.P["odom_frame"]
            m.transform.rotation.w = 1.0
            self.static_tf.sendTransform(m)
        self.publish_map()

    def w2o(self, xy):
        c, s, t = math.cos(self.T[2]), math.sin(self.T[2]), np.asarray(self.T[:2])
        d = np.asarray(xy, float) - t
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def o2w(self, xy):
        c, s = math.cos(self.T[2]), math.sin(self.T[2])
        return np.array([c * xy[0] - s * xy[1] + self.T[0], s * xy[0] + c * xy[1] + self.T[1]])

    def world_pose(self):
        op = self.odom_pose()
        if op is None or self.T is None:
            return None
        xy = self.o2w(op[:2])
        return float(xy[0]), float(xy[1]), wrap_angle(op[2] + self.T[2])

    def publish_map(self) -> None:
        km = self.km
        m = OccupancyGrid()
        m.header.frame_id = self.P["map_frame"]
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution, m.info.width, m.info.height = km.res, km.w, km.h
        o = self.w2o(km.origin)
        m.info.origin.position.x, m.info.origin.position.y = float(o[0]), float(o[1])
        yaw = -self.T[2]
        m.info.origin.orientation.z, m.info.origin.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        data = np.zeros((km.h, km.w), np.int8)
        data[km.dist < self.map_inflation] = 99
        data[km.blocked] = 100
        m.data = data.ravel().tolist()
        self.map_pub.publish(m)

    # ---- planning / action --------------------------------------------------------------------------------------------------
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def stop(self) -> None:
        self.pub.publish(Twist())

    def plan(self, wpose, gw):
        """(path, radius, goal used). When the goal is unreachable it is moved to the nearest reachable cell (within goal_snap_m)."""
        r = self.km.plan(wpose[:2], gw)
        if r is None:
            alt = self.km.nearest_reachable(wpose[:2], gw, float(self.get_parameter("goal_snap_m").value))
            if alt is not None:
                self.get_logger().info(f"goal ({gw[0]:.2f}, {gw[1]:.2f}) unreachable: moved {float(np.hypot(*(alt[0] - gw))):.2f} m to ({alt[0][0]:.2f}, {alt[0][1]:.2f})")
                gw = np.asarray(alt[0], float)
                r = self.km.plan(wpose[:2], gw)
        if r is None:
            self.get_logger().warn("no path on the known map")
            return None, 0.0, gw
        return r[0], r[1], gw

    def on_goal(self, _request):
        self.goal_gen += 1
        return GoalResponse.ACCEPT

    def execute(self, handle):
        my_gen = self.goal_gen
        goal = handle.request.pose.pose
        result = NavigateToPose.Result()
        t_wall = time.monotonic()
        while self.world_pose() is None and time.monotonic() - t_wall < 15.0:
            time.sleep(0.1)
        wp = self.world_pose()
        if wp is None:
            handle.abort()
            return result
        gw = self.o2w([goal.position.x, goal.position.y])
        period = 1.0 / self.P["rate_hz"]
        t0 = last_plan = last_prog = self.now()
        self.km.clear_blockers()
        path, rad, gw = self.plan(wp, gw)
        if path is None:
            handle.abort()
            return result
        self.info_pub.publish(String(data=json.dumps({"event": "plan", "goal_world": gw.tolist(), "radius": rad, "length": path_length(path),
                                                      "path": np.round(path, 2).tolist()})))
        anchor, win = np.array(wp[:2]), (self.now(), np.array(wp[:2]), wp[2])
        backing = None
        while rclpy.ok():
            if my_gen != self.goal_gen:                   # pre-empted by a newer goal: end quietly (the new goal owns /cmd_vel_nav)
                handle.abort()
                return result
            if handle.is_cancel_requested:
                self.stop()
                handle.canceled()
                return result
            now = self.now()
            wp = self.world_pose()
            if wp is None:
                time.sleep(period)
                continue
            pos = np.array(wp[:2])
            if now - t0 > self.P["timeout_s"]:
                break
            if np.linalg.norm(pos - anchor) > 0.15:
                anchor, last_prog = pos.copy(), now
            elif now - last_prog > self.P["stuck_timeout_s"]:
                self.get_logger().warn("no progress, aborting goal")
                break
            if backing is not None:                      # wedge recovery: slide backwards out of the contact (holonomic, no rotation)
                cmd = Twist()
                cmd.linear.x = -0.2
                self.pub.publish(cmd)
                if np.linalg.norm(pos - backing[0]) > self.P["backup_m"] or now - backing[1] > 8.0:
                    ahead = pos + 0.7 * np.array([math.cos(wp[2]), math.sin(wp[2])])
                    self.km.add_blocker(ahead, 0.3)       # whatever hurt us lies ahead: plan around it
                    backing, win = None, (now, pos.copy(), wp[2])
                    p2, r2, gw = self.plan(wp, gw)
                    if p2 is not None:
                        path, rad = p2, r2
                    last_plan = now
                    self.info_pub.publish(String(data=json.dumps({"event": "replan_blocked", "length": path_length(path), "path": np.round(path, 2).tolist()})))
                time.sleep(period)
                continue
            if now - win[0] > self.P["wedge_window_s"]:
                moved = np.linalg.norm(pos - win[1]) > 0.03 or abs(wrap_angle(wp[2] - win[2])) > 0.05
                if not moved and np.linalg.norm(pos - gw) > self.P["goal_radius"]:
                    self.get_logger().info("wedged: backing up")
                    backing = (pos.copy(), now)
                    continue
                win = (now, pos.copy(), wp[2])
            if now - last_plan > self.P["replan_period_s"]:
                p2, r2, gw = self.plan(wp, gw)
                if p2 is not None:
                    path, rad = p2, r2
                last_plan = now
            if np.linalg.norm(pos - gw) < self.P["goal_radius"]:
                self.stop()
                self.km.clear_blockers()
                handle.succeed()
                return result
            vx, vy, wz, _ = control(wp, path, self.km.clearance(pos), self.cp)
            cmd = Twist()
            cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
            self.pub.publish(cmd)
            time.sleep(period)
        self.stop()
        self.km.clear_blockers()
        handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KnownMapNav()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
