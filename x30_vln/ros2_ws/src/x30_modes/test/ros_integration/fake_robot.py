"""Minimal fake X30 for integration tests of x30_modes (no simulator).

Provides the robot-facing contract and just enough of the neighbours for every skill to run:
  * dynamics: /cmd_vel integrated with command delay and first-order velocity lag; true pose in the `map` frame
  * /odom + TF odom->base_link, static map->odom with a deliberate offset (catches odom/map mix-ups)
  * /costmap_cloud (frame odom) and /scan (frame base_link) generated from axis-aligned boxes
  * /imu/data, /camera/color/image_raw, /map, /x30/ready, /x30/estop is NOT created here (tests publish it)
  * fake Nav2 NavigateToPose: A* on the static map + pure pursuit publishing /cmd_vel_nav (so it passes mux and guard)
  * /semantic_map/objects + /semantic_map/query (objects are revealed when the robot comes within reveal_range)
  * /nav/next_frontier
  * /fake/truth (PoseStamped, map frame), /fake/stats (JSON), /fake/add_box (Float64MultiArray [x0,y0,x1,y1]: an obstacle that is
    in the lidar but not in the static map), /fake/tilt_deg (Float64)
Everything is parameterised through the `world_json` parameter (see DEFAULT_WORLD).
"""
from __future__ import annotations

import heapq
import json
import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from x30_modes.geometry import Footprint, Pose2D, quat_from_yaw, rot2, wrap_pi, yaw_from_quat
from x30_modes.guard_logic import gap_to_footprint
from x30_msgs.msg import SemanticObject, SemanticObjectArray
from x30_msgs.srv import Json

LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)

DEFAULT_WORLD = {
    "start": [0.0, 0.0, 0.0],                     # true pose in the map frame
    "map_offset": [1.5, -0.7, 0.3],               # map -> odom (odom origin = map_offset in map): deliberate mismatch
    "bounds": [-1.0, -4.0, 9.0, 4.0],             # x0, y0, x1, y1 of the room (walls just outside)
    "boxes": [[3.0, -0.5, 3.6, 0.5],              # pillar
              [6.0, 1.5, 7.0, 2.5],               # table
              [4.75, -2.75, 5.25, -2.25]],        # chair body
    "footprint": [0.5, 0.5, 0.32],                # front, rear, half width (same as the guard defaults)
    "objects": [{"id": 1, "label": "chair", "x": 5.0, "y": -2.5, "size": [0.5, 0.5, 0.9], "confidence": 0.9},
                {"id": 2, "label": "table", "x": 6.5, "y": 2.0, "size": [1.0, 1.0, 0.75], "confidence": 0.8},
                {"id": 3, "label": "pillar", "x": 3.3, "y": 0.0, "size": [0.6, 1.0, 2.0], "confidence": 0.7}],
    "reveal_range": 4.5,
    "frontiers": [[8.0, 3.0], [1.0, -3.0]],
    "tau": 0.15, "delay": 0.05,
    "nav_speed": 0.35, "inflation": 0.5,
}


def box_points(b, step=0.02):
    x0, y0, x1, y1 = b
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    return np.concatenate([
        np.stack([xs, np.full_like(xs, y0)], 1), np.stack([xs, np.full_like(xs, y1)], 1),
        np.stack([np.full_like(ys, x0), ys], 1), np.stack([np.full_like(ys, x1), ys], 1)])


class FakeRobot(Node):
    def __init__(self):
        super().__init__("fake_robot")
        w = dict(DEFAULT_WORLD)
        w.update(json.loads(self.declare_parameter("world_json", "{}").value))
        self.w = w
        self.fp = Footprint(*w["footprint"])
        self.dt = 0.01
        self.x, self.y, self.yaw = w["start"]
        self.v = np.zeros(3)
        self.queue = deque([(0.0, 0.0, 0.0)] * max(int(round(w["delay"] / self.dt)), 0))
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_stamp = 0.0
        self.mo = w["map_offset"]
        self.lock = threading.RLock()
        self.static_boxes = [list(b) for b in w["boxes"]]
        self.dyn_boxes: list[list[float]] = []
        self.walls = self._wall_boxes(w["bounds"])
        self._rebuild_points()
        self.tilt = 0.0
        self.collisions, self.in_collision, self.min_gap, self.path_len = 0, False, math.inf, 0.0
        self.frontiers = [tuple(f) for f in w["frontiers"]]
        self.cb = ReentrantCallbackGroup()

        self.tfb, self.stfb = TransformBroadcaster(self), StaticTransformBroadcaster(self)
        self._publish_static()
        self.pub_odom = self.create_publisher(Odometry, "/odom", 20)
        self.pub_cloud = self.create_publisher(PointCloud2, "/costmap_cloud", 5)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", 5)
        self.pub_imu = self.create_publisher(Imu, "/imu/data", 20)
        self.pub_img = self.create_publisher(Image, "/camera/color/image_raw", 5)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", LATCHED)
        self.pub_ready = self.create_publisher(Bool, "/x30/ready", LATCHED)
        self.pub_objs = self.create_publisher(SemanticObjectArray, "/semantic_map/objects", LATCHED)
        self.pub_truth = self.create_publisher(PoseStamped, "/fake/truth", 10)
        self.pub_stats = self.create_publisher(String, "/fake/stats", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10, callback_group=self.cb)
        self.create_subscription(Float64MultiArray, "/fake/add_box", self.on_add_box, 10, callback_group=self.cb)
        self.create_subscription(Float64MultiArray, "/fake/reset", self.on_reset, 10, callback_group=self.cb)
        self.create_subscription(Float64, "/fake/tilt_deg", lambda m: setattr(self, "tilt", m.data), 10, callback_group=self.cb)
        self.create_service(Json, "/semantic_map/query", self.on_query, callback_group=self.cb)
        self.create_service(Json, "/nav/next_frontier", self.on_frontier, callback_group=self.cb)
        self.nav_server = ActionServer(self, NavigateToPose, "navigate_to_pose", execute_callback=self.nav_execute,
                                       goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda g: CancelResponse.ACCEPT,
                                       callback_group=self.cb)
        self.pub_nav = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.grid = self._make_grid()
        self.pub_map.publish(self._grid_msg())
        self.pub_ready.publish(Bool(data=True))
        self.revealed: set[int] = set()
        self._n_pub = -1
        self.create_timer(self.dt, self.physics, callback_group=self.cb)
        self.create_timer(0.1, self.sensors, callback_group=self.cb)
        self.create_timer(0.5, self.slow, callback_group=self.cb)
        self.create_timer(0.02, self.fast, callback_group=self.cb)
        self.get_logger().info("fake_robot ready")

    # ------------------------------------------------------------ world
    @staticmethod
    def _wall_boxes(b):
        x0, y0, x1, y1 = b
        t = 0.2
        return [[x0 - t, y0 - t, x1 + t, y0], [x0 - t, y1, x1 + t, y1 + t], [x0 - t, y0, x0, y1], [x1, y0, x1 + t, y1]]

    def _rebuild_points(self):
        with self.lock:
            pts = [box_points(b) for b in self.walls + self.static_boxes + self.dyn_boxes]
            self.pts_map = np.concatenate(pts, 0)

    def on_add_box(self, m: Float64MultiArray):
        with self.lock:
            self.dyn_boxes.append(list(m.data[:4]))
        self._rebuild_points()
        self.get_logger().info(f"dynamic obstacle added: {list(m.data[:4])}")

    def on_reset(self, m: Float64MultiArray):
        """[x, y, yaw] teleports the robot and clears the statistics; a 4th value 1 also restores the world (objects, frontiers, boxes)."""
        with self.lock:
            self.x, self.y, self.yaw = m.data[0], m.data[1], m.data[2]
            self.v[:] = 0.0
            self.queue = deque([(0.0, 0.0, 0.0)] * len(self.queue))
            self.cmd = (0.0, 0.0, 0.0)
            self.collisions, self.in_collision, self.min_gap, self.path_len = 0, False, math.inf, 0.0
            if len(m.data) > 3 and m.data[3] > 0:
                self.dyn_boxes.clear()
                self.revealed.clear()
                self.frontiers = [tuple(f) for f in self.w["frontiers"]]
        self._rebuild_points()

    def _make_grid(self, res=0.1):
        x0, y0, x1, y1 = self.w["bounds"]
        ox, oy = x0 - 0.5, y0 - 0.5
        wd, ht = int((x1 - x0 + 1.0) / res), int((y1 - y0 + 1.0) / res)
        g = np.zeros((ht, wd), np.int8)
        for b in self.walls + self.static_boxes:
            i0, i1 = int((b[0] - ox) / res), int(math.ceil((b[2] - ox) / res))
            j0, j1 = int((b[1] - oy) / res), int(math.ceil((b[3] - oy) / res))
            g[max(j0, 0):j1, max(i0, 0):i1] = 100
        return {"data": g, "res": res, "ox": ox, "oy": oy}

    def _grid_msg(self):
        g = self.grid
        m = OccupancyGrid()
        m.header.frame_id = "map"
        m.info.resolution, m.info.width, m.info.height = g["res"], g["data"].shape[1], g["data"].shape[0]
        m.info.origin.position.x, m.info.origin.position.y = g["ox"], g["oy"]
        m.info.origin.orientation.w = 1.0
        m.data = g["data"].flatten().tolist()
        return m

    # ------------------------------------------------------------ frames
    def odom_from_map(self, x, y, yaw):
        """Pose in the odom frame of a map-frame pose (odom origin = map_offset)."""
        ox, oy, oyaw = self.mo
        p = rot2(-oyaw) @ [x - ox, y - oy]
        return p[0], p[1], wrap_pi(yaw - oyaw)

    def _publish_static(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "map", "odom"
        t.transform.translation.x, t.transform.translation.y = self.mo[0], self.mo[1]
        q = quat_from_yaw(self.mo[2])
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.stfb.sendTransform(t)

    # ------------------------------------------------------------ physics
    def on_cmd(self, m: Twist):
        self.cmd = (m.linear.x, m.linear.y, m.angular.z)

    def physics(self):
        self.queue.append(self.cmd)
        applied = self.queue.popleft()
        a = self.dt / (self.w["tau"] + self.dt)
        with self.lock:
            self.v += a * (np.array(applied) - self.v)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            dx, dy = (c * self.v[0] - s * self.v[1]) * self.dt, (s * self.v[0] + c * self.v[1]) * self.dt
            self.x += dx
            self.y += dy
            self.yaw = wrap_pi(self.yaw + self.v[2] * self.dt)
            self.path_len += math.hypot(dx, dy)

    def fast(self):
        """Odom/TF at 50 Hz and the clearance bookkeeping."""
        with self.lock:
            x, y, yaw = self.x, self.y, self.yaw
            v = self.v.copy()
        ox, oy, oyaw = self.odom_from_map(x, y, yaw)
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "odom", "base_link"
        t.transform.translation.x, t.transform.translation.y = ox, oy
        q = quat_from_yaw(oyaw)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tfb.sendTransform(t)
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now, "odom", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y = ox, oy
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = q
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = v
        self.pub_odom.publish(o)
        # clearance
        rel = Pose2D(x, y, yaw).inverse_apply(self.pts_map[np.hypot(self.pts_map[:, 0] - x, self.pts_map[:, 1] - y) < 2.5])
        gap = gap_to_footprint(rel, self.fp)
        self.min_gap = min(self.min_gap, gap)
        hit = gap < 0.003
        if hit and not self.in_collision:
            self.collisions += 1
            self.get_logger().warn(f"COLLISION at ({x:.2f}, {y:.2f})")
        self.in_collision = hit
        self.last_gap = gap

    # ------------------------------------------------------------ sensors
    def sensors(self):
        with self.lock:
            x, y, yaw = self.x, self.y, self.yaw
        now = self.get_clock().now().to_msg()
        near = self.pts_map[np.hypot(self.pts_map[:, 0] - x, self.pts_map[:, 1] - y) < 6.0]
        # cloud in the odom frame
        ox, oy, oyaw = self.mo
        po = (near - [ox, oy]) @ rot2(oyaw)             # = rot2(-oyaw) applied to rows
        xyz = np.concatenate([po, np.full((po.shape[0], 1), 0.2)], 1).astype(np.float32)
        c = PointCloud2()
        c.header.stamp, c.header.frame_id = now, "odom"
        c.height, c.width = 1, xyz.shape[0]
        c.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        c.is_bigendian, c.point_step, c.row_step, c.is_dense = False, 12, 12 * xyz.shape[0], True
        c.data = xyz.tobytes()
        self.pub_cloud.publish(c)
        # scan in base_link
        rel = Pose2D(x, y, yaw).inverse_apply(near)
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        rng = np.hypot(rel[:, 0], rel[:, 1])
        n = 360
        idx = np.clip(((ang + math.pi) / (2 * math.pi) * n).astype(int), 0, n - 1)
        ranges = np.full(n, np.inf, np.float32)
        np.minimum.at(ranges, idx, rng.astype(np.float32))
        sc = LaserScan()
        sc.header.stamp, sc.header.frame_id = now, "base_link"
        sc.angle_min, sc.angle_max, sc.angle_increment = -math.pi, math.pi, 2 * math.pi / n
        sc.range_min, sc.range_max = 0.05, 10.0
        sc.ranges = ranges.tolist()
        self.pub_scan.publish(sc)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = now, "imu_link"
        t = math.radians(self.tilt)
        imu.orientation.x, imu.orientation.w = math.sin(t / 2), math.cos(t / 2)
        imu.linear_acceleration.z = 9.81
        self.pub_imu.publish(imu)
        img = Image()
        img.header.stamp, img.header.frame_id = now, "camera_optical_frame"
        img.height, img.width, img.encoding, img.step = 120, 160, "rgb8", 160 * 3
        img.data = bytes([90, 140, 200]) * (160 * 120)
        self.pub_img.publish(img)
        truth = PoseStamped()
        truth.header.stamp, truth.header.frame_id = now, "map"
        truth.pose.position.x, truth.pose.position.y = x, y
        q = quat_from_yaw(yaw)
        truth.pose.orientation.x, truth.pose.orientation.y, truth.pose.orientation.z, truth.pose.orientation.w = q
        self.pub_truth.publish(truth)

    def slow(self):
        with self.lock:
            x, y = self.x, self.y
        # reveal objects and retire frontiers
        changed = False
        for o in self.w["objects"]:
            if o["id"] not in self.revealed and math.hypot(o["x"] - x, o["y"] - y) < self.w["reveal_range"]:
                self.revealed.add(o["id"])
                changed = True
        if changed or len(self.revealed) != getattr(self, '_n_pub', -1):
            self._n_pub = len(self.revealed)
            self.pub_objs.publish(self._objects_msg())
        self.frontiers = [f for f in self.frontiers if math.hypot(f[0] - x, f[1] - y) > 1.5]
        self.pub_stats.publish(String(data=json.dumps({
            "x": x, "y": y, "yaw": self.yaw, "collisions": self.collisions, "min_gap": None if math.isinf(self.min_gap) else self.min_gap,
            "path_len": self.path_len, "gap_now": getattr(self, "last_gap", None)})))

    def _objects_msg(self):
        m = SemanticObjectArray()
        m.header.frame_id = "map"
        for o in self.w["objects"]:
            if o["id"] not in self.revealed:
                continue
            so = SemanticObject()
            so.id, so.label, so.confidence = o["id"], o["label"], float(o.get("confidence", 0.9))
            so.position.x, so.position.y, so.position.z = o["x"], o["y"], 0.4
            so.size.x, so.size.y, so.size.z = o["size"]
            so.bearing_only = bool(o.get("bearing_only", False))
            so.caption = o.get("caption", "")
            m.objects.append(so)
        return m

    def on_query(self, req, res):
        q = json.loads(req.request or "{}")
        out = []
        for o in self.w["objects"]:
            if o["id"] not in self.revealed:
                continue
            if "id" in q and o["id"] != q["id"]:
                continue
            if q.get("label") and q["label"].lower() not in o["label"].lower():
                continue
            out.append({"id": o["id"], "label": o["label"], "confidence": o.get("confidence", 0.9), "position": [o["x"], o["y"], 0.4],
                        "size": o["size"], "bearing_only": bool(o.get("bearing_only", False)), "caption": "", "attributes": [], "room": ""})
        res.ok = True
        res.response = json.dumps({"objects": out[: int(q.get("limit", 10))]})
        return res

    def on_frontier(self, req, res):
        q = json.loads(req.request or "{}")
        ex = q.get("exclude", [])
        with self.lock:
            x, y = self.x, self.y
        cands = [f for f in self.frontiers if not any(math.hypot(f[0] - e[0], f[1] - e[1]) < 0.5 for e in ex)]
        res.ok = True
        if not cands:
            res.response = json.dumps({"found": False, "remaining": 0, "reason": "no frontier left"})
            return res
        f = min(cands, key=lambda f: math.hypot(f[0] - x, f[1] - y))
        res.response = json.dumps({"found": True, "goal": {"x": f[0], "y": f[1], "yaw": 0.0}, "size": 12,
                                   "distance_m": math.hypot(f[0] - x, f[1] - y), "remaining": len(cands) - 1})
        return res

    # ------------------------------------------------------------ fake Nav2
    def plan(self, start, goal):
        g = self.grid
        res, ox, oy = g["res"], g["ox"], g["oy"]
        occ = g["data"] > 50
        r = int(math.ceil(self.w["inflation"] / res))
        from scipy.ndimage import binary_dilation
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        inflated = binary_dilation(occ, structure=(xx * xx + yy * yy) <= r * r)
        s = (int((start[0] - ox) / res), int((start[1] - oy) / res))
        t = (int((goal[0] - ox) / res), int((goal[1] - oy) / res))
        h, w = occ.shape
        if not (0 <= t[0] < w and 0 <= t[1] < h) or inflated[t[1], t[0]]:
            return None
        inflated[s[1], s[0]] = False
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == t:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (cur[0] + dx, cur[1] + dy)
                if not (0 <= n[0] < w and 0 <= n[1] < h) or inflated[n[1], n[0]]:
                    continue
                nc = cost[cur] + math.hypot(dx, dy)
                if n not in cost or nc < cost[n]:
                    cost[n], came[n] = nc, cur
                    heapq.heappush(openq, (nc + math.hypot(t[0] - n[0], t[1] - n[1]), n))
        if t not in came:
            return None
        path, c = [], t
        while c is not None:
            path.append((ox + (c[0] + 0.5) * res, oy + (c[1] + 0.5) * res))
            c = came[c]
        return path[::-1]

    def nav_execute(self, gh):
        goal = gh.request.pose.pose
        gx, gy = goal.position.x, goal.position.y
        gyaw = yaw_from_quat(goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w)
        res = NavigateToPose.Result()
        with self.lock:
            start = (self.x, self.y)
        path = self.plan(start, (gx, gy))
        if path is None:
            self.get_logger().info("fake nav2: no path")
            gh.abort()
            return res
        i, t0 = 0, time.monotonic()
        while rclpy.ok():
            if gh.is_cancel_requested:
                self.pub_nav.publish(Twist())
                gh.canceled()
                return res
            with self.lock:
                x, y, yaw = self.x, self.y, self.yaw
            dist = math.hypot(gx - x, gy - y)
            fb = NavigateToPose.Feedback()
            fb.distance_remaining = float(dist)
            gh.publish_feedback(fb)
            cmd = Twist()
            if dist < 0.15:
                err = wrap_pi(gyaw - yaw)
                if abs(err) < 0.15:
                    self.pub_nav.publish(Twist())
                    gh.succeed()
                    return res
                cmd.angular.z = float(np.clip(1.2 * err, -0.5, 0.5))
            else:
                while i < len(path) - 1 and math.hypot(path[i][0] - x, path[i][1] - y) < 0.6:
                    i += 1
                tx, ty = path[i] if i < len(path) - 1 else (gx, gy)
                err = wrap_pi(math.atan2(ty - y, tx - x) - yaw)
                cmd.angular.z = float(np.clip(1.5 * err, -0.6, 0.6))
                if abs(err) < 0.6:
                    cmd.linear.x = float(min(self.w["nav_speed"], 0.25 + 0.6 * dist) * (1 - abs(err) / 0.8))
            self.pub_nav.publish(cmd)
            time.sleep(0.1)
        return res


def main():
    rclpy.init()
    node = FakeRobot()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
