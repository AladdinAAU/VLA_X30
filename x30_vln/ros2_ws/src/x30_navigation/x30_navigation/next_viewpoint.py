"""Coverage viewpoints for a fully known grid: service /nav/next_frontier (same JSON as frontier_explorer).

Known-map mode has no frontiers, so "unexplored" means "not yet seen by the camera". The node tracks which free cells of /map
fell inside the camera frustum (horizontal FOV from camera_info, range `view_range_m`, occlusion by ray casting on the grid) and, on
request, returns the nearest reachable free cell (BFS path length) that is unseen and whose surroundings hold at least
`min_unseen_cells` unseen free cells. The yaw points at the centroid of those cells.

Request : {"strategy": "nearest|...", "bias": [x, y] | null, "exclude": [[x, y], ...]}   (strategy/bias accepted; nearest is used)
Response: {"found", "goal": {"x","y","yaw"}, "size": unseen cells around, "distance_m", "remaining", "reason"}
Reset the seen set with the service /nav/next_viewpoint/reset.
"""
import json
import math
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener
from x30_msgs.srv import Json


class NextViewpoint(Node):
    def __init__(self):
        super().__init__("next_viewpoint")
        self.declare_parameter("view_range_m", 4.0)
        self.declare_parameter("default_hfov_deg", 90.0)
        self.declare_parameter("update_hz", 3.0)
        self.declare_parameter("min_unseen_cells", 25)
        self.declare_parameter("lattice_m", 0.6)
        self.declare_parameter("exclude_radius_m", 0.5)
        self.declare_parameter("visited_radius_m", 1.2)
        self.declare_parameter("base_frame", "base_link")
        self.range = self.get_parameter("view_range_m").value
        self.hfov = math.radians(self.get_parameter("default_hfov_deg").value)
        self.grid = False
        self.seen = None
        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, latched)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, 1)
        self.create_service(Json, "/nav/next_frontier", self.on_request)
        self.create_service(Json, "/nav/next_viewpoint/reset", self.on_reset)
        self.create_timer(1.0 / self.get_parameter("update_hz").value, self.update_seen)

    def on_info(self, msg):
        if msg.k[0] > 0 and msg.width > 0:
            self.hfov = 2.0 * math.atan(msg.width / (2.0 * msg.k[0]))

    def on_map(self, msg):
        arr = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.free = arr == 0
        self.occ = arr > 0
        self.res = msg.info.resolution
        self.ox, self.oy = msg.info.origin.position.x, msg.info.origin.position.y
        q = msg.info.origin.orientation
        self.oyaw = 2.0 * math.atan2(q.z, q.w)
        if self.seen is None or self.seen.shape != self.free.shape:
            self.seen = np.zeros_like(self.free)
        self.grid = True
        self.get_logger().info(f"map {arr.shape[1]}x{arr.shape[0]} @ {self.res} m, free cells {int(self.free.sum())}")

    def on_reset(self, req, resp):
        if self.seen is not None:
            self.seen[:] = False
        resp.ok, resp.response = True, json.dumps({"ok": True})
        return resp

    def pose(self):
        try:
            t = self.tfb.lookup_transform("map", self.get_parameter("base_frame").value, rclpy.time.Time())
        except Exception:
            return None
        q = t.transform.rotation
        return t.transform.translation.x, t.transform.translation.y, 2.0 * math.atan2(q.z, q.w)

    def w2c(self, x, y):
        dx, dy = x - self.ox, y - self.oy
        c, s = math.cos(-self.oyaw), math.sin(-self.oyaw)
        return (c * dx - s * dy) / self.res, (s * dx + c * dy) / self.res   # col, row (float)

    def c2w(self, col, row):
        px, py = col * self.res, row * self.res
        c, s = math.cos(self.oyaw), math.sin(self.oyaw)
        return self.ox + c * px - s * py, self.oy + s * px + c * py

    def update_seen(self):
        p = self.pose() if self.grid else None
        if p is None:
            return
        x, y, yaw = p
        n = max(int(self.hfov / math.radians(1.5)), 8)
        steps = int(self.range / (self.res * 0.7))
        h, w = self.free.shape
        # the cells under and around the robot count as visited (it stood there and scans turn it around)
        c0, r0 = self.w2c(x, y)
        rad = self.get_parameter("visited_radius_m").value / self.res
        rr, cc = np.ogrid[:h, :w]
        self.seen |= (rr - r0) ** 2 + (cc - c0) ** 2 <= rad ** 2
        for k in range(n + 1):
            a = yaw - self.hfov / 2.0 + self.hfov * k / n
            for i in range(1, steps + 1):
                d = i * self.res * 0.7
                col, row = self.w2c(x + d * math.cos(a), y + d * math.sin(a))
                ci, ri = int(round(col)), int(round(row))
                if not (0 <= ci < w and 0 <= ri < h) or self.occ[ri, ci]:
                    break
                self.seen[ri, ci] = True

    def on_request(self, req, resp):
        try:
            q = json.loads(req.request) if req.request.strip() else {}
        except json.JSONDecodeError as exc:
            resp.ok, resp.response = False, json.dumps({"found": False, "reason": f"bad json: {exc}"})
            return resp
        resp.ok, resp.response = True, json.dumps(self.find(q.get("exclude") or []))
        return resp

    def find(self, exclude):
        p = self.pose() if self.grid else None
        if p is None:
            return {"found": False, "reason": "no map or no pose"}
        x, y, _ = p
        h, w = self.free.shape
        c0, r0 = self.w2c(x, y)
        c0, r0 = int(round(c0)), int(round(r0))
        if not (0 <= c0 < w and 0 <= r0 < h) or not self.free[r0, c0]:
            ra, ca = max(r0 - 6, 0), max(c0 - 6, 0)
            near = np.argwhere(self.free[ra:r0 + 7, ca:c0 + 7])
            if len(near) == 0:
                return {"found": False, "reason": "robot is not on a free cell"}
            r0, c0 = int(near[0][0]) + ra, int(near[0][1]) + ca
        # BFS over free cells (8-connected) from the robot; the free mask is the inflated grid, i.e. safe to stand on
        dist = np.full((h, w), -1.0, dtype=np.float32)
        dist[r0, c0] = 0.0
        dq = deque([(r0, c0)])
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] < 0 and self.free[rr, cc]:
                    dist[rr, cc] = dist[r, c] + (1.414 if dr and dc else 1.0)
                    dq.append((rr, cc))
        unseen = self.free & ~self.seen & (dist >= 0)
        k = int(self.range / self.res)
        ii = np.pad(unseen.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        step = max(int(self.get_parameter("lattice_m").value / self.res), 1)
        min_n = self.get_parameter("min_unseen_cells").value
        ex_r = self.get_parameter("exclude_radius_m").value
        best, remaining = None, 0
        for r in range(r0 % step, h, step):
            for c in range(c0 % step, w, step):
                if dist[r, c] < 0.0 or not unseen[r, c]:
                    continue
                ra, rb, ca, cb = max(r - k, 0), min(r + k + 1, h), max(c - k, 0), min(c + k + 1, w)
                n = int(ii[rb, cb] - ii[ra, cb] - ii[rb, ca] + ii[ra, ca])
                if n < min_n:
                    continue
                wx, wy = self.c2w(c, r)
                if any(math.hypot(wx - e[0], wy - e[1]) < ex_r for e in exclude):
                    continue
                remaining += 1
                if best is None or dist[r, c] < best[0]:
                    best = (float(dist[r, c]), r, c, n, wx, wy)
        if best is None:
            return {"found": False, "remaining": 0, "reason": "all reachable free space has been seen"}
        d, r, c, n, wx, wy = best
        ra, rb, ca, cb = max(r - k, 0), min(r + k + 1, h), max(c - k, 0), min(c + k + 1, w)
        rs, cs = np.nonzero(unseen[ra:rb, ca:cb])
        gx, gy = self.c2w(ca + cs.mean(), ra + rs.mean())
        return {"found": True, "goal": {"x": wx, "y": wy, "yaw": math.atan2(gy - wy, gx - wx)}, "size": n,
                "distance_m": d * self.res, "remaining": remaining, "reason": ""}


def main():
    rclpy.init()
    node = NextViewpoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
