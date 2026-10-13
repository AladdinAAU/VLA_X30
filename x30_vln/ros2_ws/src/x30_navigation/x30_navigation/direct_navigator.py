"""direct_navigator: Nav2-free `navigate_to_pose` action server (planner_backend:=direct).

Plans a path on /map (grid Dijkstra, direct_nav.plan_path; straight line when there is no map or no path), follows it with a
pure-pursuit law on the active odometry (TF map -> base_link) and publishes Twist on /cmd_vel_nav. No obstacle avoidance of its
own: the safety guard stops the robot in front of obstacles; a goal that makes no progress for `stuck_timeout_s` is aborted.
Replans every `replan_period_s` on the newest map.
"""
from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from .direct_nav import TrackParams, plan_path, straight_line, track
from .geometry import wrap_angle


class DirectNavigator(Node):
    def __init__(self) -> None:
        super().__init__("direct_navigator")
        d = self.declare_parameter
        d("cmd_topic", "/cmd_vel_nav")
        d("map_topic", "/map")
        d("map_frame", "map")
        d("base_frame", "base_link")
        d("rate_hz", 10.0)
        d("goal_radius", 0.3)             # m; success when closer (XY)
        d("yaw_tolerance", 0.3)           # rad, only when the goal yaw is aligned (align_yaw)
        d("align_yaw", False)             # turn to the goal yaw after arriving
        d("robot_radius", 0.42)           # m, path clearance on the map
        d("use_unknown_space", True)      # second planning attempt treats unknown cells as free
        d("replan_period_s", 2.0)
        d("stuck_timeout_s", 40.0)        # no progress of 0.15 m in this time: abort
        d("wedge_window_s", 3.0)          # neither 0.03 m nor 0.05 rad of motion in this window while commanding: back up
        d("backup_m", 0.4)
        d("timeout_s", 180.0)
        for k, v in vars(TrackParams()).items():
            d("track." + k, v)
        g = self.get_parameter
        self.p = {k: g(k).value for k in ("cmd_topic", "map_frame", "base_frame", "rate_hz", "goal_radius", "yaw_tolerance", "align_yaw",
                                          "robot_radius", "wedge_window_s", "backup_m", "use_unknown_space", "replan_period_s", "stuck_timeout_s", "timeout_s")}
        self.tp = TrackParams(**{k: type(v)(g("track." + k).value) for k, v in vars(TrackParams()).items()})
        self.map: OccupancyGrid | None = None
        self.tf_buffer = Buffer()
        self.tfl = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, g("map_topic").value, lambda m: setattr(self, "map", m), qos)
        self.pub = self.create_publisher(Twist, self.p["cmd_topic"], 10)
        cb = ReentrantCallbackGroup()
        self.server = ActionServer(self, NavigateToPose, "navigate_to_pose", self.execute, goal_callback=lambda _: GoalResponse.ACCEPT,
                                   callback_group=cb)
        self.get_logger().info("direct_navigator ready (no Nav2)")

    def pose(self) -> tuple[float, float, float] | None:
        try:
            t = self.tf_buffer.lookup_transform(self.p["map_frame"], self.p["base_frame"], rclpy.time.Time())
        except Exception:
            return None
        q, v = t.transform.rotation, t.transform.translation
        return v.x, v.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def plan(self, pose, goal_xy) -> np.ndarray:
        m = self.map
        if m is not None:
            grid = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
            org = np.array([m.info.origin.position.x, m.info.origin.position.y])
            for unknown in ((False, True) if self.p["use_unknown_space"] else (False,)):
                for rad in (self.p["robot_radius"], 0.3):          # full clearance first, then squeeze through narrow doors
                    path = plan_path(grid, m.info.resolution, org, pose[:2], goal_xy, rad, unknown)
                    if path is not None:
                        return path
        return straight_line(pose[:2], goal_xy)

    def stop(self) -> None:
        self.pub.publish(Twist())

    def execute(self, handle):
        goal = handle.request.pose.pose
        gxy = np.array([goal.position.x, goal.position.y])
        gyaw = math.atan2(2 * (goal.orientation.w * goal.orientation.z), 1 - 2 * goal.orientation.z ** 2)
        result = NavigateToPose.Result()
        period = 1.0 / self.p["rate_hz"]
        t0 = last_plan = last_prog = time.monotonic()
        pose = None
        while pose is None and time.monotonic() - t0 < 5.0:
            pose = self.pose()
            time.sleep(0.1)
        if pose is None:
            handle.abort()
            return result
        path, anchor = self.plan(pose, gxy), np.array(pose[:2])
        win = (time.monotonic(), np.array(pose[:2]), pose[2])
        backing: np.ndarray | None = None
        back_t0 = 0.0
        while rclpy.ok():
            if handle.is_cancel_requested:
                self.stop()
                handle.canceled()
                return result
            now = time.monotonic()
            pose = self.pose()
            if pose is None:
                time.sleep(period)
                continue
            if now - t0 > self.p["timeout_s"]:
                break
            if np.linalg.norm(np.array(pose[:2]) - anchor) > 0.15:
                anchor, last_prog = np.array(pose[:2]), now
            elif now - last_prog > self.p["stuck_timeout_s"]:
                self.get_logger().warn("no progress, aborting goal")
                break
            if backing is not None:                      # wedge recovery: reverse straight out
                cmd = Twist()
                cmd.linear.x = -0.25
                self.pub.publish(cmd)
                if np.linalg.norm(np.array(pose[:2]) - backing) > self.p["backup_m"] or now - back_t0 > 12.0:
                    backing, win = None, (now, np.array(pose[:2]), pose[2])
                    path, last_plan = self.plan(pose, gxy), now
                time.sleep(period)
                continue
            if now - win[0] > self.p["wedge_window_s"]:
                moved = np.linalg.norm(np.array(pose[:2]) - win[1]) > 0.03 or abs(wrap_angle(pose[2] - win[2])) > 0.05
                if not moved and np.linalg.norm(np.array(pose[:2]) - gxy) > self.p["goal_radius"]:
                    self.get_logger().info("wedged: backing up")
                    backing, back_t0 = np.array(pose[:2]), now
                    continue
                win = (now, np.array(pose[:2]), pose[2])
            if now - last_plan > self.p["replan_period_s"]:
                path, last_plan = self.plan(pose, gxy), now
            vx, vy, wz, dist = track(pose, path, self.tp)
            cmd = Twist()
            if dist < self.p["goal_radius"]:
                err = wrap_angle(gyaw - pose[2])
                if self.p["align_yaw"] and abs(err) > self.p["yaw_tolerance"]:
                    cmd.angular.z = float(np.clip(self.tp.k_yaw * err, -self.tp.wz_max, self.tp.wz_max))
                else:
                    self.stop()
                    handle.succeed()
                    return result
            else:
                cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
            self.pub.publish(cmd)
            time.sleep(period)
        self.stop()
        handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DirectNavigator()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
