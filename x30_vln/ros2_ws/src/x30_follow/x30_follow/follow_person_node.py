"""follow_person node: tracker + target lock + velocity controller behind the Navigate action /x30/follow_person/run.

mode_manager (mode `follow_person`) delegates here. Flow:
  ARMED      the goal is accepted and the node waits (no motion) until the operator calls the service /follow/lock while the person stands in front
             of the robot (nearest confirmed person within acquire_range_m and +-acquire_fov_deg/2, or a given track_id). The lock stores the
             person's appearance signature (colour histograms of the detector crop), position and velocity. params_json {"auto_lock": true} locks
             the nearest person as soon as one is confirmed (tests only).
  FOLLOW     every cycle target_lock decides which track is the locked person (position continuity + appearance) and the controller of
             controller.py drives /cmd_vel_nav (-> mux -> safety guard) to keep d_target, never closer than d_min.
  AMBIGUOUS  two or more people are plausible and not separable (e.g. one in the same outfit next to the target): hold still (zero velocity), never
             guess; resolves by itself when they separate; after ambiguous_timeout_s the task ends with stop_reason no_progress.
  LOST       nobody plausible: stop (turn toward the last bearing for a few seconds), the lock keeps looking with a growing gate, give up after
             lost_timeout_s (no_progress).
  PLANNED    (planner_follow) while the straight line to the person crosses an obstacle (she went through a door, around a corner) the node does not
             drive at her: it hands the planner (navigate_to_pose: known_map_nav in the sim, Nav2 on the robot) a goal on HER TRAIL d_target behind her and
             re-sends it as she moves; out of sight it sends the planner to the last place she was seen. Direct pursuit resumes when the line is free.
  The task ends when the goal's timeout_s elapses (success, finished) or on cancel. /follow/release drops the lock and returns to ARMED.
Services (x30_msgs/srv/Json): /follow/lock {"track_id": optional}, /follow/release {}. Topic /follow/status (JSON, about 5 Hz). Result message: JSON metrics.
"""
from __future__ import annotations

import dataclasses
import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from x30_msgs.action import Navigate
from x30_msgs.msg import Detection3DArray
from x30_msgs.srv import Json

from . import appearance
from .controller import FollowParams, command
from .lock import LockParams, TargetLock
from .tracker import Det, PersonTracker, TrackerParams
from .trail import Breadcrumbs, GridMap


class FollowPerson(Node):
    def __init__(self):
        super().__init__('follow_person')
        d = self.declare_parameter
        d('map_frame', 'map'), d('base_frame', 'base_link'), d('cmd_topic', '/cmd_vel_nav'), d('detections_topic', '/perception/detections')
        d('control_hz', 20.0), d('lock_wait_s', 600.0), d('acquire_range_m', 4.5), d('acquire_fov_deg', 120.0), d('use_appearance', True)
        d('planner_follow', True), d('nav_action', 'navigate_to_pose'), d('map_topic', '/map'), d('block_hold_s', 0.4), d('unblock_hold_s', 1.0)
        d('lost_standoff_m', 0.5), d('replan_period_s', 1.5), d('goal_move_m', 0.7), d('trail_step_m', 0.3)
        d('lost_after_s', 2.0), d('lost_timeout_s', 12.0), d('ambiguous_timeout_s', 15.0), d('scan_turn_s', 4.0), d('scan_wz', 0.3)
        for f in dataclasses.fields(FollowParams):
            d('ctrl.' + f.name, f.default)
        for f in dataclasses.fields(TrackerParams):
            d('trk.' + f.name, f.default)
        for f in dataclasses.fields(LockParams):
            d('lock.' + f.name, f.default)
        g = self.get_parameter
        keys = ('map_frame', 'base_frame', 'cmd_topic', 'detections_topic', 'control_hz', 'lock_wait_s', 'acquire_range_m', 'acquire_fov_deg',
                'use_appearance', 'planner_follow', 'nav_action', 'map_topic', 'block_hold_s', 'unblock_hold_s', 'replan_period_s', 'goal_move_m',
                'trail_step_m', 'lost_standoff_m', 'lost_after_s', 'lost_timeout_s', 'ambiguous_timeout_s', 'scan_turn_s', 'scan_wz')
        self.P = {k: g(k).value for k in keys}
        self.cp = FollowParams(**{f.name: type(f.default)(g('ctrl.' + f.name).value) for f in dataclasses.fields(FollowParams)})
        self.tp = TrackerParams(**{f.name: type(f.default)(g('trk.' + f.name).value) for f in dataclasses.fields(TrackerParams)})
        self.lp = LockParams(**{f.name: type(f.default)(g('lock.' + f.name).value) for f in dataclasses.fields(LockParams)})
        self.tracker = PersonTracker(self.tp)
        self.lock = TargetLock(self.lp)
        self.mx = threading.Lock()                          # tracker + lock are touched by the detection callback and the control loop
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.pub = self.create_publisher(Twist, self.P['cmd_topic'], 10)
        self.status_pub = self.create_publisher(String, '/follow/status', 10)
        self.create_subscription(Detection3DArray, self.P['detections_topic'], self.on_dets, 10)
        self.cb = ReentrantCallbackGroup()
        self.grid = None                                    # /map in the map frame (inflated in known-map mode): line of sight to the person
        self.crumbs = Breadcrumbs(float(self.P['trail_step_m']))
        self.nav = ActionClient(self, NavigateToPose, self.P['nav_action'], callback_group=self.cb)
        self.nav_handle = None
        self.create_subscription(OccupancyGrid, self.P['map_topic'], self.on_map, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE))
        self.create_service(Json, '/follow/lock', self.svc_lock, callback_group=self.cb)
        self.create_service(Json, '/follow/release', self.svc_release, callback_group=self.cb)
        self.gen = 0
        self.state = 'idle'
        self.last_status = -1e9
        self.det_stats = [0, 0.0, 0, 0]                     # frames, ms in on_dets, persons, descriptors computed
        self.server = ActionServer(self, Navigate, '/x30/follow_person/run', self.execute, goal_callback=self.on_goal,
                                   cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.cb)
        self.get_logger().info(f'follow_person up: target {self.cp.d_target} m, floor {self.cp.d_min} m, appearance {self.P["use_appearance"]}')

    # ------------------------------------------------------------------ inputs
    def on_dets(self, m: Detection3DArray) -> None:
        t0 = time.perf_counter()
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.mx:                                       # descriptors are computed only for people near the locked target (all of them while armed)
            near = (self.lock.search_region(t)[0], self.lock.search_region(t)[1] + 0.5) if self.lock.locked else None
        dets, n_sig = [], 0
        for d in m.detections:
            if d.label != 'person' or d.bearing_only:
                continue
            sig = None
            if self.P['use_appearance'] and len(d.crop.data) and (near is None or math.hypot(d.centroid.x - near[0][0], d.centroid.y - near[0][1]) <= near[1]):
                sig = appearance.signature(bytes(d.crop.data))
                n_sig += 1
            dets.append(Det(d.centroid.x, d.centroid.y, float(d.score), int(d.point_count), t, sig))
        with self.mx:
            self.tracker.update(t, dets)
        self.det_stats[0] += 1
        self.det_stats[1] += (time.perf_counter() - t0) * 1000.0
        self.det_stats[2] += len(dets)
        self.det_stats[3] += n_sig

    def on_map(self, m: OccupancyGrid) -> None:
        self.grid = GridMap(np.array(m.data, np.int8).reshape(m.info.height, m.info.width), float(m.info.resolution),
                            m.info.origin.position.x, m.info.origin.position.y)

    def send_nav_goal(self, x: float, y: float, yaw: float) -> bool:
        """Goal for the planner in the map frame; a newer goal pre-empts the running one (known_map_nav) so it can be re-sent while she walks."""
        if not self.nav.server_is_ready():
            return False
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = self.P['map_frame']
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = float(x), float(y)
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        fut = self.nav.send_goal_async(g)
        fut.add_done_callback(lambda f: setattr(self, 'nav_handle', f.result()))
        return True

    def cancel_nav(self) -> None:
        h, self.nav_handle = self.nav_handle, None
        if h is not None:
            h.cancel_goal_async()

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def base_pose(self):
        try:
            tf = self.buf.lookup_transform(self.P['map_frame'], self.P['base_frame'], rclpy.time.Time())
        except Exception:
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    @staticmethod
    def to_base(pose, x, y):
        dx, dy = x - pose[0], y - pose[1]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return c * dx + s * dy, -s * dx + c * dy

    # ------------------------------------------------------------------ lock services
    def pick_in_front(self, pose, t):
        """(range, bearing, track) of the nearest confirmed person in front, or None. Caller holds self.mx."""
        best = None
        for tr in self.tracker.confirmed():
            if t - tr.last_t > 1.5:
                continue
            px, py = self.to_base(pose, *self.tracker.predict(tr, t))
            rng, brg = math.hypot(px, py), math.atan2(py, px)
            if rng <= self.P['acquire_range_m'] and abs(brg) <= math.radians(self.P['acquire_fov_deg']) / 2 and (best is None or rng < best[0]):
                best = (rng, brg, tr)
        return best

    def try_lock(self, track_id=None) -> dict:
        t, pose = self.now(), self.base_pose()
        if pose is None:
            return {'ok': False, 'message': 'no robot pose (TF map -> base_link missing)'}
        with self.mx:
            if track_id is not None:
                tr = self.tracker.get(int(track_id))
                if tr is None:
                    return {'ok': False, 'message': f'no track {track_id}'}
                px, py = self.to_base(pose, *self.tracker.predict(tr, t))
                rng, brg = math.hypot(px, py), math.atan2(py, px)
            else:
                best = self.pick_in_front(pose, t)
                if best is None:
                    return {'ok': False, 'message': f'no confirmed person in front within {self.P["acquire_range_m"]} m'}
                rng, brg, tr = best
            self.lock.lock(tr, t)
            return {'ok': True, 'track_id': tr.id, 'range_m': round(rng, 2), 'bearing_deg': round(math.degrees(brg), 1),
                    'has_signature': tr.sig is not None, 'n_tracks': len(self.tracker.tracks),
                    'message': 'locked' + ('' if tr.sig is not None else ' (no appearance signature: position only)')}

    def svc_lock(self, req, resp):
        try:
            body = json.loads(req.request) if req.request else {}
        except ValueError:
            body = {}
        out = self.try_lock(body.get('track_id'))
        resp.ok, resp.response = bool(out['ok']), json.dumps(out)
        self.get_logger().info(f'lock: {out}')
        return resp

    def svc_release(self, _req, resp):
        with self.mx:
            self.lock.release()
        resp.ok, resp.response = True, json.dumps({'ok': True, 'message': 'released'})
        return resp

    # ------------------------------------------------------------------ action
    def on_goal(self, _request):
        self.gen += 1                                       # a new goal pre-empts a running one
        return GoalResponse.ACCEPT

    def stop(self) -> None:
        self.pub.publish(Twist())

    def publish_status(self, **kw) -> None:
        t = self.now()
        if t - self.last_status < 0.2:
            return
        self.last_status = t
        self.status_pub.publish(String(data=json.dumps({'t': round(t, 2), 'state': self.state, 'n_tracks': len(self.tracker.tracks),
                                                        'locked': self.lock.locked, **kw})))

    def execute(self, handle):
        my_gen = self.gen
        goal = handle.request
        res = Navigate.Result()
        try:
            over = json.loads(goal.params_json) if goal.params_json else {}
        except ValueError:
            over = {}
        cp = dataclasses.replace(self.cp, **{k: type(getattr(self.cp, k))(v) for k, v in over.items() if hasattr(self.cp, k)})
        timeout = float(goal.timeout_s) if goal.timeout_s > 0 else 120.0
        auto_lock = bool(over.get('auto_lock', False))
        t_start = self.now()
        dt_cmd = 1.0 / float(self.P['control_hz'])
        m = {'tracked_s': 0.0, 'total_s': 0.0, 'lost_events': 0, 'ambiguous_events': 0, 'ambiguous_s': 0.0, 'min_range': math.inf,
             'abs_err_sum': 0.0, 'err_n': 0, 'path_m': 0.0}
        fb = Navigate.Feedback()

        def feedback(status, rng=math.nan):
            fb.status, fb.current_skill, fb.distance_to_goal, fb.agent_turn = status, 'follow_person', float(rng), 0
            handle.publish_feedback(fb)

        def finish(ok, reason, msg):
            self.cancel_nav()
            self.stop()
            self.state = 'idle'
            res.success, res.stop_reason = ok, reason
            n_f = max(1, self.det_stats[0])
            m.update({'det_ms_mean': self.det_stats[1] / n_f, 'persons_per_frame': self.det_stats[2] / n_f, 'descriptors_per_frame': self.det_stats[3] / n_f})
            res.message = msg + ' ' + json.dumps({k: (round(v, 3) if isinstance(v, float) and math.isfinite(v) else v) for k, v in m.items()})
            res.path_length_m, res.duration_s, res.agent_turns = float(m['path_m']), float(self.now() - t_start), 0
            (handle.succeed if ok else handle.abort)()
            return res

        def interrupted():
            if my_gen != self.gen:
                self.state = 'idle'
                handle.abort()
                return True
            if handle.is_cancel_requested:
                self.cancel_nav()
                self.stop()
                self.state = 'idle'
                handle.canceled()
                return True
            return False

        with self.mx:
            self.lock.release()                              # every goal starts armed with no lock
        self.det_stats = [0, 0.0, 0, 0]
        prev = (0.0, 0.0, 0.0)
        last_cmd_t = self.now()
        last_brg = 0.0
        amb_since = lost_since = None
        prev_pose = None
        self.crumbs.clear()
        mode, blocked_since, clear_since, goal_xy, goal_t = 'direct', None, None, None, -1e9
        use_plan = bool(over.get('planner_follow', self.P['planner_follow']))      # params_json {"planner_follow": false} = old behaviour (tests)
        m['planned_s'] = 0.0
        m['planner_goals'] = 0
        self.state = 'armed'
        while True:
            if interrupted():
                return res
            t = self.now()
            if t - t_start > timeout:
                return finish(self.state not in ('armed',), 'finished' if self.state != 'armed' else 'no_progress',
                              'follow duration complete.' if self.state != 'armed' else 'no lock command arrived.')
            if self.state == 'armed':
                if auto_lock:
                    self.try_lock()
                if self.lock.locked:
                    self.state = 'follow'
                    prev = (0.0, 0.0, 0.0)
                    last_cmd_t = t
                    self.get_logger().info(f'locked on track {self.lock.track_id}')
                    continue
                if t - t_start > float(self.P['lock_wait_s']):
                    return finish(False, 'no_progress', 'no lock command arrived.')
                feedback('armed: waiting for /follow/lock')
                self.publish_status(target=None)
                time.sleep(0.05)
                continue
            if not self.lock.locked:                         # released by the operator
                self.stop()
                self.state = 'armed'
                continue
            if t - last_cmd_t < dt_cmd:
                time.sleep(0.01)
                continue
            dt = t - last_cmd_t
            last_cmd_t = t
            pose = self.base_pose()
            if pose is None:
                continue
            if prev_pose is not None:
                m['path_m'] += math.hypot(pose[0] - prev_pose[0], pose[1] - prev_pose[1])
            prev_pose = pose
            m['total_s'] += dt
            with self.mx:
                dec = self.lock.select(self.tracker, t)
                tr = dec.track
                pred = self.tracker.predict(tr, t) if tr is not None else None
            if dec.state == 'ok' and tr is not None:
                if self.state != 'follow':
                    self.get_logger().info(f'target visible again (track {tr.id})')
                self.state, amb_since, lost_since = 'follow', None, None
                px, py = self.to_base(pose, *pred)
                rng, brg = math.hypot(px, py), math.atan2(py, px)
                last_brg = brg
                if t - tr.last_t < 0.6:
                    self.crumbs.add(tr.x, tr.y)                  # the route she walked: passable by construction
                blocked = use_plan and self.grid is not None and not self.grid.line_free((pose[0], pose[1]), pred)
                if blocked:
                    blocked_since, clear_since = blocked_since if blocked_since is not None else t, None
                else:
                    blocked_since, clear_since = None, clear_since if clear_since is not None else t
                if mode == 'direct' and blocked_since is not None and t - blocked_since >= float(self.P['block_hold_s']):
                    mode = 'planned'
                    self.get_logger().info('straight line to the person is blocked: following her trail with the planner')
                elif mode == 'planned' and clear_since is not None and t - clear_since >= float(self.P['unblock_hold_s']):
                    mode, goal_xy = 'direct', None
                    self.cancel_nav()
                    prev = (0.0, 0.0, 0.0)
                    self.get_logger().info('line to the person is free again: direct pursuit')
                vx = vy = wz = 0.0
                if mode == 'planned':
                    m['planned_s'] += dt
                    if rng < cp.d_min + 0.3:                     # too close (she turned back toward the robot): the planner steps aside, the controller backs off
                        if goal_xy is not None:
                            self.cancel_nav()
                            goal_xy = None
                        c, s = math.cos(pose[2]), math.sin(pose[2])
                        vx, vy, wz = command(px, py, c * tr.vx + s * tr.vy, -s * tr.vx + c * tr.vy, cp, prev, dt)
                        prev = (vx, vy, wz)
                        tw = Twist()
                        tw.linear.x, tw.linear.y, tw.angular.z = min(vx, 0.0), vy, wz
                        self.pub.publish(tw)
                    else:
                        gx, gy = self.crumbs.point_behind((tr.x, tr.y), cp.d_target) or pred
                        k = 0
                        while self.grid is not None and not self.grid.free(gx, gy) and k < 8:     # not into an obstacle: walk the goal toward her along the trail
                            k += 1
                            gx, gy = self.crumbs.point_behind((tr.x, tr.y), max(0.0, cp.d_target - 0.3 * k)) or pred
                        moved = goal_xy is None or math.hypot(gx - goal_xy[0], gy - goal_xy[1]) > float(self.P['goal_move_m'])
                        if (moved and t - goal_t >= 0.5) or t - goal_t > float(self.P['replan_period_s']):
                            if self.send_nav_goal(gx, gy, math.atan2(pred[1] - gy, pred[0] - gx)):
                                goal_xy, goal_t = (gx, gy), t
                                m['planner_goals'] += 1
                else:
                    if goal_xy is not None:                      # a goal left over from LOST / PLANNED must not fight the controller
                        self.cancel_nav()
                        goal_xy = None
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    vx, vy, wz = command(px, py, c * tr.vx + s * tr.vy, -s * tr.vx + c * tr.vy, cp, prev, dt)
                    prev = (vx, vy, wz)
                    tw = Twist()
                    tw.linear.x, tw.linear.y, tw.angular.z = vx, vy, wz
                    self.pub.publish(tw)
                m['tracked_s'] += dt
                m['min_range'] = min(m['min_range'], rng)
                m['abs_err_sum'] += abs(rng - cp.d_target)
                m['err_n'] += 1
                self.publish_status(target=tr.id, range=round(rng, 2), bearing=round(math.degrees(brg), 1), x=round(pred[0], 2), y=round(pred[1], 2),
                                    vx=round(tr.vx, 2), vy=round(tr.vy, 2), cmd=[round(vx, 2), round(wz, 2)], cost=round(dec.cost, 2), mode=mode,
                                    goal=None if goal_xy is None else [round(goal_xy[0], 2), round(goal_xy[1], 2)],
                                    second=None if not math.isfinite(dec.second_cost) else round(dec.second_cost, 2))
                feedback('following', rng)
                continue
            prev = (0.0, 0.0, 0.0)
            if dec.state == 'ambiguous':
                if amb_since is None:
                    amb_since, self.state = t, 'ambiguous'
                    m['ambiguous_events'] += 1
                    self.get_logger().warn(f'target ambiguous ({dec.candidates} candidates): holding still')
                m['ambiguous_s'] += dt
                if goal_xy is not None:
                    self.cancel_nav()
                    goal_xy = None
                self.pub.publish(Twist())                    # hold: never guess between look-alikes
                with self.mx:
                    why = self.lock.explain(self.tracker, t)
                self.publish_status(candidates=dec.candidates, cost=round(dec.cost, 2), second=round(dec.second_cost, 2), why=why)
                feedback('ambiguous: holding')
                if t - amb_since > float(self.P['ambiguous_timeout_s']):
                    return finish(False, 'no_progress', 'target stayed ambiguous.')
                continue
            # lost
            if lost_since is None:
                lost_since, self.state = t, 'lost'
                m['lost_events'] += 1
                self.get_logger().warn('target lost')
                px, py = self.to_base(pose, *self.lock.predicted(t))
                last_brg = math.atan2(py, px)
            last_seen = self.crumbs.last() if use_plan else None
            if last_seen is not None:                    # go to (almost) where she was last seen: she is out of sight, so the 2 m floor between people does not apply
                head = last_seen                         # (d_target short of that spot was where the robot already stood: it never went around the corner, corner2 A/B)
                last_seen = self.crumbs.point_behind(head, float(self.P['lost_standoff_m'])) or head
                k = 0
                while self.grid is not None and not self.grid.free(*last_seen) and k < 6:
                    k += 1
                    last_seen = self.crumbs.point_behind(head, float(self.P['lost_standoff_m']) + 0.3 * k) or head
            if last_seen is not None and (goal_xy is None or math.hypot(last_seen[0] - goal_xy[0], last_seen[1] - goal_xy[1]) > 0.5):
                if self.send_nav_goal(last_seen[0], last_seen[1], math.atan2(last_seen[1] - pose[1], last_seen[0] - pose[0])):
                    goal_xy, goal_t = last_seen, t        # go to where she was last seen (around the corner), the lock finds her again from there
                    m['planner_goals'] += 1
            if goal_xy is None or last_seen is None:      # no planner goal: turn toward where she was last seen
                tw = Twist()
                if t - lost_since <= float(self.P['scan_turn_s']) and abs(last_brg) > math.radians(20):
                    tw.angular.z = math.copysign(float(self.P['scan_wz']), last_brg)
                self.pub.publish(tw)
            with self.mx:
                why = self.lock.explain(self.tracker, t)
            self.publish_status(lost_for=round(t - lost_since, 1), why=why)
            feedback('lost')
            if t - lost_since > float(self.P['lost_timeout_s']):
                return finish(False, 'no_progress', 'lost the person.')


def main(args=None):
    rclpy.init(args=args)
    node = FollowPerson()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
