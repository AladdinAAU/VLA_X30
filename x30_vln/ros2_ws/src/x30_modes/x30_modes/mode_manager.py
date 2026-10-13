"""mode_manager: the /x30/navigate action, /x30/set_mode service and /x30/mode topic.

A Navigate goal selects a mode and runs a task:
  goal_pose        -> skill goto_pose (Nav2)
  explore          -> skill explore in a loop until the frontiers are exhausted
  velocity_follow  -> delegated to the agent's action /x30/agent/follow  (x30_msgs/action/Navigate)
  object_nav       -> delegated to the agent's action /x30/agent/run     (x30_msgs/action/Navigate)
Around the task run the monitors: timeout, progress watchdog, soft e-stop / long safety stops, teleop takeover. Switching modes or
sending a new goal cancels the running task first, zeroes the velocity, and only then changes the mode.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from x30_msgs.action import Navigate, Skill
from x30_msgs.msg import SafetyStatus
from x30_msgs.srv import SetMode

from .geometry import yaw_from_quat
from .mode_logic import MODES, TASK_MODES, ModeCore, PathLength, ProgressWatchdog, TaskHandle
from .paths import find_configs_dir, load_mode_config
from .ros_io import LATCHED_QOS, lookup_pose, now_s


@dataclass
class Outcome:
    success: bool
    stop_reason: str
    message: str
    agent_turns: int = 0


class ActionRun:
    """One goal on an action client, polled instead of awaited so a monitor loop can supervise it."""

    def __init__(self, client: ActionClient, accept_timeout: float = 5.0):
        self.client, self.accept_timeout = client, accept_timeout
        self._gfut = self._rfut = self._gh = None
        self._t0 = 0.0
        self.error = ""
        self.cancelled = False

    def start(self, goal, feedback_cb=None, server_wait: float = 2.0) -> str:
        if not self.client.wait_for_server(timeout_sec=server_wait):
            self.error = "unavailable"
            return self.error
        self._gfut = self.client.send_goal_async(goal, feedback_callback=feedback_cb)
        self._t0 = time.monotonic()
        return ""

    def poll(self):
        """Result wrapper (with .result and .status) once finished, else None. `.error` is set if the goal never started."""
        if self.error or self._gfut is None:
            return None
        if self._rfut is None:
            if self._gfut.done():
                gh = self._gfut.result()
                if gh is None or not gh.accepted:
                    self.error = "rejected"
                    return None
                self._gh = gh
                self._rfut = gh.get_result_async()
            elif time.monotonic() - self._t0 > self.accept_timeout:
                self.error = "no_response"
                return None
            else:
                return None
        return self._rfut.result() if self._rfut.done() else None

    def cancel(self, wait: float = 6.0) -> bool:
        """Cancel and wait (bounded) for the final result. True if the goal ended."""
        self.cancelled = True
        if self._gfut is None or self.error:
            return True
        t_end = time.monotonic() + wait
        while self._rfut is None and not self._gfut.done() and time.monotonic() < t_end:
            time.sleep(0.02)
        if self._gh is None and self._gfut.done() and self._gfut.result() is not None and self._gfut.result().accepted:
            self._gh = self._gfut.result()
            self._rfut = self._gh.get_result_async()
        if self._gh is None:
            return True
        self._gh.cancel_goal_async()
        while time.monotonic() < t_end:
            if self._rfut is not None and self._rfut.done():
                return True
            time.sleep(0.02)
        return False


class ModeManager(Node):
    def __init__(self):
        super().__init__("mode_manager")
        p = self.declare_parameter
        self.profile = str(p("profile", "sim").value)
        self.odom_topic = str(p("odom_topic", "/odom").value)
        self.map_frame, self.base_frame = str(p("map_frame", "map").value), str(p("base_frame", "base_link").value)
        self.skill_prefix = str(p("skill_prefix", "/x30/skill/").value)
        self.follow_action = str(p("follow_action", "/x30/agent/follow").value)
        self.run_action = str(p("run_action", "/x30/agent/run").value)
        self.person_action = str(p("person_action", "/x30/follow_person/run").value)   # x30_follow node
        cancel_wait = float(p("cancel_wait_s", 6.0).value)
        self.configs_dir = find_configs_dir()
        self.cb = ReentrantCallbackGroup()

        self.core = ModeCore(self.stop_motion, self.publish_mode, cancel_wait)
        self.pub_mode = self.create_publisher(String, "/x30/mode", LATCHED_QOS)
        self.pub_zero = self.create_publisher(Twist, str(p("nav_cmd_topic", "/cmd_vel_nav").value), 10)
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)

        self.pose: tuple[float, float, float] | None = None
        self.path = PathLength()
        self.safety: SafetyStatus | None = None
        self.estop_since: float | None = None
        self.stopped_since: float | None = None
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 20, callback_group=self.cb)
        self.create_subscription(SafetyStatus, "/x30/safety", self.on_safety, 10, callback_group=self.cb)
        self.create_subscription(String, "/x30/cmd_source", self.on_source, 10, callback_group=self.cb)
        self.teleop_cancels = True
        self._fb_state: dict = {"turn": 0, "skill": "", "status": ""}

        self.c_follow = ActionClient(self, Navigate, self.follow_action, callback_group=self.cb)
        self.c_run = ActionClient(self, Navigate, self.run_action, callback_group=self.cb)
        self.c_person = ActionClient(self, Navigate, self.person_action, callback_group=self.cb)
        self.c_skill: dict[str, ActionClient] = {}
        for name in ("goto_pose", "explore"):
            self.c_skill[name] = ActionClient(self, Skill, self.skill_prefix + name, callback_group=self.cb)

        self.create_service(SetMode, "/x30/set_mode", self.on_set_mode, callback_group=self.cb)
        self.server = ActionServer(self, Navigate, "/x30/navigate", execute_callback=self.execute,
                                   goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=self.on_cancel, callback_group=self.cb)
        self.publish_mode("idle")
        self.get_logger().info(f"mode_manager ready (profile {self.profile}, configs {self.configs_dir})")

    # ------------------------------------------------------------ plumbing
    def publish_mode(self, mode: str) -> None:
        self.pub_mode.publish(String(data=mode))
        self.get_logger().info(f"mode: {mode}")

    def stop_motion(self) -> None:
        z = Twist()
        for _ in range(3):
            self.pub_zero.publish(z)
            time.sleep(0.02)

    def on_odom(self, m: Odometry) -> None:
        q = m.pose.pose.orientation
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quat(q.x, q.y, q.z, q.w))
        self.path.update(self.pose[0], self.pose[1])

    def on_safety(self, m: SafetyStatus) -> None:
        self.safety = m
        now = now_s(self)
        self.estop_since = (self.estop_since or now) if (m.stopped and m.reason == "estop") else None
        self.stopped_since = (self.stopped_since or now) if (m.stopped and m.reason in ("obstacle", "no_lidar", "tilt")) else None

    def on_source(self, m: String) -> None:
        if m.data == "teleop" and self.teleop_cancels and self.core.task is not None and not self.core.task.done:
            self.get_logger().warn("teleop took over: cancelling the running task")
            self.core.cancel_current("teleop took over (operator override)", "aborted")

    def on_cancel(self, gh) -> CancelResponse:
        self.core.cancel_current("cancelled by the client", "aborted")
        return CancelResponse.ACCEPT

    def on_set_mode(self, req: SetMode.Request, res: SetMode.Response) -> SetMode.Response:
        ok, msg, prev = self.core.set_mode(req.mode)
        res.ok, res.message, res.previous_mode = ok, msg, prev
        return res

    # ------------------------------------------------------------ task factories
    def goal_pose_args(self, goal, cfg: dict) -> tuple[dict | None, str]:
        ps = goal.goal
        frame = ps.header.frame_id
        if frame not in ("", self.map_frame):
            return None, f"goal_pose expects the goal in the '{self.map_frame}' frame, got '{frame}'"
        q = ps.pose.orientation
        args = {"x": float(ps.pose.position.x), "y": float(ps.pose.position.y)}
        if abs(q.x) + abs(q.y) + abs(q.z) + abs(q.w) > 1e-6:
            args["yaw_deg"] = math.degrees(yaw_from_quat(q.x, q.y, q.z, q.w))
        if "tolerance_m" in cfg:
            args["tolerance_m"] = float(cfg["tolerance_m"])
        return args, ""

    def skill_goal(self, name: str, args: dict, timeout: float) -> Skill.Goal:
        g = Skill.Goal()
        g.name, g.args_json, g.timeout_s = name, json.dumps(args), float(timeout)
        return g

    # ------------------------------------------------------------ Navigate execution
    def execute(self, gh) -> Navigate.Result:
        goal = gh.request
        mode = goal.mode
        result = Navigate.Result()
        if mode not in TASK_MODES:
            gh.abort()
            result.success, result.stop_reason = False, "error"
            result.message = f"Unknown or non-task mode '{mode}'. Valid: {', '.join(TASK_MODES)}."
            return result
        try:
            overrides = json.loads(goal.params_json) if goal.params_json.strip() else {}
            if not isinstance(overrides, dict):
                raise ValueError("params_json must be a JSON object")
        except (ValueError, json.JSONDecodeError) as ex:
            gh.abort()
            result.success, result.stop_reason, result.message = False, "error", f"Bad params_json: {ex}"
            return result
        cfg = {**load_mode_config(mode, self.profile, self.configs_dir), **overrides}
        handle = self.core.begin_task(mode)
        self.teleop_cancels = bool(cfg.get("teleop_cancels_task", True))
        self.path.reset()
        self.estop_since = self.stopped_since = None
        t0 = now_s(self)
        timeout = float(goal.timeout_s) if goal.timeout_s > 0 else float(cfg.get("timeout_s", 300.0))
        watch = ProgressWatchdog(float(cfg.get("no_progress_window_s", 60.0)), float(cfg.get("progress_min_distance_m", 0.15)))
        watch.reset(t0, self.pose)
        outcome: Outcome | None = None
        task = None
        try:
            task, outcome = self.start_task(mode, goal, cfg, timeout)
            fb_state = self._fb_state
            last_skill, last_turn = "", 0
            while outcome is None:
                now = now_s(self)
                if handle.cancelled:
                    task.cancel()
                    outcome = Outcome(False, handle.stop_reason, handle.reason)
                    break
                out = task.poll()
                if out is not None:
                    outcome = out
                    break
                if now - t0 > timeout:
                    handle.cancel(f"task timeout after {timeout:.0f} s", "timeout")
                    continue
                if self.estop_since is not None:
                    handle.cancel("soft e-stop active", "safety")
                    continue
                if self.stopped_since is not None and now - self.stopped_since > float(cfg.get("safety_abort_s", 45.0)):
                    handle.cancel(f"blocked by the safety guard for more than {cfg.get('safety_abort_s', 45.0)} s "
                                  f"({self.safety.reason if self.safety else 'obstacle'})", "safety")
                    continue
                if fb_state["skill"] != last_skill or fb_state["turn"] != last_turn:
                    last_skill, last_turn = fb_state["skill"], fb_state["turn"]
                    watch.note_progress(now, self.pose)
                if watch.update(now, self.pose):
                    handle.cancel(f"no progress: the robot neither moved nor turned for {watch.window_s:.0f} s", "no_progress")
                    continue
                self.publish_feedback(gh, mode, goal, fb_state)
                time.sleep(0.1)
        except Exception as ex:                                        # keep the server alive, report the failure
            self.get_logger().error(f"task crashed: {type(ex).__name__}: {ex}")
            if task is not None:
                task.cancel()
            outcome = Outcome(False, "error", f"Internal error in the mode manager: {type(ex).__name__}: {ex}")
        finally:
            self.core.end_task(handle)
        result.success = bool(outcome.success)
        result.stop_reason = outcome.stop_reason
        result.message = outcome.message
        result.path_length_m = float(self.path.total)
        result.duration_s = float(now_s(self) - t0)
        result.agent_turns = int(outcome.agent_turns)
        self.get_logger().info(f"task {mode} ended: success={result.success} reason={result.stop_reason} path={result.path_length_m:.2f} m "
                               f"time={result.duration_s:.1f} s: {result.message}")
        if outcome.stop_reason == "aborted" and handle.cancelled and gh.is_cancel_requested:
            gh.canceled()
        elif outcome.success:
            gh.succeed()
        else:
            gh.abort()
        return result

    def publish_feedback(self, gh, mode: str, goal, st: dict) -> None:
        f = Navigate.Feedback()
        f.status, f.current_skill, f.agent_turn = st["status"] or mode, st["skill"], int(st["turn"])
        f.distance_to_goal = math.nan
        if mode == "goal_pose":
            pose = lookup_pose(self.buf, self.map_frame, self.base_frame)
            if pose is not None:
                f.distance_to_goal = float(math.hypot(goal.goal.pose.position.x - pose.x, goal.goal.pose.position.y - pose.y))
        gh.publish_feedback(f)

    def start_task(self, mode: str, goal, cfg: dict, timeout: float):
        """Returns (task, outcome): outcome is set when the task could not even start."""
        if mode in ("velocity_follow", "object_nav", "follow_person"):
            client, action = {"velocity_follow": (self.c_follow, self.follow_action), "object_nav": (self.c_run, self.run_action),
                              "follow_person": (self.c_person, self.person_action)}[mode]
            task = DelegateTask(self, client, goal, action)
        elif mode == "goal_pose":
            args, err = self.goal_pose_args(goal, cfg)
            if args is None:
                return None, Outcome(False, "error", err)
            task = SkillTask(self, "goto_pose", args, timeout)
        else:
            task = ExploreTask(self, cfg, timeout)
        err = task.begin()
        if err:
            return task, Outcome(False, "error", err)
        return task, None


# ---------------------------------------------------------------------------------------------- tasks


class SkillTask:
    """A single skill call as a task."""

    def __init__(self, mm: ModeManager, name: str, args: dict, timeout: float):
        self.mm, self.name, self.args, self.timeout = mm, name, args, timeout
        self.run = ActionRun(mm.c_skill[name])

    def begin(self) -> str:
        self.mm._fb_state = {"turn": 0, "skill": self.name, "status": f"running {self.name}"}
        err = self.run.start(self.mm.skill_goal(self.name, self.args, self.timeout))
        if err:
            return f"The skill server is not running (no /x30/skill/{self.name}). Start x30_modes skill_server."
        return ""

    def cancel(self) -> None:
        self.run.cancel()

    def poll(self) -> Outcome | None:
        r = self.run.poll()
        if self.run.error:
            return Outcome(False, "error", f"Skill {self.name} could not start ({self.run.error}).")
        if r is None:
            return None
        res = r.result
        try:
            data = json.loads(res.data_json) if res.data_json else {}
        except json.JSONDecodeError:
            data = {}
        if res.success:
            return Outcome(True, "finished", res.message)
        reason = {"blocked": "safety", "timeout": "timeout", "cancelled": "aborted", "estop": "safety"}.get(str(data.get("reason", "")), "aborted")
        return Outcome(False, reason, res.message)


class ExploreTask:
    """Calls the explore skill repeatedly until no frontier is left."""

    def __init__(self, mm: ModeManager, cfg: dict, timeout: float):
        self.mm, self.cfg, self.timeout = mm, cfg, timeout
        self.run: ActionRun | None = None
        self.visited = 0
        self.failures = 0
        self.t_end = 0.0

    def _next(self) -> str:
        args = {k: self.cfg[k] for k in ("strategy", "bias", "steps", "scan", "until_label") if k in self.cfg}
        args.setdefault("steps", 1)
        self.run = ActionRun(self.mm.c_skill["explore"])
        self.mm._fb_state = {"turn": self.visited, "skill": "explore", "status": f"exploring ({self.visited} frontiers visited)"}
        if self.run.start(self.mm.skill_goal("explore", args, max(self.t_end - now_s(self.mm), 5.0))):
            return "The skill server is not running (no /x30/skill/explore). Start x30_modes skill_server."
        return ""

    def begin(self) -> str:
        self.t_end = now_s(self.mm) + self.timeout
        return self._next()

    def cancel(self) -> None:
        if self.run:
            self.run.cancel()

    def poll(self) -> Outcome | None:
        r = self.run.poll()
        if self.run.error:
            return Outcome(False, "error", f"Explore skill could not start ({self.run.error}).")
        if r is None:
            return None
        res = r.result
        try:
            data = json.loads(res.data_json) if res.data_json else {}
        except json.JSONDecodeError:
            data = {}
        if data.get("found") is not None:
            return Outcome(True, "finished", res.message)
        if data.get("exhausted"):
            self.visited += len(data.get("visited", []))
            return Outcome(True, "finished", f"Exploration complete: no frontier left after {self.visited} visited. {res.message}")
        if res.success:
            self.visited += len(data.get("visited", []))
            self.failures = 0
        else:
            if data.get("reason") == "no_frontier_service":
                return Outcome(False, "error", res.message)
            self.failures += 1
            if self.failures >= 3:
                return Outcome(False, "aborted", f"Exploration stuck after {self.failures} failed steps: {res.message}")
        err = self._next()
        return Outcome(False, "error", err) if err else None


class DelegateTask:
    """Forwards the Navigate goal to an agent action server (x30_agent) and relays feedback and result."""

    def __init__(self, mm: ModeManager, client: ActionClient, goal, action_name: str):
        self.mm, self.goal, self.action_name = mm, goal, action_name
        self.run = ActionRun(client)

    def begin(self) -> str:
        g = Navigate.Goal()
        g.mode, g.instruction, g.target_label = self.goal.mode, self.goal.instruction, self.goal.target_label
        g.goal, g.episode_id, g.timeout_s, g.params_json = self.goal.goal, self.goal.episode_id, self.goal.timeout_s, self.goal.params_json
        self.mm._fb_state = {"turn": 0, "skill": "", "status": f"delegated to {self.action_name}"}

        def on_fb(fb):
            f = fb.feedback
            self.mm._fb_state.update(turn=int(f.agent_turn), skill=f.current_skill, status=f.status)

        err = self.run.start(g, on_fb)
        if err:
            return (f"The agent action {self.action_name} is not available (x30_agent not running?). "
                    "Start the agent, or use mode goal_pose / explore which need no agent.")
        return ""

    def cancel(self) -> None:
        self.run.cancel()

    def poll(self) -> Outcome | None:
        r = self.run.poll()
        if self.run.error:
            return Outcome(False, "error", f"Agent action {self.action_name} did not start ({self.run.error}).")
        if r is None:
            return None
        res = r.result
        reason = res.stop_reason if res.stop_reason in ("finished", "timeout", "aborted", "safety", "no_progress", "error") else "aborted"
        return Outcome(bool(res.success), reason, res.message, int(res.agent_turns))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ModeManager()
    ex = MultiThreadedExecutor(num_threads=10)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
