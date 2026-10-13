"""ROS 2 nodes: reasoner_agent (/x30/agent/run, object_nav) and instruction_follower (/x30/agent/follow, velocity_follow).

Both are Navigate action servers with the same structure. The agent kind is a parameter (and can be overridden per goal
through params_json {"agent_kind": ...}): llm | scripted_search | dead_reckon.

params_json keys understood: seed (sampling seed), trace_dir (absolute dir for traces), agent_kind, max_turns, policy (nested overrides of configs/agent/<mode>.yaml), reference_path
([[x, y], ...]), reference_frame ("map" | "world"), start ({x, y, yaw} of the episode start in the world frame), use_verify.
"""
from __future__ import annotations

import json
import os
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from x30_msgs.action import Navigate

from x30_common import VLMClient, find_config_dir, load_models_config

from ..core.baselines import DeadReckon, ScriptedSearch, target_label_from, world_to_map_path
from ..core.follower import FollowerAgent
from ..core.policy import load_policy
from ..core.reasoner import ReasonerAgent
from ..core.skills import load_registry
from ..core.types import ABORTED, ERROR, AgentResult, Pose2D, Task
from .ros_env import RosEnv, quat_yaw


class AgentServer(Node):
    mode = "object_nav"
    action_name = "/x30/agent/run"
    policy_name = "object_nav"
    node_name = "reasoner_agent"

    def __init__(self):
        super().__init__(self.node_name)
        P = self.declare_parameter
        self.agent_kind = P("agent_kind", "llm").value
        self.config_dir = P("config_dir", "").value
        self.results_dir = P("results_dir", os.environ.get("X30_RESULTS", "/x30/results")).value
        self.run_name = P("run_name", time.strftime("run_%Y%m%d_%H%M%S")).value
        self.map_frame = P("map_frame", "map").value
        self.base_frame = P("base_frame", "base_link").value
        self.camera_topic = P("camera_topic", "/camera/color/image_raw").value
        self.annotated_topic = P("annotated_map_topic", "/semantic_map/annotated_map").value
        self.models_yaml = P("models_config", "").value
        self.default_overrides = P("policy_overrides_json", "{}").value
        self.write_traces = P("write_traces", True).value
        cdir = self.config_dir or (str(find_config_dir()) if find_config_dir() else "")
        if not cdir:
            raise RuntimeError("configs/ directory not found: set param config_dir or X30_CONFIG_DIR")
        self.config_dir = cdir
        self.registry = load_registry(os.path.join(cdir, "skills.yaml"))
        self.models = load_models_config(self.models_yaml or None)
        self.client = VLMClient.from_config("reasoner", self.models)
        self.cb = ReentrantCallbackGroup()
        self.env = RosEnv(self, self.map_frame, self.base_frame, self.camera_topic, self.annotated_topic)
        self._busy = threading.Lock()
        self._cancel = threading.Event()
        self.server = ActionServer(self, Navigate, self.action_name, execute_callback=self._execute, goal_callback=self._on_goal,
                                   cancel_callback=self._on_cancel, callback_group=self.cb)
        self.get_logger().info(f"{self.node_name} ready: {self.action_name}, agent_kind={self.agent_kind}, model={self.client.base_url} "
                               f"({self.client.model}), policy={self.policy_name}")

    # ---------------------------------------------------------------- action plumbing
    def _on_goal(self, goal):
        if goal.mode and goal.mode != self.mode:
            self.get_logger().warn(f"goal mode '{goal.mode}' differs from this server's mode '{self.mode}'; accepted")
        if self._busy.locked():
            self.get_logger().warn("rejecting goal: an episode is already running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _handle):
        self.get_logger().info("cancel requested")
        self._cancel.set()
        return CancelResponse.ACCEPT

    def _execute(self, handle):
        goal = handle.request
        with self._busy:
            self._cancel.clear()

            def fb(status, skill, distance, turn):
                m = Navigate.Feedback()
                m.status, m.current_skill, m.distance_to_goal, m.agent_turn = status, skill, float(distance), int(turn)
                handle.publish_feedback(m)

            self.env.set_feedback(fb)
            try:
                res = self._run_goal(goal)
            except Exception as ex:                         # an episode must always end with a result
                self.get_logger().error(f"episode crashed: {type(ex).__name__}: {ex}")
                res = AgentResult(False, ERROR, f"crash: {type(ex).__name__}: {ex}")
            finally:
                self.env.set_feedback(None)
        out = Navigate.Result()
        out.success, out.stop_reason, out.message = bool(res.success), res.stop_reason, res.message[:500]
        out.duration_s, out.agent_turns = float(res.duration_s), int(res.turns)
        if self._cancel.is_set() and res.stop_reason == ABORTED:
            handle.canceled()
        elif res.stop_reason == ABORTED:
            handle.abort()
        else:
            handle.succeed()
        self.get_logger().info(f"episode {goal.episode_id}: success={res.success} stop={res.stop_reason} turns={res.turns} msg={res.message[:100]}")
        return out

    # ---------------------------------------------------------------- one episode
    def _task(self, goal, params: dict) -> Task:
        gp = None
        if goal.goal.header.frame_id or goal.goal.pose.position.x or goal.goal.pose.position.y:
            p = goal.goal.pose
            gp = Pose2D(p.position.x, p.position.y, quat_yaw(p.orientation))
        return Task(instruction=goal.instruction, mode=self.mode, target_label=goal.target_label, episode_id=goal.episode_id or "episode",
                    max_turns=int(params.get("max_turns", 0)), time_budget_s=float(goal.timeout_s or 0), params=params,
                    reference_path=params.get("reference_path", []) or [], goal_pose=gp)

    def _run_goal(self, goal) -> AgentResult:
        try:
            params = json.loads(goal.params_json) if goal.params_json else {}
            if not isinstance(params, dict):
                raise ValueError("params_json must be an object")
        except ValueError as ex:
            return AgentResult(False, ERROR, f"bad params_json: {ex}")
        task = self._task(goal, params)
        kind = params.get("agent_kind", self.agent_kind)
        trace_dir = os.path.join(self.results_dir, self.run_name, "traces") if self.write_traces else None
        if params.get("trace_dir"):                          # the eval runner chooses where traces go
            trace_dir = str(params["trace_dir"])
        self.client.seed = int(params["seed"]) if params.get("seed") is not None else None
        ov = json.loads(self.default_overrides or "{}")
        for k, v in (params.get("policy") or {}).items():
            ov[k] = {**ov.get(k, {}), **v} if isinstance(v, dict) else v
        return self.run_agent(kind, task, params, ov, trace_dir)

    def run_agent(self, kind: str, task: Task, params: dict, overrides: dict, trace_dir: str | None) -> AgentResult:
        raise NotImplementedError


class ReasonerAgentNode(AgentServer):
    mode, action_name, policy_name, node_name = "object_nav", "/x30/agent/run", "object_nav", "reasoner_agent"

    def run_agent(self, kind, task, params, overrides, trace_dir):
        pol = load_policy(str(params.get("policy_name") or self.policy_name), os.path.join(self.config_dir, "agent"), overrides)   # e.g. "locate_nav"
        if kind == "llm":
            ag = ReasonerAgent(self.client, self.env, self.registry, pol, trace_dir, self._cancel, self.models.hash(), "object_nav")
            return ag.run(task)
        if kind == "scripted_search":
            label = task.target_label or target_label_from(task.instruction)
            return ScriptedSearch(self.env, self.registry, pol, trace_dir, self._cancel, use_verify=bool(params.get("use_verify", True))).run(task, label)
        if kind == "dead_reckon":
            return _dead_reckon(self.env, task, params, trace_dir, self._cancel)
        return AgentResult(False, ERROR, f"unknown agent_kind '{kind}'")


class InstructionFollowerNode(AgentServer):
    mode, action_name, policy_name, node_name = "velocity_follow", "/x30/agent/follow", "velocity_follow", "instruction_follower"

    def run_agent(self, kind, task, params, overrides, trace_dir):
        if kind == "llm":
            pol = load_policy(self.policy_name, os.path.join(self.config_dir, "agent"), overrides)
            return FollowerAgent(self.client, self.env, pol, trace_dir, self._cancel, self.models.hash()).run(task)
        if kind == "dead_reckon":
            return _dead_reckon(self.env, task, params, trace_dir, self._cancel)
        return AgentResult(False, ERROR, f"agent_kind '{kind}' is not available for velocity_follow (use llm or dead_reckon)")


def _dead_reckon(env: RosEnv, task: Task, params: dict, trace_dir, cancel) -> AgentResult:
    path = task.reference_path
    pose = env.pose()
    if path and params.get("reference_frame", "world") == "world":
        st = params.get("start")
        if pose is None or not st:
            return AgentResult(False, ERROR, "dead_reckon with a world-frame path needs the robot pose and params_json 'start' {x, y, yaw}")
        path = world_to_map_path(path, (float(st["x"]), float(st["y"]), float(st.get("yaw", 0.0))), pose)
    return DeadReckon(env, trace_dir, cancel).run(task, waypoints=path)


def _spin(node_cls) -> None:
    rclpy.init()
    node = node_cls()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def reasoner_main() -> None:
    _spin(ReasonerAgentNode)


def follower_main() -> None:
    _spin(InstructionFollowerNode)


