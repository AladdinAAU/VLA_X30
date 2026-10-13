"""skill_server: one process serving /x30/skill/<name> (x30_msgs/action/Skill) for every skill of configs/skills.yaml,
plus /x30/primitive (blocking velocity primitive executor for the instruction follower).
"""
from __future__ import annotations

import json
import threading
import traceback

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from x30_msgs.action import Skill
from x30_msgs.srv import Json

from .primitives import parse_primitive_text, primitive_to_skill
from .registry import SkillRegistry
from .robot_api import RobotAPI
from .ros_io import declare_dataclass_params
from .skills.base import SkillConfig, SkillContext, SkillResult
from .skills.exploration import ExplorationSkills
from .skills.motion import MotionSkills
from .skills.navigation import NavigationSkills
from .skills.verify import VerifySkill


class SkillServer(Node):
    def __init__(self):
        super().__init__("skill_server")
        self.cfg = declare_dataclass_params(self, SkillConfig())
        self.registry = SkillRegistry.load(self.cfg.skills_registry or None)
        self.group = ReentrantCallbackGroup()
        self.api = RobotAPI(self, self.cfg, self.group)
        self.motion = MotionSkills(self.api, self.cfg)
        self.nav = NavigationSkills(self.api, self.cfg, self.motion, self.group)
        self.explore = ExplorationSkills(self.api, self.cfg, self.motion, self.nav)
        self.verify = VerifySkill(self.api, self.cfg, self.motion)
        self.impl = {
            "goto_pose": self.nav.goto_pose, "goto_object": self.nav.goto_object, "approach": self.nav.approach,
            "explore": self.explore.explore, "search": self.explore.search, "turn_to": self.motion.turn_to,
            "move_relative": self.motion.move_relative, "scan": self.motion.scan, "verify": self.verify.run, "stop": self.stop_skill,
        }
        self._lock = threading.Lock()
        self._active_name: str | None = None
        self._active_ctx: SkillContext | None = None
        self._ctx_by_goal: dict[bytes, SkillContext] = {}
        self._cancel_early: set[bytes] = set()
        self._servers = []
        for name in self.registry.names():
            if name not in self.impl:
                self.get_logger().warn(f"registry skill '{name}' has no implementation; not served")
                continue
            self._servers.append(ActionServer(
                self, Skill, f"/x30/skill/{name}", execute_callback=lambda gh, n=name: self.execute(gh, n),
                goal_callback=lambda req: GoalResponse.ACCEPT, cancel_callback=self.on_cancel, callback_group=self.group))
        self.create_service(Json, "/x30/primitive", self.on_primitive, callback_group=self.group)
        self.get_logger().info(f"skill_server ready: {', '.join(s for s in self.registry.names() if s in self.impl)}")

    # ------------------------------------------------------------ execution core
    def run_skill(self, name: str, args_json: str, timeout_s: float, gh=None, cancel: threading.Event | None = None) -> SkillResult:
        spec = self.registry.skills[name]
        args, err = self.registry.parse_args(name, args_json)
        if args is None:
            return SkillResult(False, err, {"reason": "invalid_args"})
        timeout = timeout_s if timeout_s > 0 else spec.default_timeout_s
        fb = None
        if gh is not None:
            def fb(text: str, progress: float) -> None:
                f = Skill.Feedback()
                f.status, f.progress = text, float(progress)
                gh.publish_feedback(f)
        ctx = SkillContext(self.api.now, timeout, fb, cancel)
        if gh is not None:
            uid = bytes(gh.goal_id.uuid)
            self._ctx_by_goal[uid] = ctx
            if uid in self._cancel_early or gh.is_cancel_requested:
                ctx.cancel_event.set()
        needs_lock = spec.exclusive or (name == "verify" and args.get("object_id") is not None and args.get("face_object", True))
        got = False
        try:
            if name == "stop":
                return self.stop_skill(ctx, args)
            if needs_lock:
                got = self._lock.acquire(blocking=False)
                if not got:
                    return SkillResult(False, f"Busy: skill '{self._active_name}' is still running. Wait for it to finish, or call stop to cancel it, "
                                       "then repeat this call.", {"reason": "busy", "running": self._active_name})
                self._active_name, self._active_ctx = name, ctx
            try:
                res = self.impl[name](ctx, args)
            except Exception as ex:                                     # a skill must never kill the server
                self.get_logger().error(f"skill {name} crashed: {traceback.format_exc()}")
                res = SkillResult(False, f"Internal error in skill {name}: {type(ex).__name__}: {ex}. Next: retry once; if it repeats, "
                                  "use another skill and report the problem.", {"reason": "internal_error"})
            res.data.setdefault("elapsed_s", round(ctx.elapsed, 1))
            return res
        finally:
            if needs_lock and spec.category == "motion":
                self.api.halt()
            if got:
                self._active_name, self._active_ctx = None, None
                self._lock.release()
            if gh is not None:
                self._ctx_by_goal.pop(bytes(gh.goal_id.uuid), None)
                self._cancel_early.discard(bytes(gh.goal_id.uuid))

    def execute(self, gh, name: str) -> Skill.Result:
        req = gh.request
        if req.name and req.name != name:
            self.get_logger().warn(f"goal for /x30/skill/{name} carries name '{req.name}'; using '{name}'")
        res = self.run_skill(name, req.args_json, float(req.timeout_s), gh)
        out = Skill.Result()
        out.success, out.message = bool(res.success), res.message
        out.data_json = json.dumps(res.data, default=str)
        if gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.succeed()
        return out

    def on_cancel(self, gh) -> CancelResponse:
        uid = bytes(gh.goal_id.uuid)
        ctx = self._ctx_by_goal.get(uid)
        if ctx is not None:
            ctx.cancel_event.set()
        else:
            self._cancel_early.add(uid)
        return CancelResponse.ACCEPT

    def stop_skill(self, ctx: SkillContext, args: dict) -> SkillResult:
        running = self._active_name
        other = self._active_ctx
        if other is not None and other is not ctx:
            other.cancel_event.set()
        self.api.halt(5)
        if running is not None:
            for _ in range(60):                       # the cancelled skill releases the lock within a moment
                if self._active_name is None:
                    break
                ctx.sleep(0.05)
            self.api.halt(3)
        msg = f"Stopped. Cancelled the running skill '{running}'." if running else "Stopped (no skill was running); zero velocity commanded."
        return SkillResult(True, msg, {"cancelled_skill": running})

    # ------------------------------------------------------------ /x30/primitive
    def on_primitive(self, req: Json.Request, res: Json.Response) -> Json.Response:
        try:
            body = json.loads(req.request or "{}")
            if "text" in body:
                prim = parse_primitive_text(str(body["text"]))
            else:
                from .primitives import Primitive
                action = str(body["action"])
                prim = Primitive(action, abs(float(body.get("value", 0.0))))
            name, args = primitive_to_skill(prim)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as ex:
            res.ok = False
            res.response = json.dumps({"success": False, "message": f"Bad primitive request: {ex}"})
            return res
        out = self.run_skill(name, json.dumps(args), 0.0)
        res.ok = bool(out.success)
        res.response = json.dumps({"success": out.success, "message": out.message, "skill": name, "args": args, **out.data}, default=str)
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SkillServer()
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
