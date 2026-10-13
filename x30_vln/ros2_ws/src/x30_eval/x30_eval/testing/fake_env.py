"""FAKE environment for tests and dry runs. NOT a simulator of the X30, Gazebo or any real navigation stack.

A tiny 2D world (two rooms joined by a door) with a unicycle robot, a noisy "SLAM" pose in its own frame, a toy
semantic mapper and scripted policies standing in for the system under test. It implements the same ``Env``
interface as ``RosEnv``, so the complete runner / scoring / report / sweep / export chain can be exercised without
Gazebo or ROS. Numbers produced with it say nothing about the real system; they only prove the evaluation code.

Episode field ``fake_policy`` (in the episode file) selects the behaviour:
  oracle        plans on the grid, drives to ``standoff_m`` from the goal, stops and claims success
  wrong_object  drives to an object of another label and claims success
  never_stop    reaches the goal but never ends the task (system timeout)
  wander        random walk until timeout
  loop          repeats the same tool call, ends with stop_reason no_progress
  collider      drives into the nearest wall, ends with stop_reason aborted
  hang          never ends and ignores everything (evaluator timeout)
  task_error    ends with a connection error message (an infrastructure failure)
Parameters read from the config overlay (so sweeps have something to tune):
  modes/object_nav.standoff_m   (default 1.0)     robot_sim.limits.vx   (default 0.5)
Failure injection: ``inject={"<episode_id>": ["reset_error", "task_error", ...]}`` consumed one entry per attempt;
the kind ``health_error`` makes ``check_health`` raise.
"""
from __future__ import annotations

import json
import math
import os
import zlib
from pathlib import Path

import numpy as np

from ..core import metrics as M
from ..core.episodes import Episode
from ..core.geometry import SE2, wrap
from ..core.grid import OccGrid, grid_from_rects, shortest_path
from ..core.infra import InfraError
from ..core.overlay import deep_merge
from ..core.recording import Recording, TaskResult
from ..env.base import Env

SCENE = "fake_house"
CAMERA = {"K": [60.0, 60.0, 32.0, 24.0], "width": 64, "height": 48, "base_to_cam": {"t": [0.3, 0.0, 0.3], "q": [-0.5, 0.5, -0.5, 0.5]}}
DT = 0.1


def default_grid() -> OccGrid:
    """12 x 8 m, a dividing wall at x=6 with a door between y=3.0 and y=4.5."""
    return grid_from_rects(12.0, 8.0, 0.1, [[5.9, 0.0, 6.1, 3.0], [5.9, 4.5, 6.1, 8.0]])


def default_objects() -> list[dict]:
    def o(i, label, x, y, sx=0.5, sy=0.5):
        return {"id": i, "name": f"{label}_{i}", "label": label, "x": x, "y": y, "z": 0.4, "sx": sx, "sy": sy, "sz": 0.8}
    return [o(1, "chair", 3.0, 1.5), o(2, "chair", 9.0, 6.0), o(3, "table", 3.0, 6.0, 1.2, 0.8),
            o(4, "plant", 10.0, 1.5), o(5, "sofa", 9.0, 3.5, 1.8, 0.8)]


def _param(overlay: dict, dotted: str, default):
    stem, _, rest = dotted.partition(".")
    node = overlay.get(stem, {})
    for k in rest.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


class FakeEnv(Env):
    name = "fake"

    def __init__(self, overlay: dict | None = None, grid: OccGrid | None = None, objects: list[dict] | None = None,
                 inject: dict[str, list[str]] | None = None, contacts: bool = False, seed: int = 0):
        self.overlay = overlay or {}
        self.grid = grid or default_grid()
        self.objects = objects or default_objects()
        self.inject = {k: list(v) for k, v in (inject or {}).items()}
        self.use_contacts = contacts
        self.vx = float(_param(self.overlay, "robot_sim.limits.vx", 0.5))
        self.standoff = float(_param(self.overlay, "modes/object_nav.standoff_m", 1.0))
        self.attempts: dict[str, int] = {}
        self.base_seed = seed
        self._health_fail = False
        self.ep: Episode | None = None
        self.resets = 0

    # ------------------------------------------------------------------ Env interface
    def check_health(self) -> None:
        if self._health_fail:
            raise InfraError("fake container down", "container")

    def _next_injection(self, ep_id: str) -> str | None:
        queue = self.inject.get(ep_id)
        return queue.pop(0) if queue else None

    def reset_episode(self, ep: Episode, start: dict, seed: int, cfg: dict) -> None:
        self.attempts[ep.id] = self.attempts.get(ep.id, 0) + 1
        self.ep = ep
        self._injected = self._next_injection(ep.id)
        if self._injected == "health_error":
            self._health_fail = True
            raise InfraError("fake container down", "container")
        if self._injected == "reset_error":
            raise InfraError("service /sim/set_robot_pose not available", "service")
        self.rng = np.random.default_rng([self.base_seed, seed, zlib.crc32(ep.id.encode())])
        self.pose = np.array([start["x"], start["y"], start["yaw"]], float)
        self.start_se2 = SE2(*self.pose)
        self.map_objects: list[dict] = []
        self.drift = np.zeros(3)
        self.resets += 1
        self.policy = None

    def begin(self, ep: Episode, mode: str, params: dict, trace_path: str | None) -> None:
        self.t = 0.0
        self.mode = mode
        self.policy = ep.extra.get("fake_policy", "oracle")
        if self._injected == "task_error":
            self.policy = "task_error"
        self.rec = Recording(scene_info={"name": SCENE, "camera": CAMERA})
        self.rec.gt_objects = [dict(o) for o in self.objects]
        self.rec.contacts_available = self.use_contacts
        self.trace_path = trace_path
        self.frames_dir = params.get("frames_dir")
        self.trace_f = None
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            self.trace_f = open(trace_path, "w", encoding="utf-8")
        self.turn = 0
        self.next_turn_t = 1.0
        self.result: TaskResult | None = None
        self.cancelled = False
        self.last_stopped = False
        self.t_pair = 0.0
        self.t_map = 0.0
        self.last_frame_t = -1.0
        self.wander_heading = float(self.rng.uniform(-math.pi, math.pi))
        self._plan(ep)
        self._record()

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        for _ in range(max(1, int(round(dt / DT)))):
            self._step()

    def task_result(self) -> TaskResult | None:
        return self.result

    def cancel(self) -> None:
        self.cancelled = True

    def finish_recording(self) -> Recording:
        if self.trace_f:
            self.trace_f.close()
            self.trace_f = None
        self.rec.map_objects = [dict(o) for o in self.map_objects]
        return self.rec

    def wall_limit_s(self, ep: Episode, cfg: dict) -> float | None:
        return None

    def world_info(self) -> dict:
        return {"name": SCENE}

    # ------------------------------------------------------------------ planning and policies
    def _goal_region(self, ep: Episode, radius: float):
        g = ep.goal
        if g["type"] == "pose":
            return lambda xs, ys: np.hypot(xs - g["x"], ys - g["y"]) <= radius
        inst = [o for o in self.objects if M.label_match(o["label"], g["label"])]

        def region(xs, ys):
            return np.min([np.hypot(np.maximum(np.abs(xs - o["x"]) - o["sx"] / 2, 0), np.maximum(np.abs(ys - o["y"]) - o["sy"] / 2, 0))
                           for o in inst], axis=0) <= radius
        return region

    def _plan(self, ep: Episode) -> None:
        self.waypoints: list[tuple[float, float]] = []
        pol = self.policy
        if pol in ("oracle", "never_stop"):
            radius = ep.goal.get("radius", 0.5) if ep.goal["type"] == "pose" else self.standoff
            radius = min(radius, 0.4) if (ep.goal["type"] == "pose") else radius
            res = shortest_path(self.grid, self.pose[:2], self._goal_region(ep, radius), inflate_m=0.3)
            self.waypoints = res[1][1:] if res else []
        elif pol == "wrong_object":
            others = [o for o in self.objects if not M.label_match(o["label"], ep.goal.get("label", ""))]
            tgt = min(others, key=lambda o: math.hypot(o["x"] - self.pose[0], o["y"] - self.pose[1]))
            res = shortest_path(self.grid, self.pose[:2], (tgt["x"], tgt["y"]), inflate_m=0.3)
            self.waypoints = res[1][1:-1] if res and len(res[1]) > 2 else (res[1][1:] if res else [])
        elif pol == "collider":
            self.waypoints = []
        self.wp_index = 0

    def _drive(self, v: float, w: float) -> None:
        """Unicycle step with lateral noise that grows with speed (so speed trades off against collisions)."""
        noise = float(self.rng.normal(0.0, 0.03 * v / 0.5))
        x, y, yaw = self.pose
        dist = v * DT
        self.pose = np.array([x + dist * math.cos(yaw) - noise * DT * math.sin(yaw),
                              y + dist * math.sin(yaw) + noise * DT * math.cos(yaw), wrap(yaw + w * DT)])

    def _toward(self, target: tuple[float, float]) -> tuple[float, float, float]:
        dx, dy = target[0] - self.pose[0], target[1] - self.pose[1]
        err = wrap(math.atan2(dy, dx) - self.pose[2])
        w = float(np.clip(2.0 * err, -0.8, 0.8))
        v = self.vx * max(0.0, math.cos(err)) if abs(err) < 0.9 else 0.0
        return v, w, math.hypot(dx, dy)

    def _turn_trace(self, tool: str, args: dict, ok: bool = True) -> None:
        """One row in the format of x30_agent's TraceWriter (kind=turn with messages, images, usage, result)."""
        if not self.trace_f:
            return
        import matplotlib.image as mpimg
        self.turn += 1
        img_dir = Path(self.trace_path).with_suffix("")
        img_dir = img_dir.parent / f"{img_dir.name}_img"
        img_dir.mkdir(parents=True, exist_ok=True)
        pic = np.zeros((48, 64, 3), np.float32)
        pic[..., 0], pic[..., 1], pic[..., 2] = (self.pose[0] % 12) / 12, (self.pose[1] % 8) / 8, (self.turn % 5) / 5
        mpimg.imsave(img_dir / f"t{self.turn:03d}_0.png", pic)
        state = f"Task: {self.ep.instruction}\nPose: {self.pose[0]:.1f}, {self.pose[1]:.1f}. Objects mapped: {len(self.map_objects)}."
        msgs = [{"role": "system", "content": "You are the high-level controller of a quadruped robot. Call exactly one tool per turn."},
                {"role": "user", "content": [{"type": "text", "text": state}, {"type": "image_url", "image_url": {"url": "[image] (1234 chars)"}}]}]
        row = {"kind": "turn", "t": round(self.t, 2), "turn": self.turn, "pose": [float(v) for v in self.pose],
               "images": [f"{img_dir.name}/t{self.turn:03d}_0.png"], "prompt_tokens": 1500 + 40 * self.turn, "completion_tokens": 25,
               "latency_s": round(float(self.rng.normal(0.8, 0.1)), 3), "messages": msgs,
               "reply": {"content": "", "tool_calls": [{"name": tool, "raw_arguments": json.dumps(args)}]},
               "tool": tool, "args": args, "parse_ok": True, "refused": False, "success": ok,
               "result": {"success": ok, "message": f"{tool} done", "data": {}}}
        self.trace_f.write(json.dumps(row) + "\n")
        self.trace_f.flush()

    _last_frame: str | None = None

    def _finish(self, success: bool, reason: str, message: str = "") -> None:
        self.result = TaskResult(success=success, stop_reason=reason, message=message, duration_s=self.t,
                                 agent_turns=self.turn, stopped_by_system=True, path_length_m=0.0)
        self.rec.t_end = self.t

    def _policy_step(self) -> tuple[float, float]:
        ep, pol = self.ep, self.policy
        if pol == "hang":
            return 0.0, 0.0
        if self.t >= ep.time_limit_s:
            self._finish(False, "timeout", "system time limit")
            return 0.0, 0.0
        if self.t >= self.next_turn_t:
            self.next_turn_t += 5.0
            tool = {"loop": "scan", "wander": "explore", "collider": "move_relative"}.get(pol, "goto_object")
            args = {"degrees": 360} if tool == "scan" else {"label": ep.target_label, "step": self.turn}
            self._turn_trace(tool, args)
        if pol == "task_error":
            if self.t >= 2.0:
                self._finish(False, "error", "LLM error: Connection refused (http://vlm:8080)")
            return 0.0, 0.0
        if pol == "loop":
            if self.turn >= 8:
                self._finish(False, "no_progress", "no progress detected")
            return 0.0, 0.3
        if pol == "wander":
            ahead = self.grid.min_dist_to_occupied(self.pose[0] + 0.9 * math.cos(self.pose[2]), self.pose[1] + 0.9 * math.sin(self.pose[2]), 0.5)
            if self.t % 6.0 < DT or ahead < 0.45:
                self.wander_heading = float(self.pose[2] + self.rng.uniform(1.0, 2.5) * self.rng.choice([-1, 1]))
            err = wrap(self.wander_heading - self.pose[2])
            return (self.vx * 0.5 if abs(err) < 0.5 else 0.0), float(np.clip(1.5 * err, -0.6, 0.6))
        if pol == "collider":
            err = wrap(math.pi - self.pose[2])           # face the west wall
            if abs(err) > 0.2:
                return 0.0, float(np.clip(2 * err, -0.8, 0.8))
            return self.vx, 0.0
        # waypoint followers
        if self.wp_index < len(self.waypoints):
            v, w, d = self._toward(self.waypoints[self.wp_index])
            if d < 0.15:
                self.wp_index += 1
            return v, w
        if pol == "never_stop":
            return 0.0, 0.0
        self._turn_trace("finish", {"reason": "reached"})
        self._finish(True, "finished", f"{pol} done")
        return 0.0, 0.0

    # ------------------------------------------------------------------ simulation step and recording
    def _step(self) -> None:
        if self.result is not None or self.cancelled:
            self.t += DT
            self._record()
            return
        v, w = self._policy_step()
        if self.result is None and v != 0.0:
            clearance = self.grid.min_dist_to_occupied(self.pose[0] + 0.35 * math.cos(self.pose[2]),
                                                       self.pose[1] + 0.35 * math.sin(self.pose[2]), 0.6)
            stopped = clearance < 0.12 and self.policy != "collider"
            if stopped:
                v = 0.0
            self.rec.safety.append({"t": round(self.t, 2), "stopped": stopped, "limiting": stopped,
                                    "reason": "obstacle" if stopped else "", "min_dist": clearance}) \
                if stopped != self.last_stopped else None
            self.last_stopped = stopped
        if self.result is None:
            self._drive(v, w)
        self.t += DT
        if self.policy == "collider" and self.result is None and self.grid.occupied_at(*self.pose[:2] + 0.45 * np.array(
                [math.cos(self.pose[2]), math.sin(self.pose[2])])):
            if self.use_contacts:
                self.rec.contacts.append(round(self.t, 2))
            self._finish(False, "aborted", "collision")
        self._record()

    def _slam_pose(self) -> np.ndarray:
        self.drift += self.rng.normal(0.0, [0.004, 0.004, 0.0008])
        local = self.start_se2.inverse().compose(SE2(*self.pose))
        return np.array([local.x + self.drift[0], local.y + self.drift[1], wrap(local.yaw + self.drift[2])])

    def _record(self) -> None:
        self.rec.gt.append([round(self.t, 3), *map(float, self.pose)])
        if self.t - self.t_pair >= 0.1 - 1e-9:
            self.t_pair = self.t
            s = self._slam_pose()
            self.rec.pairs.append([round(self.t, 3), *map(float, s), *map(float, self.pose)])
        if self.t - self.t_map >= 0.5 - 1e-9:
            self.t_map = self.t
            self._map_update()
            if self.frames_dir and self.t - self.last_frame_t >= 1.0 - 1e-9:
                self._save_frame()

    def _save_frame(self) -> None:
        import matplotlib.image as mpimg
        Path(self.frames_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(self.frames_dir, f"{len(self.rec.frames):05d}.png")
        img = np.zeros((48, 64, 3), np.float32)
        img[..., 0] = (self.pose[0] % 12) / 12
        img[..., 1] = (self.pose[1] % 8) / 8
        mpimg.imsave(path, img)
        self.last_frame_t = self.t
        self._last_frame = path
        self.rec.frames.append({"t": round(self.t, 2), "path": path, "gt": [float(v) for v in self.pose], "gt_z": 0.4})
        from ..export.perception import project_object_box
        items = []
        for o in self.objects:                                        # toy perception: detects 90% of the visible objects with box noise
            pr = project_object_box(o, (self.pose[0], self.pose[1], 0.4, self.pose[2]), CAMERA)
            if pr and self.rng.random() < 0.9:
                items.append({"label": o["label"], "score": 0.8, "box": [v + float(self.rng.normal(0, 1.0)) for v in pr["box"]], "bearing_only": False})
        if self.rng.random() < 0.1:
            items.append({"label": "lamp", "score": 0.4, "box": [5.0, 5.0, 20.0, 25.0], "bearing_only": False})
        self.rec.detections.append({"t": round(self.t, 2), "image_stamp": self.t, "items": items})

    def _map_update(self) -> None:
        """Toy semantic mapper: objects within 5 m, +-60 deg and with line of sight are detected with 0.15 m noise."""
        x, y, yaw = self.pose
        for o in self.objects:
            dx, dy = o["x"] - x, o["y"] - y
            d = math.hypot(dx, dy)
            if d > 5.0 or abs(wrap(math.atan2(dy, dx) - yaw)) > math.radians(60) or not self._los(x, y, o["x"], o["y"]):
                continue
            p = self.start_se2.inverse().apply(np.array([o["x"], o["y"]]) + self.rng.normal(0, 0.15, 2))
            for m in self.map_objects:
                if m["label"] == o["label"] and math.hypot(m["x"] - p[0], m["y"] - p[1]) < 0.4:
                    m["observations"] += 1
                    break
            else:
                self.map_objects.append({"id": len(self.map_objects), "label": o["label"], "x": float(p[0]), "y": float(p[1]),
                                         "z": o["z"], "confidence": 0.8, "observations": 1, "bearing_only": False})

    def _los(self, x0, y0, x1, y1) -> bool:
        for s in np.linspace(0.0, 1.0, int(math.hypot(x1 - x0, y1 - y0) / 0.1) + 2):
            if self.grid.occupied_at(x0 + s * (x1 - x0), y0 + s * (y1 - y0)):
                return False
        return True


def merged_overlay(*parts: dict) -> dict:
    out: dict = {}
    for p in parts:
        out = deep_merge(out, p)
    return out


def fake_optimum(cfg: dict):
    """OptimumProvider that knows the fake scene's grid."""
    from ..core.optimum import OptimumProvider
    return OptimumProvider(cfg, grids={SCENE: default_grid()})
