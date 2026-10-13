"""Episode suite runner (ROS-free): drives an Env through the episodes, repeats, retries infrastructure failures."""
from __future__ import annotations

import re
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .episodes import Episode, episode_key
from .infra import InfraError, looks_like_infra
from .manifest import comparable
from .optimum import OptimumProvider
from .recording import TaskResult
from .runlog import RunDir
from .scoring import score_episode
from .summary import build_summary, derive_difficulty
from .traces import read_trace
from ..env.base import Env


class SuiteAborted(RuntimeError):
    """Too many consecutive infrastructure failures: the bench is down, continuing would only burn time."""


@dataclass
class SuiteOptions:
    run_name: str
    repeats: int = 1
    seed: int = 0
    force_mode: str | None = None
    params: dict = field(default_factory=dict)       # extra Navigate params for every episode
    resume: bool = True
    force_resume: bool = False                       # resume even if the manifest differs (non-comparable results)
    save_tracks: bool = True
    log: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep


def safe_name(key: str) -> str:
    """File-system and agent-trace safe name (the agent's TraceWriter applies the same character set and an 80 character cap)."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.replace("#", "_"))
    return name if len(name) <= 80 else f"{name[:70]}_{zlib.crc32(name.encode()):08x}"


def jitter_start(ep: Episode, seed: int, cfg: dict, optimum: OptimumProvider) -> dict:
    """Seeded start perturbation (optional). Skipped when no GT grid exists to prove the new start is in free space."""
    jx, jy = cfg["run"]["start_jitter_xy_m"], cfg["run"]["start_jitter_yaw_rad"]
    start = dict(ep.start)
    if jx <= 0 and jy <= 0:
        return start
    rng = np.random.default_rng([seed, zlib.crc32(ep.id.encode())])
    grid = optimum.grid_for(ep.scene)
    for _ in range(20):
        dx, dy = rng.uniform(-jx, jx, 2)
        if grid is None or not grid.inflated(0.3)[grid.to_cell(start["x"] + dx, start["y"] + dy)]:
            start["x"] += float(dx)
            start["y"] += float(dy)
            break
    start["yaw"] = start["yaw"] + float(rng.uniform(-jy, jy))
    return start


def run_attempt(env: Env, ep: Episode, repeat: int, seed: int, cfg: dict, run: RunDir, opts: SuiteOptions,
                optimum: OptimumProvider) -> dict:
    """One attempt of one episode. Raises InfraError for bench problems; returns a scored record otherwise
    (possibly flagged ``infrastructure_failure`` when the system's own result looks like a bench failure)."""
    rc = cfg["run"]
    key = episode_key(ep.id, repeat)
    mode = opts.force_mode or ep.mode
    env.check_health()
    start = jitter_start(ep, seed, cfg, optimum)
    env.reset_episode(ep, start, seed, cfg)
    trace_path = str(run.traces / f"{safe_name(key)}.jsonl")
    params = {"seed": seed, "episode": ep.id, "repeat": repeat, "trace_dir": str(run.traces),
              "trace_id": safe_name(key), "trace_file": trace_path,
              "frames_dir": str(run.frames / safe_name(key)), **ep.params, **opts.params}
    t_wall = time.monotonic()
    env.begin(ep, mode, params, trace_path)
    limit = ep.time_limit_s + rc["cancel_grace_s"]
    wall_limit = env.wall_limit_s(ep, cfg)
    task: TaskResult | None = None
    while True:
        task = env.task_result()
        if task is not None:
            break
        if env.now() >= limit:
            env.cancel()
            task = TaskResult(success=False, stop_reason="evaluator_timeout", message="time limit reached", duration_s=env.now())
            break
        if wall_limit is not None and time.monotonic() - t_wall > wall_limit:
            env.cancel()
            task = TaskResult(success=False, stop_reason="evaluator_timeout",
                              message=f"wall-clock limit {wall_limit:.0f}s reached (sim time {env.now():.1f}s)", duration_s=env.now())
            break
        env.advance(rc["poll_dt_s"])
    t_end = env.now()
    env.advance(rc["grace_s"])
    rec = env.finish_recording()
    rec.t_end = min(t_end, ep.time_limit_s + rc["cancel_grace_s"])
    wall_s = time.monotonic() - t_wall
    reason = looks_like_infra(task, cfg["infra"]["error_patterns"])
    base = {"episode_id": ep.id, "repeat": repeat, "seed": seed, "scene": ep.scene, "split": ep.split}
    if reason:
        return {**base, "infrastructure_failure": True, "infra_reason": f"{task.stop_reason}: {task.message[:200]}",
                "infra_pattern": reason}
    res = score_episode(ep, rec, task, cfg, optimum, optimum.grid_for(ep.scene), read_trace(trace_path) or None)
    res.update(repeat=repeat, seed=seed, wall_s=round(wall_s, 2), mode=mode, mode_overridden=bool(opts.force_mode and opts.force_mode != ep.mode),
               difficulty=ep.difficulty or derive_difficulty(res.get("optimal_path_m"), cfg),
               start_xy=[round(start["x"], 3), round(start["y"], 3)], world=rec.scene_info.get("name") if rec.scene_info else None)
    if opts.save_tracks:
        rec.save(run.tracks / f"{safe_name(key)}.json.gz")
    return res


def run_suite(env: Env, episodes: list[Episode], cfg: dict, run: RunDir, opts: SuiteOptions, optimum: OptimumProvider,
              manifest: dict | None = None) -> dict:
    """Run every (repeat, episode) not yet in episodes.jsonl. Append-only and crash safe; returns the summary."""
    log = opts.log
    run.create()
    if manifest is not None:
        old = run.read_manifest()
        if old is not None and opts.resume:
            diffs = comparable(old, manifest)
            if diffs and not opts.force_resume:
                raise RuntimeError("cannot resume run '%s': it was started with a different setup (%s). Use a new "
                                   "--run-name, or --force-resume if you accept non-comparable results."
                                   % (opts.run_name, "; ".join(diffs)))
        if old is None or not opts.resume:
            run.write_json(run.manifest, manifest)
    done = run.completed() if opts.resume else set()
    todo = [(r, e) for r in range(opts.repeats) for e in episodes if (e.id, r) not in done]
    planned = opts.repeats * len(episodes)
    log(f"[{opts.run_name}] {len(done)} done, {len(todo)} to run ({len(episodes)} episodes x {opts.repeats} repeats)")
    rc = cfg["run"]
    consecutive_infra = 0
    worlds: set[str] = set()
    try:
        for n, (repeat, ep) in enumerate(todo, 1):
            key = episode_key(ep.id, repeat)
            seed = opts.seed + repeat
            res = None
            attempts = 0
            for attempts in range(1, rc["max_infra_retries"] + 2):
                reason = None
                try:
                    res = run_attempt(env, ep, repeat, seed, cfg, run, opts, optimum)
                    if res.get("infrastructure_failure"):
                        reason = res["infra_reason"]
                except InfraError as exc:
                    reason = f"{exc.kind}: {exc}"
                if reason is None:
                    break
                run.append_infra({"episode_id": ep.id, "repeat": repeat, "attempt": attempts, "reason": reason,
                                  "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                log(f"[{key}] infrastructure failure (attempt {attempts}/{rc['max_infra_retries'] + 1}): {reason}")
                res = None
                if attempts <= rc["max_infra_retries"]:
                    opts.sleep(rc["infra_backoff_s"])
                    env.recover(cfg)
            if res is None:
                res = {"episode_id": ep.id, "repeat": repeat, "seed": seed, "scene": ep.scene, "split": ep.split,
                       "infrastructure_failure": True, "infra_reason": reason, "status": "infra_failed",
                       "success": False}
                consecutive_infra += 1
            else:
                res["status"] = "done"
                consecutive_infra = 0
                if res.get("world"):
                    worlds.add(res["world"])
            res["attempts"] = attempts
            res["run_name"] = opts.run_name
            run.append_result(res)
            log(f"[{n}/{len(todo)}] {key}: " + (
                f"INFRA FAILED ({res['infra_reason'][:80]})" if res["status"] == "infra_failed" else
                f"success={res['success']} oracle={res.get('oracle_success')} dtg={res.get('final_distance_m')} "
                f"spl={res.get('spl')} t={res.get('duration_s')}s cat={res.get('failure_category')}"))
            if consecutive_infra >= rc.get("abort_after_infra_episodes", 2):
                raise SuiteAborted(f"{consecutive_infra} consecutive episodes failed for infrastructure reasons; "
                                   f"fix the bench and re-run with the same --run-name to resume")
    finally:
        info = env.world_info()
        if manifest is not None and (worlds or info):
            m = run.read_manifest() or manifest
            m["worlds"] = sorted(set(m.get("worlds", [])) | worlds | ({info["name"]} if info.get("name") else set()))
            run.write_json(run.manifest, m)
        summary = build_summary(run.results(), run.infra_failures(), run.read_manifest(), planned, cfg.get("report", {}).get("n_boot", 2000))
        run.write_json(run.summary, summary)
    log(f"[{opts.run_name}] summary: {summary['scored_runs']} scored, {summary['infra_failed_runs']} infra-failed; "
        f"SR={summary['metrics'].get('success', {}).get('mean')}")
    return summary

