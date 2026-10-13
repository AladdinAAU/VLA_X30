"""Episode runner CLI:  ros2 run x30_eval run_episodes --episodes <file> --run-name <name> [options]

Also runnable without ROS against the fake environment:  python -m x30_eval.run_episodes --env fake ...
Exit codes: 0 finished, 1 error, 2 held-out guard refused, 3 aborted (bench down; re-run with the same --run-name to resume).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

from .core import episodes as E
from .core.config import load_eval_config, project_root
from .core.manifest import build_manifest
from .core.optimum import OptimumProvider
from .core.overlay import ENV_VAR, deep_merge, dotted_to_overlay, load_overlay_files
from .core.runlog import HeldoutGuard, HeldoutGuardError, RunDir
from .core.runner import SuiteAborted, SuiteOptions, run_suite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_episodes", description=__doc__.split("\n")[0])
    p.add_argument("--episodes", required=True, help="episode file (YAML or JSONL, INTERFACES.md section 6)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--repeats", type=int, default=1, help="independent repeats of every episode (seed = --seed + repeat)")
    p.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    p.add_argument("--final", action="store_true", help="REQUIRED for heldout/all: the one-time final evaluation")
    p.add_argument("--allow-repeat-heldout", action="store_true", help="another held-out evaluation after a first one (recorded)")
    p.add_argument("--mode", default="", help="only episodes of these modes (comma list: object_nav,velocity_follow,goal_pose)")
    p.add_argument("--force-mode", default="", help="run every selected episode in this mode (e.g. goal_pose as a trivial policy)")
    p.add_argument("--episode-ids", default="", help="comma list of episode ids")
    p.add_argument("--scenes", default="", help="comma list of scene ids")
    p.add_argument("--limit", type=int, default=0, help="run only the first N selected episodes (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", choices=["ros", "fake"], default="ros")
    p.add_argument("--wall-time", action="store_true", help="use wall time instead of /clock for the task clock")
    p.add_argument("--config", default="", help="eval config file (default configs/eval/eval.yaml)")
    p.add_argument("--overlay", action="append", default=[], help="YAML overlay file (repeatable); also exported as X30_CONFIG_OVERLAY")
    p.add_argument("--set", action="append", default=[], metavar="FILE.KEY=VALUE",
                   help="inline overlay entry, YAML value, e.g. modes/object_nav.standoff_m=1.2")
    p.add_argument("--params", default="{}", help="JSON dict of extra Navigate params for every episode")
    p.add_argument("--results-dir", default="", help="output root (default <project>/results)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--force-resume", action="store_true", help="resume although configs/episodes changed (results become non-comparable)")
    p.add_argument("--no-tracks", action="store_true", help="do not save per-episode recordings")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--fake-inject", default="{}", help="(fake env) JSON {episode_id: [reset_error|task_error|health_error, ...]}")
    return p


def parse_overlay(args) -> dict:
    overlay = load_overlay_files(args.overlay) if args.overlay else {}
    inline = {}
    for item in args.set:
        key, _, val = item.partition("=")
        inline = deep_merge(inline, dotted_to_overlay({key: yaml.safe_load(val)}))
    return deep_merge(overlay, inline)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        from rclpy.utilities import remove_ros_args
        argv = remove_ros_args(args=["x"] + argv)[1:]
    except ImportError:
        pass
    args = build_parser().parse_args(argv)
    root = project_root()
    results_root = Path(args.results_dir) if args.results_dir else root / "results"
    overlay = parse_overlay(args)
    cfg = load_eval_config(args.config or None, overlay=overlay)
    eps = E.select(E.load_episodes(args.episodes), args.split, [m for m in args.mode.split(",") if m] or None,
                   [i for i in args.episode_ids.split(",") if i] or None, [s for s in args.scenes.split(",") if s] or None)
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        print("no episodes selected", file=sys.stderr)
        return 1
    print(f"{len(eps)} episodes x {args.repeats} repeats, split={args.split}, env={args.env}")
    if args.dry_run:
        for e in eps:
            print(f"  {e.id:30s} {e.scene:20s} {args.force_mode or e.mode:15s} {e.split:8s} limit={e.time_limit_s:.0f}s")
        return 0

    try:
        HeldoutGuard(results_root).check_and_register(
            args.run_name, args.split, args.final, args.allow_repeat_heldout,
            {"episodes_file": str(args.episodes), "n_episodes": len(eps), "repeats": args.repeats})
    except HeldoutGuardError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    run = RunDir(results_root / args.run_name).create()
    if overlay:
        (run.path / "overlay.yaml").write_text(yaml.safe_dump(overlay, sort_keys=True), encoding="utf-8")
        os.environ[ENV_VAR] = str(run.path / "overlay.yaml")
    arg_record = {**{k: v for k, v in vars(args).items() if k not in ("overlay", "set")}, "overlay_files": args.overlay, "set": args.set}
    manifest = build_manifest(root, cfg, args.run_name, arg_record, args.episodes, [e.id for e in eps], overlay)
    opts = SuiteOptions(run_name=args.run_name, repeats=args.repeats, seed=args.seed, force_mode=args.force_mode or None,
                        params=json.loads(args.params), resume=not args.no_resume, force_resume=args.force_resume,
                        save_tracks=not args.no_tracks)
    optimum = OptimumProvider(cfg, root)
    if args.env == "fake":
        from .testing.fake_env import FakeEnv, fake_optimum
        optimum = fake_optimum(cfg)
        env = FakeEnv(overlay=overlay, inject=json.loads(args.fake_inject), seed=args.seed)
        print("NOTE: --env fake is a scripted toy world; results only exercise the evaluation code.")
    else:
        from .env.ros_env import RosEnv
        env = RosEnv(cfg, use_sim_time=not args.wall_time)
    t0 = time.time()
    try:
        summary = run_suite(env, eps, cfg, run, opts, optimum, manifest)
    except SuiteAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted; completed episodes are saved, re-run with the same --run-name to resume", file=sys.stderr)
        return 1
    finally:
        env.close()
    m = summary["metrics"]
    print(f"done in {time.time() - t0:.0f}s: SR={m.get('success', {}).get('mean')} SPL={m.get('spl', {}).get('mean')} "
          f"-> {run.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
