"""python -m x30_eval.sweep {run|confirm|finalize|status} --spec spec.yaml --episodes episodes.yaml [--env fake|ros]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.config import load_eval_config, project_root
from ..core.episodes import load_episodes
from ..core.optimum import OptimumProvider
from ..core.overlay import check_overlay_against
from ..core.runlog import HeldoutGuardError
from .evaluator import RunnerEvaluator
from .harness import Sweep
from .space import load_spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="x30_eval.sweep", description=__doc__)
    ap.add_argument("command", choices=["run", "confirm", "finalize", "status"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--env", choices=["ros", "fake"], default="ros")
    ap.add_argument("--results-dir", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--n-trials", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="required for `finalize`: evaluates the held-out split (once)")
    args = ap.parse_args(argv)

    root = project_root()
    results = Path(args.results_dir) if args.results_dir else root / "results"
    spec = load_spec(args.spec)
    cfg = load_eval_config(args.config or None)
    study_dir = results / "sweeps" / spec.study
    optimum = OptimumProvider(cfg, root)
    if args.env == "fake":
        from ..testing.fake_env import FakeEnv, fake_optimum
        optimum = fake_optimum(cfg)
        factory = lambda overlay: FakeEnv(overlay=overlay, seed=spec.seed)  # noqa: E731
    else:
        missing = check_overlay_against(root / "configs", spec.overlay_for({p.name: p.low if p.low is not None else (p.choices or [0])[0] for p in spec.params}))
        if missing:
            print(f"sweep parameters not found in configs/: {missing}", file=sys.stderr)
            return 1
        from .evaluator import StackEnvFactory
        factory = StackEnvFactory(cfg, spec.stack_restart_cmd, spec.stack_wait_s, study_dir / "overlays")
    sweep = Sweep(spec, RunnerEvaluator(cfg, optimum, study_dir, factory), load_episodes(args.episodes), study_dir, results)
    try:
        if args.command == "run":
            out = sweep.run(args.n_trials or None)
        elif args.command == "confirm":
            out = sweep.confirm()
        elif args.command == "finalize":
            out = sweep.finalize(args.final)
        else:
            out = json.loads((study_dir / "search.json").read_text(encoding="utf-8"))
    except HeldoutGuardError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
