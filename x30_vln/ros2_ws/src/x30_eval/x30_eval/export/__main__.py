"""python -m x30_eval.export {sft|navila|perception} --runs results/a results/b --out DIR [options]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="x30_eval.export", description=__doc__)
    ap.add_argument("kind", choices=["sft", "navila", "perception"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["hf", "llamafactory"], default="hf", help="sft only")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-scenes", default="", help="comma list; default: deterministic hash split by scene")
    ap.add_argument("--include", choices=["success", "success_instance", "all"], default="success", help="sft only")
    ap.add_argument("--max-collisions", type=int, default=0, help="sft only; -1 disables the filter")
    ap.add_argument("--tools-json", default="", help="sft only: tool schema list stored with every sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    runs, out = [Path(r) for r in a.runs], Path(a.out)
    vs = [s for s in a.val_scenes.split(",") if s] or None
    if a.kind == "sft":
        from .sft import export_sft
        stats = export_sft(runs, out, fmt=a.format, val_fraction=a.val_fraction, val_scenes=vs, seed=a.seed, include=a.include,
                           max_collisions=None if a.max_collisions < 0 else a.max_collisions, tools_json=Path(a.tools_json) if a.tools_json else None)
    elif a.kind == "navila":
        from .navila import export_navila
        stats = export_navila(runs, out, val_fraction=a.val_fraction, val_scenes=vs, seed=a.seed)
    else:
        from .perception import export_perception
        stats = export_perception(runs, out)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
