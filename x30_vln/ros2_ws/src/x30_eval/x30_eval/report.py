"""Reports: python -m x30_eval.report <run dir> [<run dir> ...] [--out DIR] [--baseline NAME]

Per run: headline metrics (mean, bootstrap 95% CI over episodes, std over repeats), breakdowns by scene / difficulty /
mode / goal type, failure categories with example episodes, infrastructure bookkeeping. With several runs: side-by-side
table and paired-bootstrap comparisons against the baseline. Writes report.md, CSV tables and PNG plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from .core import stats as S
from .core.config import load_eval_config, project_root
from .core.recording import Recording
from .core.runlog import HELDOUT_LOCK, RunDir
from .core.summary import breakdown, failure_table, scored

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
HEADLINE = ["success", "oracle_success", "spl", "final_distance_m", "duration_s", "collisions", "interventions", "agent_turns"]
COMPARE = ["success", "oracle_success", "spl", "final_distance_m", "duration_s", "collisions", "agent_turns"]


class LoadedRun:
    def __init__(self, path: Path):
        self.dir = RunDir(path)
        self.name = path.name
        self.manifest = self.dir.read_manifest() or {}
        self.results = self.dir.results()
        self.infra = self.dir.infra_failures()
        self.flat = scored(self.results)
        self.exhausted = [r for r in self.results if r.get("status") == "infra_failed"]


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def metric_cell(sm: dict | None, pct: bool = False) -> str:
    if not sm:
        return "n/a"
    nd = 3 if not pct else 1
    k = 100.0 if pct else 1.0
    cell = f"{sm['mean'] * k:.{nd}f} [{sm['ci_lo'] * k:.{nd}f}, {sm['ci_hi'] * k:.{nd}f}]"
    if sm.get("std_over_repeats") is not None:
        cell += f" sd{sm['std_over_repeats'] * k:.{nd}f}"
    return cell


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def heldout_note(results_root: Path, run: LoadedRun) -> str | None:
    lock = results_root / HELDOUT_LOCK
    if not lock.exists():
        return None
    for i, r in enumerate(json.loads(lock.read_text(encoding="utf-8"))["runs"], 1):
        if r["run"] == run.name:
            return f"HELD-OUT evaluation number {i} (recorded {r['time_utc']}). Do not tune on these numbers."
    return None


def run_section(run: LoadedRun, sm: dict[str, dict], results_root: Path, n_boot: int) -> str:
    man, args = run.manifest, run.manifest.get("args", {})
    lines = [f"## Run `{run.name}`", ""]
    note = heldout_note(results_root, run)
    if note:
        lines += [f"**{note}**", ""]
    git = man.get("git") or {}
    lines.append(
        f"- split `{args.get('split')}`, env `{args.get('env')}`, repeats {args.get('repeats')}, seed {args.get('seed')}, "
        f"forced mode `{args.get('force_mode') or '-'}`, episodes file `{Path(str(man.get('episodes_file', ''))).name}` (sha {man.get('episodes_sha256')})")
    lines.append(f"- config hash `{man.get('config_hash')}`, prompts hash `{man.get('prompts_hash')}`, overlay hash `{man.get('overlay_hash')}`, "
                 f"git `{(git.get('commit') or 'n/a')[:10]}`{' (dirty)' if git.get('dirty') else ''}, worlds {man.get('worlds')}")
    lines.append(f"- scored runs {len(run.flat)}, episodes {len({r['episode_id'] for r in run.flat})}; infrastructure incidents "
                 f"{len(run.infra)} (re-run, not counted); runs lost to infrastructure after all retries {len(run.exhausted)} (EXCLUDED)")
    if run.exhausted:
        lines.append(f"- **WARNING: {len(run.exhausted)} run(s) are missing because of infrastructure failures; the metrics describe fewer episodes than planned.**")
    bases = {r.get("spl_basis") for r in run.flat}
    if "euclidean" in bases or "unavailable" in bases:
        lines.append("- SPL basis: some episodes used a straight-line optimum (no GT grid): SPL is pessimistic for those. See `spl_basis` in the CSV.")
    lines += ["", "### Headline metrics", "", "Mean [95% bootstrap CI over episodes] and sd = standard deviation over repeats of the suite mean.", ""]
    labels = {k: (lab, hb) for k, lab, hb in S.METRICS}
    rows = []
    for key in [k for k, _, _ in S.METRICS if k in sm]:
        pct = key in ("success", "oracle_success", "spl", "ndtw", "sdtw", "map_recall_covered", "map_duplicate_rate")
        rows.append([labels[key][0] + (" [%]" if pct else ""), metric_cell(sm[key], pct), sm[key]["n_episodes"]])
    lines.append(md_table(["metric", "value", "episodes"], rows))
    for title, key in (("scene", "scene"), ("difficulty", "difficulty"), ("mode", "mode"), ("goal type", "goal_type")):
        bd = breakdown(run.flat, key, n_boot)
        if len(bd) <= 1 and key != "scene":
            continue
        rows = [[name, v["n_episodes"], v["n_runs"], fmt((v.get("success") or {}).get("mean")), fmt((v.get("oracle_success") or {}).get("mean")),
                 fmt((v.get("spl") or {}).get("mean")), fmt((v.get("final_distance_m") or {}).get("mean"), 2),
                 fmt((v.get("duration_s") or {}).get("mean"), 1)] for name, v in bd.items()]
        lines += ["", f"### By {title}", "", md_table([title, "episodes", "runs", "SR", "OSR", "SPL", "DTG [m]", "time [s]"], rows)]
    ft = failure_table(run.flat, 3)
    n = len(run.flat)
    lines += ["", "### Failure categories (heuristic classifier, see EVALUATION.md)", ""]
    rows = [[c, k, f"{100.0 * k / n:.0f}%" if n else "n/a"] for c, k in sorted(ft["counts"].items(), key=lambda kv: -kv[1])]
    lines.append(md_table(["category", "runs", "share"], rows))
    for cat, exs in ft["examples"].items():
        lines.append(f"- `{cat}` examples: " + "; ".join(f"{e['episode_id']}#r{e['repeat']} ({'; '.join(e['evidence'] or [])})" for e in exs))
    return "\n".join(lines)


def comparison_section(runs: list[LoadedRun], summaries: dict[str, dict], baseline: LoadedRun, n_boot: int) -> tuple[str, list[list]]:
    lines = ["## Comparison", "", f"Baseline: `{baseline.name}`. Differences are (run - baseline) on episodes both ran; "
             "paired bootstrap over episodes (95% CI, two-sided p). Few episodes give wide intervals: read the CI, not only the mean.", ""]
    labels = {k: lab for k, lab, _ in S.METRICS}
    header = ["metric"] + [r.name for r in runs]
    rows = [[labels[k]] + [fmt((summaries[r.name].get(k) or {}).get("mean")) for r in runs] for k in COMPARE]
    lines.append(md_table(header, rows))
    csv_rows: list[list] = []
    for r in runs:
        if r is baseline:
            continue
        lines += ["", f"### `{r.name}` vs `{baseline.name}`", ""]
        rows = []
        for k in COMPARE:
            pb = S.paired_bootstrap(baseline.flat, r.flat, k, n_boot)
            if pb["n_pairs"] == 0:
                continue
            rows.append([labels[k], pb["n_pairs"], fmt(pb["mean_a"]), fmt(pb["mean_b"]), f"{pb['diff']:+.3f}",
                         f"[{pb['ci_lo']:+.3f}, {pb['ci_hi']:+.3f}]", fmt(pb["p_value"], 3),
                         f"{pb['episodes_b_better']}/{pb['episodes_a_better']}"])
            csv_rows.append([r.name, baseline.name, k, pb["n_pairs"], pb["mean_a"], pb["mean_b"], pb["diff"], pb["ci_lo"], pb["ci_hi"], pb["p_value"]])
        lines.append(md_table(["metric", "pairs", "baseline", "run", "diff", "95% CI", "p", "episodes up/down"], rows))
        if len({r.manifest.get("episodes_sha256"), baseline.manifest.get("episodes_sha256")}) > 1:
            lines.append("\n_Note: the runs used different episode files; only common episode ids are paired._")
        if r.manifest.get("config_hash") == baseline.manifest.get("config_hash") and r.manifest.get("overlay_hash") == baseline.manifest.get("overlay_hash"):
            lines.append("\n_Note: identical config and overlay hashes: differences come from run-to-run variability, not from a configuration change._")
    return "\n".join(lines), csv_rows


# ---------------------------------------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "font.size": 9})
    return plt


def plot_headline(runs, summaries, out: Path) -> None:
    plt = _plt()
    keys = [("success", "SR"), ("oracle_success", "OSR"), ("spl", "SPL")]
    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(runs), 3.4))
    w = 0.8 / len(runs)
    for i, r in enumerate(runs):
        xs, ys, lo, hi = [], [], [], []
        for j, (k, _) in enumerate(keys):
            sm = summaries[r.name].get(k)
            if sm:
                xs.append(j + i * w - 0.4 + w / 2)
                ys.append(sm["mean"])
                lo.append(sm["mean"] - sm["ci_lo"])
                hi.append(sm["ci_hi"] - sm["mean"])
        ax.bar(xs, ys, w * 0.9, yerr=[lo, hi], capsize=3, color=PALETTE[i % len(PALETTE)], label=r.name)
    ax.set_xticks(range(len(keys)), [lab for _, lab in keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction (95% CI over episodes)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "headline.png")
    plt.close(fig)


def plot_by_scene(runs, out: Path, n_boot: int) -> None:
    plt = _plt()
    scenes = sorted({r_["scene"] for run in runs for r_ in run.flat})
    if not scenes:
        return
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * len(scenes) * len(runs) / 2, 3.4))
    w = 0.8 / len(runs)
    for i, r in enumerate(runs):
        bd = breakdown(r.flat, "scene", n_boot)
        ys = [(bd.get(s, {}).get("success") or {}).get("mean", np.nan) for s in scenes]
        ax.bar(np.arange(len(scenes)) + i * w - 0.4 + w / 2, ys, w * 0.9, color=PALETTE[i % len(PALETTE)], label=r.name)
    ax.set_xticks(range(len(scenes)), scenes, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("SR")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sr_by_scene.png")
    plt.close(fig)


def plot_failures(runs, out: Path) -> None:
    plt = _plt()
    cats = sorted({r_.get("failure_category", "other") for run in runs for r_ in run.flat})
    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(runs), 3.6))
    bottoms = np.zeros(len(runs))
    for c_i, cat in enumerate(cats):
        vals = np.array([sum(1 for r_ in run.flat if r_.get("failure_category") == cat) / max(len(run.flat), 1) for run in runs])
        ax.bar(range(len(runs)), vals, bottom=bottoms, color=PALETTE[c_i % len(PALETTE)], label=cat)
        bottoms += vals
    ax.set_xticks(range(len(runs)), [r.name for r in runs], rotation=15)
    ax.set_ylabel("share of runs")
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "failure_categories.png")
    plt.close(fig)


def plot_dtg_cdf(runs, out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, r in enumerate(runs):
        v = np.sort([x["final_distance_m"] for x in r.flat if x.get("final_distance_m") is not None])
        if len(v):
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=PALETTE[i % len(PALETTE)], label=r.name)
    ax.set_xlabel("final distance to goal [m]")
    ax.set_ylabel("fraction of runs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dtg_cdf.png")
    plt.close(fig)


def plot_trajectories(run: LoadedRun, out: Path, root: Path, cfg: dict, max_n: int = 12) -> None:
    from .core.optimum import OptimumProvider
    from .core.episodes import episode_key
    from .core.runner import safe_name
    plt = _plt()
    opt = OptimumProvider(cfg, root)
    items = []
    for r in run.flat[:max_n]:
        p = run.dir.tracks / f"{safe_name(episode_key(r['episode_id'], r['repeat']))}.json.gz"
        if p.exists():
            items.append((r, Recording.load(p)))
    if not items:
        return
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (r, rec) in zip(axes.ravel(), items):
        ax.axis("on")
        g = opt.grid_for(r["scene"])
        if g is not None:
            h, w = g.occ.shape
            ax.imshow(g.occ, origin="lower", cmap="Greys", extent=[g.origin[0], g.origin[0] + w * g.resolution, g.origin[1], g.origin[1] + h * g.resolution], alpha=0.6)
        gt = rec.gt_array()
        if len(gt):
            ax.plot(gt[:, 1], gt[:, 2], color=PALETTE[0], lw=1.2)
            ax.plot(gt[0, 1], gt[0, 2], "o", color=PALETTE[2], ms=4)
        for o in rec.gt_objects:
            hit = r.get("goal_label") and o["label"] == r["goal_label"]
            ax.plot(o["x"], o["y"], "*" if hit else ".", color=PALETTE[3] if hit else "#666666", ms=8 if hit else 4)
        ax.set_aspect("equal")
        ax.set_title(f"{r['episode_id']}#{r['repeat']} {'ok' if r.get('success') else r.get('failure_category')}", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / f"trajectories_{run.name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def build_report(run_dirs: list[Path], out: Path, baseline: str | None = None, root: Path | None = None, cfg: dict | None = None) -> Path:
    root = root or project_root()
    cfg = cfg or load_eval_config(root=root)
    n_boot = cfg["report"]["n_boot"]
    runs = [LoadedRun(Path(p)) for p in run_dirs]
    for r in runs:
        if not r.results:
            raise SystemExit(f"{r.dir.path}: no episodes.jsonl records")
    out.mkdir(parents=True, exist_ok=True)
    summaries = {r.name: S.summarise_all(r.flat, n_boot, cfg["report"]["seed"]) for r in runs}
    results_root = runs[0].dir.path.parent
    parts = ["# Evaluation report", "", f"Runs: {', '.join('`%s`' % r.name for r in runs)}", ""]
    parts += [run_section(r, summaries[r.name], results_root, n_boot) + "\n" for r in runs]
    csv_cmp: list[list] = []
    if len(runs) > 1:
        base = next((r for r in runs if r.name == baseline), runs[0]) if baseline else runs[0]
        text, csv_cmp = comparison_section(runs, summaries, base, n_boot)
        parts.append(text)
    # CSV
    rows = [[r.name, k, s["mean"], s["ci_lo"], s["ci_hi"], s["std_over_repeats"], s["n_episodes"]] for r in runs for k, s in summaries[r.name].items()]
    write_csv(out / "metrics.csv", ["run", "metric", "mean", "ci_lo", "ci_hi", "std_over_repeats", "n_episodes"], rows)
    if csv_cmp:
        write_csv(out / "comparison.csv", ["run", "baseline", "metric", "pairs", "mean_baseline", "mean_run", "diff", "ci_lo", "ci_hi", "p"], csv_cmp)
    for r in runs:
        keys = sorted({k for x in r.flat for k, v in x.items() if not isinstance(v, (list, dict))})
        write_csv(out / f"episodes_{r.name}.csv", keys, [[x.get(k) for k in keys] for x in r.flat])
        for kind in ("scene", "difficulty", "mode"):
            bd = breakdown(r.flat, kind, n_boot)
            write_csv(out / f"breakdown_{kind}_{r.name}.csv", [kind, "episodes", "runs", "sr", "osr", "spl", "dtg_m", "time_s"],
                      [[n, v["n_episodes"], v["n_runs"]] + [(v.get(m) or {}).get("mean") for m in ("success", "oracle_success", "spl", "final_distance_m", "duration_s")]
                       for n, v in bd.items()])
        ft = failure_table(r.flat, 50)
        write_csv(out / f"failures_{r.name}.csv", ["category", "count"], sorted(ft["counts"].items()))
    # plots
    plot_headline(runs, summaries, out)
    plot_by_scene(runs, out, n_boot)
    plot_failures(runs, out)
    plot_dtg_cdf(runs, out)
    for r in runs:
        plot_trajectories(r, out, root, cfg)
    parts += ["", "## Files", "", "Plots: `headline.png`, `sr_by_scene.png`, `failure_categories.png`, `dtg_cdf.png`, `trajectories_<run>.png`. "
              "Tables: `metrics.csv`, `comparison.csv`, `episodes_<run>.csv`, `breakdown_*.csv`, `failures_*.csv`."]
    (out / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out / "report.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m x30_eval.report", description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (results/<name>)")
    ap.add_argument("--out", default="", help="output directory (default <results>/_reports/<run names>)")
    ap.add_argument("--baseline", default="", help="run name used as baseline in comparisons (default: first)")
    ap.add_argument("--config", default="")
    args = ap.parse_args(argv)
    dirs = [Path(r) for r in args.runs]
    out = Path(args.out) if args.out else dirs[0].parent / "_reports" / "+".join(d.name for d in dirs)
    cfg = load_eval_config(args.config or None)
    path = build_report(dirs, out, args.baseline or None, cfg=cfg)
    print(f"report written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
