#!/usr/bin/env python3
"""Figures for docs/reports/agent.md from results/agent_live/* (run after live_eval.py and run_fake_world.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "results" / "agent_live"
OUT = ROOT / "docs" / "figures" / "agent"
OUT.mkdir(parents=True, exist_ok=True)


def fig_live_eval() -> None:
    rep = json.loads((RES / "live_eval" / "report.json").read_text())
    rows = [json.loads(l) for l in (RES / "live_eval" / "states_auto.jsonl").read_text().splitlines() if l.strip()]
    types = sorted({r["type"] for r in rows})
    raw = [sum(r["correct"] for r in rows if r["type"] == t) / sum(1 for r in rows if r["type"] == t) for t in types]
    har = [sum(r.get("correct_with_harness", False) for r in rows if r["type"] == t) / sum(1 for r in rows if r["type"] == t) for t in types]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    x = range(len(types))
    ax[0].bar([i - 0.2 for i in x], raw, 0.4, label="raw reply", color="#4c78a8")
    ax[0].bar([i + 0.2 for i in x], har, 0.4, label="after harness fixes", color="#f58518")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([t.split("_", 1)[0] + " " + t.split("_", 1)[1].replace("_", " ") for t in types], fontsize=7, rotation=30, ha="right")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("fraction of 9 states with the best next step")
    ax[0].set_title("Qwen3-VL-8B, single-turn hand-made states (n=72, tool_choice=auto)")
    ax[0].legend(fontsize=8, loc="lower right")
    lat = rep.get("cold_warm", {})
    il = rep.get("image_latency", {})
    labels = ["cold prefix", "cached prefix"] + [f"imgs: {k}" for k in il]
    vals = [lat.get("cold_mean_s", 0), lat.get("warm_mean_s", 0)] + [v["latency_mean_s"] for v in il.values()]
    ax[1].barh(range(len(vals)), vals, color=["#e45756", "#54a24b"] + ["#72b7b2"] * len(il))
    ax[1].set_yticks(range(len(vals)))
    ax[1].set_yticklabels(labels, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("seconds per turn (3.4k prompt tokens)", fontsize=8)
    ax[1].set_title("Latency per turn")
    fig.tight_layout()
    fig.savefig(OUT / "a1_live_eval.png", dpi=110)
    plt.close(fig)


def fig_fake_episodes() -> None:
    runs = [("object_nav, text only", "fake_objnav"), ("object_nav, map+camera", "fake_objnav_images"), ("object_nav, verify wrong 20%", "fake_objnav_noise"),
            ("velocity_follow, target only", "fake_follow"), ("velocity_follow, route (dev seeds)", "fake_follow_route"), ("velocity_follow, route (fresh seeds)", "fake_follow_route_fresh")]
    runs = [(n, d) for n, d in runs if (RES / d / "episodes.jsonl").exists()]
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.4))
    ax = ax.ravel()
    for a, (name, d) in zip(ax, runs):
        rows = [json.loads(l) for l in (RES / d / "episodes.jsonl").read_text().splitlines() if l.strip()]
        cols = ["#54a24b" if r["gt_success"] else "#e45756" for r in rows]
        a.bar(range(len(rows)), [r["turns"] for r in rows], color=cols)
        sr = sum(r["gt_success"] for r in rows)
        a.set_title(f"{name}\n{sr}/{len(rows)} reached the target (FAKE world)", fontsize=9)
        a.set_xlabel("episode")
        a.set_ylabel("agent turns")
    fig.tight_layout()
    fig.savefig(OUT / "a2_fake_world_episodes.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    fig_live_eval()
    fig_fake_episodes()
    print("figures in", OUT)
