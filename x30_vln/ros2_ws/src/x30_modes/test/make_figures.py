"""Figures for docs/figures/modes from results/modes/*.json (run on the host: python make_figures.py)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
RES, FIG = ROOT / "results" / "modes", ROOT / "docs" / "figures" / "modes"
FIG.mkdir(parents=True, exist_ok=True)

fake = json.loads((RES / "integration_all.json").read_text())["measurements"]
sim = json.loads((RES / "sim_check.json").read_text())

# 1. guard stopping gaps in the fake-robot tests
rows = [r for r in fake["guard_stopping"] if not r["case"].startswith("sudden")]
sudden = [r for r in fake["guard_stopping"] if r["case"].startswith("sudden")]
fig, ax = plt.subplots(figsize=(8, 3.8))
labels = [f"{r['case'].replace('_wall', ' (past chair)')}\n{r['speed']:+.1f} m/s" for r in rows] + [f"box appears\n{r['appear_gap_m']} m ahead" for r in sudden]
vals = [r["stop_gap_m"] for r in rows + sudden]
ax.bar(range(len(vals)), vals, color="#3b7dd8")
ax.axhline(0.20, color="#c0392b", ls="--", label="configured stop margin 0.20 m")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("gap robot edge to obstacle at standstill (m)")
ax.set_title("Safety guard, fake robot: standstill gap (no collisions in any case)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "m1_guard_fake_robot.png", dpi=130)

# 2. primitive accuracy: fake robot and simulator
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, unit in zip(axs, ("cm", "deg")):
    f = [r for r in fake["primitives"] if (unit in r["cmd"])]
    s = [r for r in sim["primitives"] if r["unit"] == unit]
    ax.plot(range(len(f)), [r["error"] for r in f], "o", label="fake robot (odometry-exact truth)")
    ax.plot(range(len(f), len(f) + len(s)), [r["error"] for r in s], "s", label="simulator (ground truth)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(f"error ({unit})")
    ax.set_xlabel("trial")
    ax.set_title("translation error" if unit == "cm" else "rotation error")
axs[0].legend(fontsize=7)
fig.suptitle("Velocity primitives: commanded vs measured (100 cm, 50 cm, 30 cm; 90, 45 deg)")
fig.tight_layout()
fig.savefig(FIG / "m2_primitive_accuracy.png", dpi=130)

# 3. simulator wall approach
tr = json.loads((RES / "approach_trace_0.6.json").read_text())
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
t = [r["t"] for r in tr]
ax[0].plot(t, [r["x"] for r in tr], label="x (ground truth)")
ax[0].set_ylabel("x (m), wall face at x = 5.9")
ax[1].plot(t, [r["vx_out"] for r in tr], label="/cmd_vel vx (guard output)")
ax[1].plot(t, [min(r["guard_clear"], 3.0) for r in tr], label="guard clearance (m, capped 3)")
ax[1].set_xlabel("time (s)")
ax[1].legend(fontsize=8)
ax[0].set_title("Simulator (kinematic), raw lidar guard, 0.6 m/s command toward the east wall")
fig.tight_layout()
fig.savefig(FIG / "m3_sim_wall_approach.png", dpi=130)
print("figures written to", FIG)
