"""NaVILA-style step data for the instruction follower: (history frames, current frame, instruction) -> next primitive.

Source: recorded camera frames with GT poses (``run.frames.enabled`` in configs/eval/eval.yaml; stored in tracks/*.json.gz) of
successful episodes. The executed path is converted into the language actions of NaVILA / INTERFACES.md section 5:
``move forward N cm`` (25 cm steps, merged up to 3), ``turn left|right N degrees`` (15 degree steps, merged up to 3), ``stop``.
Whatever policy produced the episode (a Nav2 goal_pose run, a teleoperator, the reasoner), the labels describe what the robot
actually did, so only episodes that reached the goal are used. Held-out episodes are never exported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..core.geometry import SE2
from ..core.recording import Recording
from ..core.runlog import RunDir
from ..core.runner import safe_name
from .sft import split_scenes

PROMPT = ("Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations: {history} "
          "and current observation: <image>. Your assigned task is: \"{instruction}\" Analyze this series of images to decide your next move, "
          "which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if the task is completed.")


def action_text(a: dict) -> str:
    if a["type"] == "stop":
        return "The next action is stop."
    if a["type"] == "move_forward":
        return f"The next action is move forward {a['value']} cm."
    return f"The next action is turn {'left' if a['type'] == 'turn_left' else 'right'} {a['value']} degrees."


def trajectory_to_primitives(poses: list[tuple[float, float, float]], fwd_cm: int = 25, turn_deg: int = 15, max_merge: int = 3) -> list[tuple[int, dict]]:
    """Convert consecutive frame poses (x, y, yaw) into [(decision_frame_index, action)].

    Each step between two frames is a turn (|yaw change| at least half a turn step and larger, in steps, than the forward
    motion), a forward move (at least a quarter of a forward step) or idle. Consecutive steps of the same kind (and turn
    direction) are summed, rounded to whole steps and merged up to ``max_merge`` steps per action, as NaVILA does. The
    decision frame of an action is the frame where its motion starts. A final ``stop`` is attached to the last frame."""
    out: list[tuple[int, dict]] = []
    if len(poses) < 2:
        return out
    fwd_m, ang = fwd_cm / 100.0, math.radians(turn_deg)
    grp: dict | None = None

    def flush():
        nonlocal grp
        if grp is not None:
            n = grp["cum"] / (ang if grp["kind"] == "turn" else fwd_m)
            if abs(n) >= 0.5:
                steps = int(min(max_merge, max(1, round(abs(n)))))
                if grp["kind"] == "turn":
                    out.append((grp["start"], {"type": "turn_left" if n > 0 else "turn_right", "value": steps * turn_deg}))
                else:
                    out.append((grp["start"], {"type": "move_forward", "value": steps * fwd_cm}))
        grp = None

    for i in range(len(poses) - 1):
        rel = SE2(*poses[i]).inverse().compose(SE2(*poses[i + 1]))
        f, t = max(rel.x, 0.0), rel.yaw
        if abs(t) >= 0.5 * ang and abs(t) / ang >= f / fwd_m:
            kind, val = "turn", t
        elif f >= 0.25 * fwd_m:
            kind, val = "fwd", f
        else:
            continue
        if grp is not None and (grp["kind"] != kind or (kind == "turn" and grp["cum"] * val < 0)):
            flush()
        if grp is not None and abs(grp["cum"] + val) / (ang if kind == "turn" else fwd_m) > max_merge + 0.5:
            flush()                                          # would exceed the merge limit: close the action, start the next one here
        if grp is None:
            grp = {"kind": kind, "start": i, "cum": 0.0}
        grp["cum"] += val
        if abs(grp["cum"]) / (ang if kind == "turn" else fwd_m) >= max_merge - 0.5:
            flush()
    flush()
    out.append((len(poses) - 1, {"type": "stop", "value": 0}))
    return out


def export_navila(run_dirs: list[Path], out_dir: Path, *, history: int = 8, val_fraction: float = 0.2, val_scenes: list[str] | None = None,
                  seed: int = 0, min_frames: int = 6, fwd_cm: int = 25, turn_deg: int = 15, max_stop_share: float = 0.3) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    skipped: dict[str, int] = {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for path in run_dirs:
        rd = RunDir(path)
        for res in rd.results():
            if res.get("status", "done") != "done":
                continue
            if res.get("split") == "heldout":
                skip("heldout_never_exported")
                continue
            if not res.get("success"):
                skip("not_success")
                continue
            tp = rd.tracks / f"{safe_name(res['episode_id'] + '#r' + str(res['repeat']))}.json.gz"
            if not tp.exists():
                skip("no_track")
                continue
            frames = [f for f in Recording.load(tp).frames if Path(f["path"]).exists()]
            if len(frames) < min_frames:
                skip("too_few_frames")
                continue
            poses = [tuple(f["gt"][:3]) for f in frames]
            acts = trajectory_to_primitives(poses, fwd_cm, turn_deg)
            anchors = [i for i, _ in acts]
            for k, (i, a) in enumerate(acts):
                prev = anchors[:k]
                pick = [prev[int(round(x))] for x in np.linspace(0, len(prev) - 1, min(history, len(prev)))] if prev else []
                imgs = [frames[p]["path"] for p in pick] + [frames[i]["path"]]
                prompt = PROMPT.format(history=" ".join("<image>" for _ in pick) or "(none)", instruction=res.get("instruction", ""))
                samples.append({"id": f"{rd.path.name}/{res['episode_id']}#r{res['repeat']}#s{k}", "episode": res["episode_id"], "scene": res["scene"],
                                "instruction": res.get("instruction", ""), "images": imgs, "prompt": prompt, "answer": action_text(a), "action": a})
    # stop rebalancing: NaVILA rebalances "stop"; keep at most max_stop_share of the samples as stops
    stops = [s for s in samples if s["action"]["type"] == "stop"]
    moves = [s for s in samples if s["action"]["type"] != "stop"]
    cap = int(max_stop_share * len(samples))
    if len(stops) > cap:
        skip("stop_rebalanced")
        stops = stops[:cap]
    samples = moves + stops
    val, leaky = split_scenes([s["scene"] for s in samples], val_fraction, seed, val_scenes)
    n = {"train": 0, "val": 0}
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as ft, open(out_dir / "val.jsonl", "w", encoding="utf-8") as fv:
        for s in samples:
            part = "val" if s["scene"] in val else "train"
            rec = {**s, "conversations": [{"from": "human", "value": s["prompt"]}, {"from": "gpt", "value": s["answer"]}]}
            (fv if part == "val" else ft).write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[part] += 1
    stats = {"samples": len(samples), "written": n, "val_scenes": sorted(val), "leaky_single_scene_split": leaky, "skipped": skipped,
             "action_counts": {t: sum(1 for s in samples if s["action"]["type"] == t) for t in ("move_forward", "turn_left", "turn_right", "stop")}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
