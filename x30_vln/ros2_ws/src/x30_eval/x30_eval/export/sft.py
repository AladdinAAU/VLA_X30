"""SFT dataset for the reasoner VLM from agent traces (chat + tool-call format), with quality filters.

Each kept reasoner turn becomes one training sample: the exact request messages the model saw (system prompt, task and
state text, camera/map images, short tool-call history) and, as the target, the tool call the agent made. Only turns of
episodes that succeeded (or pass the filters below) are used, and held-out episodes are never exported.

Formats
  hf            Hugging Face chat format used by TRL / Unsloth for Qwen3-VL: {"messages": [...], "images": [...]} where image
                parts are {"type": "image", "image": "<path relative to the dataset dir>"} and tool-call arguments are dicts.
  llamafactory  ShareGPT style for LLaMA-Factory: system / conversations (human, function_call, observation) / images / tools.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from ..core.runlog import RunDir
from ..core.runner import safe_name
from ..core.traces import read_trace


def split_scenes(scenes: list[str], val_fraction: float, seed: int, val_scenes: list[str] | None = None) -> tuple[set[str], bool]:
    """(validation scene set, leaky). With a single scene the split falls back to episodes and is flagged leaky."""
    scenes = sorted(set(scenes))
    if val_scenes:
        return set(val_scenes), False
    if len(scenes) < 2:
        return set(), True
    rank = sorted(scenes, key=lambda s: hashlib.sha1(f"{seed}:{s}".encode()).hexdigest())
    return set(rank[: max(1, math.ceil(val_fraction * len(scenes)))]), False


def _args_dict(a) -> dict:
    if isinstance(a, str):
        try:
            return json.loads(a)
        except json.JSONDecodeError:
            return {}
    return a or {}


def _fill_images(messages: list[dict], images: list[str], fmt: str) -> list[dict] | None:
    """Replace the [image] placeholders (in order) by image references; None when counts do not match."""
    out, k = [], 0
    for m in messages:
        m = copy.deepcopy(m)
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "image_url":
                    if k >= len(images):
                        return None
                    parts.append({"type": "image", "image": images[k]})
                    k += 1
                else:
                    parts.append(p)
            m["content"] = parts
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = _args_dict(tc["function"].get("arguments"))
        out.append(m)
    return out if k == len(images) else None


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join("<image>" if p.get("type") == "image" else p.get("text", "") for p in content or [])


def to_sharegpt(messages: list[dict], images: list[str], tools: list | None) -> dict | None:
    """hf-format messages -> LLaMA-Factory ShareGPT record; None if roles cannot alternate as LLaMA-Factory requires."""
    system = next((_text(m["content"]) for m in messages if m["role"] == "system"), "")
    conv = []
    for m in messages:
        if m["role"] == "user":
            conv.append({"from": "human", "value": _text(m["content"])})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            f = m["tool_calls"][0]["function"]
            conv.append({"from": "function_call", "value": json.dumps({"name": f["name"], "arguments": f["arguments"]}, ensure_ascii=False)})
        elif m["role"] == "assistant":
            conv.append({"from": "gpt", "value": _text(m["content"])})
        elif m["role"] == "tool":
            conv.append({"from": "observation", "value": _text(m["content"])})
    for i, c in enumerate(conv):
        ok = c["from"] in ("human", "observation") if i % 2 == 0 else c["from"] in ("gpt", "function_call")
        if not ok:
            return None
    rec = {"system": system, "conversations": conv, "images": images}
    if tools:
        rec["tools"] = json.dumps(tools, ensure_ascii=False)
    return rec


def export_sft(run_dirs: list[Path], out_dir: Path, *, fmt: str = "hf", val_fraction: float = 0.2, val_scenes: list[str] | None = None,
               seed: int = 0, include: str = "success", max_collisions: int | None = 0, max_turns: int = 40,
               max_parse_error_rate: float = 0.1, min_spl: float = 0.0, max_repeat: int = 2,
               tools_json: Path | None = None, copy_images: bool = True) -> dict:
    """Write train/val files and stats.json. ``include``: success | success_instance | all (all = only filters apply; for
    rejection-sampling experiments, never for held-out claims)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = json.loads(Path(tools_json).read_text(encoding="utf-8")) if tools_json else None
    rejected: Counter = Counter()
    samples: list[dict] = []
    seen: set[str] = set()
    n_eps = 0
    for rd_path in run_dirs:
        rd = RunDir(rd_path)
        for res in rd.results():
            if res.get("status", "done") != "done":
                continue
            n_eps += 1
            why = None
            if res.get("split") == "heldout":
                why = "heldout_never_exported"
            elif include == "success" and not res.get("success"):
                why = "not_success"
            elif include == "success_instance" and not res.get("success_instance"):
                why = "not_named_instance"
            elif max_collisions is not None and (res.get("collisions") or 0) > max_collisions:
                why = "collisions"
            elif (res.get("spl") or 0.0) < min_spl:
                why = "low_spl"
            tr = res.get("trace") or {}
            if why is None and (tr.get("turns") or 0) > max_turns:
                why = "too_many_turns"
            if why is None and tr.get("turns") and (tr.get("parse_errors", 0) / tr["turns"]) > max_parse_error_rate:
                why = "parse_errors"
            trace_id = safe_name(f"{res['episode_id']}#r{res['repeat']}")
            rows = read_trace(rd.traces / f"{trace_id}.jsonl")
            if why is None and not rows:
                why = "no_trace"
            if why:
                rejected[why] += 1
                continue
            run_rows = 0
            prev = None
            repeat_run = 0
            for row in rows:
                if row.get("kind") != "turn" or not row.get("tool") or row.get("parse_ok") is False or row.get("refused") or row.get("success") is False:
                    rejected["turn_filtered"] += 1
                    continue
                if not row.get("messages"):
                    rejected["no_messages"] += 1
                    continue
                sig = (row["tool"], json.dumps(row.get("args", {}), sort_keys=True))
                repeat_run = repeat_run + 1 if sig == prev else 1
                prev = sig
                if repeat_run > max_repeat:
                    rejected["repeated_call"] += 1
                    continue
                imgs = []
                for rel in row.get("images") or []:
                    src = rd.traces / rel
                    if not src.exists():
                        imgs = None
                        break
                    dst_rel = f"images/{rd.path.name}/{rel}"
                    if copy_images:
                        dst = out_dir / dst_rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dst)
                    imgs.append(dst_rel)
                msgs = _fill_images(row["messages"], imgs, fmt) if imgs is not None else None
                if msgs is None:
                    rejected["image_mismatch"] += 1
                    continue
                msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": row["tool"], "arguments": row.get("args", {})}}]})
                last_user = next((_text(m["content"]) for m in reversed(msgs) if m["role"] == "user"), "")
                key = hashlib.sha1((res["scene"] + last_user + json.dumps(sig)).encode()).hexdigest()
                if key in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(key)
                samples.append({"id": f"{rd.path.name}/{trace_id}#t{row.get('turn')}", "run": rd.path.name, "episode": res["episode_id"], "scene": res["scene"],
                                "mode": res.get("mode"), "turn": row.get("turn"), "messages": msgs, "images": imgs, "tools": tools,
                                "meta": {"success": bool(res.get("success")), "spl": res.get("spl"), "episode_turns": tr.get("turns")}})
                run_rows += 1
            rejected["episode_kept"] += 1 if run_rows else 0
    val, leaky = split_scenes([s["scene"] for s in samples], val_fraction, seed, val_scenes)
    if leaky:    # one scene only: split by episode, which leaks scene appearance between train and val
        eps = sorted({s["episode"] for s in samples})
        val_eps = set(eps[:: max(2, round(1 / max(val_fraction, 0.01)))]) if len(eps) > 1 else set()
        is_val = lambda s: s["episode"] in val_eps  # noqa: E731
    else:
        is_val = lambda s: s["scene"] in val  # noqa: E731
    counts = {"train": 0, "val": 0}
    files = {k: open(out_dir / f"{k}.jsonl", "w", encoding="utf-8") for k in counts}
    try:
        for s in samples:
            part = "val" if is_val(s) else "train"
            rec = s
            if fmt == "llamafactory":
                rec = to_sharegpt(s["messages"], s["images"], s["tools"])
                if rec is None:
                    rejected["roles_not_alternating"] += 1
                    continue
                rec["id"] = s["id"]
            files[part].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[part] += 1
    finally:
        for f in files.values():
            f.close()
    if fmt == "llamafactory":
        info = {"x30_nav_sft": {"file_name": "train.jsonl", "formatting": "sharegpt",
                                "columns": {"messages": "conversations", "system": "system", "images": "images", "tools": "tools"},
                                "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt",
                                         "observation_tag": "observation", "function_tag": "function_call"}},
                "x30_nav_sft_val": {"file_name": "val.jsonl", "formatting": "sharegpt",
                                    "columns": {"messages": "conversations", "system": "system", "images": "images", "tools": "tools"},
                                    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt",
                                             "observation_tag": "observation", "function_tag": "function_call"}}}
        (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    stats = {"format": fmt, "episodes_considered": n_eps, "samples": len(samples), "written": counts, "val_scenes": sorted(val), "leaky_single_scene_split": leaky,
             "rejected": dict(rejected), "scenes": sorted({s["scene"] for s in samples}), "include": include,
             "tool_counts": dict(Counter(s["messages"][-1]["tool_calls"][0]["function"]["name"] for s in samples))}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
