"""Exporters on traces/frames produced by the FAKE environment."""
import json
import math
from pathlib import Path

import pytest

from x30_eval.core.optimum import OptimumProvider
from x30_eval.core.runlog import RunDir
from x30_eval.core.runner import SuiteOptions, run_suite
from x30_eval.export.navila import action_text, export_navila, trajectory_to_primitives
from x30_eval.export.perception import export_perception, iou, project_object_box
from x30_eval.export.sft import export_sft
from x30_eval.testing.fake_env import CAMERA, FakeEnv, default_grid

PLANT = {"type": "object", "label": "plant", "instance": "plant_4", "radius": 1.5}
POSE = {"type": "pose", "x": 9.0, "y": 6.0}


@pytest.fixture()
def run_dir(tmp_path, cfg, mk):
    g = default_grid()
    opt = OptimumProvider(cfg, grids={"fake_a": g, "fake_b": g, "fake_c": g})
    eps = []
    for scene in ("fake_a", "fake_b", "fake_c"):
        for i in range(2):
            e = mk(f"{scene}_ok{i}", PLANT if i == 0 else POSE, start=(1.0, 1.0 + 0.5 * i, 0.0), mode="object_nav" if i == 0 else "goal_pose")
            e.scene = scene
            eps.append(e)
        bad = mk(f"{scene}_bad", PLANT, policy="wrong_object")
        bad.scene = scene
        eps.append(bad)
    held = mk("held", PLANT, split="heldout")
    held.scene = "fake_a"
    eps.append(held)
    rd = RunDir(tmp_path / "run")
    run_suite(FakeEnv(), eps, cfg, rd, SuiteOptions(run_name="run", log=lambda m: None), opt,
              {"run_name": "run", "args": {"split": "all", "repeats": 1}, "config_hash": "x", "prompts_hash": "", "episodes_sha256": "e", "overlay_hash": "o"})
    return rd


def test_sft_hf_format_filters_split_and_dedup(tmp_path, run_dir):
    out = tmp_path / "sft"
    st = export_sft([run_dir.path], out, val_fraction=0.34, seed=1)
    assert st["rejected"]["not_success"] == 3 and st["rejected"]["heldout_never_exported"] == 1
    assert st["written"]["train"] > 0 and st["written"]["val"] > 0 and not st["leaky_single_scene_split"]
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (out / "val.jsonl").read_text().splitlines()]
    assert {s["scene"] for s in train}.isdisjoint({s["scene"] for s in val})                   # split by scene, no leakage
    assert all("held" not in s["episode"] and "bad" not in s["episode"] for s in train + val)
    s = train[0]
    last = s["messages"][-1]
    assert last["role"] == "assistant" and isinstance(last["tool_calls"][0]["function"]["arguments"], dict)
    parts = [p for m in s["messages"] if isinstance(m["content"], list) for p in m["content"]]
    imgs = [p["image"] for p in parts if p["type"] == "image"]
    assert imgs == s["images"] and all((out / i).exists() for i in imgs)
    assert st["tool_counts"]["finish"] >= 1


def test_sft_dedup_removes_identical_states(tmp_path, run_dir):
    import shutil
    twin = tmp_path / "twin"
    shutil.copytree(run_dir.path, twin)                         # the same states seen a second time (e.g. a re-run)
    one = export_sft([run_dir.path], tmp_path / "o1")
    two = export_sft([run_dir.path, twin], tmp_path / "o2")
    assert two["samples"] == one["samples"] and two["rejected"]["duplicate"] >= one["samples"]


def test_sft_single_scene_is_flagged_leaky(tmp_path, cfg, mk):
    opt = OptimumProvider(cfg, grids={"fake_a": default_grid()})
    eps = [mk(f"e{i}", PLANT) for i in range(3)]
    for e in eps:
        e.scene = "fake_a"
    rd = RunDir(tmp_path / "r")
    run_suite(FakeEnv(), eps, cfg, rd, SuiteOptions(run_name="r", log=lambda m: None), opt,
              {"run_name": "r", "args": {}, "config_hash": "x", "prompts_hash": "", "episodes_sha256": "e", "overlay_hash": "o"})
    assert export_sft([rd.path], tmp_path / "o")["leaky_single_scene_split"] is True


def test_sft_llamafactory_format(tmp_path, run_dir):
    out = tmp_path / "lf"
    export_sft([run_dir.path], out, fmt="llamafactory", val_fraction=0.34)
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["x30_nav_sft"]["formatting"] == "sharegpt"
    rec = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    roles = [c["from"] for c in rec["conversations"]]
    assert roles[0] == "human" and roles[-1] == "function_call" and "<image>" in rec["conversations"][0]["value"]
    assert json.loads(rec["conversations"][-1]["value"])["name"]
    assert rec["system"] and len(rec["images"]) == rec["conversations"][0]["value"].count("<image>")


def test_trajectory_to_primitives_merges_turns_and_moves():
    poses = [(0.25 * i, 0.0, 0.0) for i in range(5)]
    yaw = 0.0
    for _ in range(3):
        yaw += math.radians(15)
        poses.append((1.0, 0.0, yaw))
    for k in (1, 2):
        poses.append((1.0 + 0.25 * k * math.cos(yaw), 0.25 * k * math.sin(yaw), yaw))
    acts = [a for _, a in trajectory_to_primitives(poses)]
    assert acts == [{"type": "move_forward", "value": 75}, {"type": "move_forward", "value": 25}, {"type": "turn_left", "value": 45},
                    {"type": "move_forward", "value": 50}, {"type": "stop", "value": 0}]
    assert action_text(acts[2]) == "The next action is turn left 45 degrees."
    right = [(0, 0, 0.0), (0, 0, -math.radians(30)), (0, 0, -math.radians(60))]
    assert [a for _, a in trajectory_to_primitives(right)][:2] == [{"type": "turn_right", "value": 30}, {"type": "turn_right", "value": 30}]


def test_navila_export(tmp_path, run_dir):
    st = export_navila([run_dir.path], tmp_path / "nav", val_fraction=0.34)
    assert st["samples"] > 0 and st["action_counts"]["move_forward"] > 0 and st["action_counts"]["stop"] > 0
    assert st["skipped"]["not_success"] == 3 and st["skipped"]["heldout_never_exported"] == 1
    rows = [json.loads(line) for line in (tmp_path / "nav" / "train.jsonl").read_text().splitlines()]
    r = next(x for x in rows if x["action"]["type"] == "move_forward")
    assert r["answer"].startswith("The next action is move forward") and r["prompt"].count("<image>") == len(r["images"])
    assert all(Path(p).exists() for p in r["images"]) and r["conversations"][1]["from"] == "gpt"


def test_projection_geometry():
    robot = (1.0, 1.0, 0.4, 0.0)                                       # facing +x
    ahead = {"x": 4.0, "y": 1.0, "z": 0.4, "sx": 0.5, "sy": 0.5, "sz": 0.8}
    box = project_object_box(ahead, robot, CAMERA)["box"]
    assert abs((box[0] + box[2]) / 2 - 32.0) < 1.0 and abs((box[1] + box[3]) / 2 - 24.0 - 60 * 0.1 / 2.7) < 6        # centred horizontally
    assert project_object_box({**ahead, "x": -2.0}, robot, CAMERA) is None                                           # behind the robot
    left = project_object_box({**ahead, "y": 2.0}, robot, CAMERA)["box"]
    assert left[0] < box[0]                                                                                           # +y (left) is smaller u
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_perception_export_agreement(tmp_path, run_dir):
    st = export_perception([run_dir.path], tmp_path / "perc")
    assert st["frames"] > 20 and st["frames_with_gt_projection"] == st["frames"] and st["skipped"].get("heldout") == 1
    ag = st["agreement"]
    assert 0.5 < ag["recall"] <= 1.0 and 0.5 < ag["precision"] <= 1.0 and "occlusion" in ag["note"]
    ann = json.loads((tmp_path / "perc" / "annotations.jsonl").read_text().splitlines()[0])
    assert (tmp_path / "perc" / ann["image"]).exists()
    coco = json.loads((tmp_path / "perc" / "pseudo_labels_coco.json").read_text())
    assert coco["annotations"] and coco["categories"]
