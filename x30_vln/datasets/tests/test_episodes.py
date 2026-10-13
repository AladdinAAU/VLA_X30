import copy

import pytest

from x30_datasets import registry
from x30_datasets.episodes import io as eio

VALID = {
    "id": "w_objnav_001", "scene": "w", "mode": "object_nav", "instruction": "Find a chair.",
    "start": {"x": 0.0, "y": 1.0, "yaw": 0.2},
    "goal": {"type": "object", "label": "chair", "instance": "chair_01", "radius": 1.5, "any_instance": False,
             "instances": ["chair_01"], "other_instances": ["chair_02"]},
    "time_limit_s": 300, "split": "dev", "reference_path": [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
}


def test_valid_episode_passes():
    assert eio.schema_errors(VALID) == []


@pytest.mark.parametrize("mutate", [
    lambda e: e.pop("start"),
    lambda e: e.update(mode="teleport"),
    lambda e: e.update(split="test"),
    lambda e: e["start"].pop("yaw"),
    lambda e: e["goal"].update(radius=-1),
    lambda e: e.update(time_limit_s=0),
    lambda e: e.update(extra_field=1),
    lambda e: e.update(reference_path=[[0, 1]]),
])
def test_invalid_episode_fails(mutate):
    e = copy.deepcopy(VALID)
    mutate(e)
    assert eio.schema_errors(e)


def test_semantic_checks():
    objs = {"chair_01": {"label": "chair"}, "chair_02": {"label": "chair"}}
    assert eio.semantic_errors(VALID, objs, {"w"}) == []
    assert eio.semantic_errors(VALID, objs, {"other"})
    e = copy.deepcopy(VALID)
    e["goal"]["instances"] = ["table_01"]
    assert eio.semantic_errors(e, objs, {"w"})


def test_split_reproducible_and_balanced():
    keys = [f"w:chair_{i}" for i in range(2000)]
    a = [eio.assign_split(k, 3) for k in keys]
    assert a == [eio.assign_split(k, 3) for k in keys]
    assert a != [eio.assign_split(k, 4) for k in keys]
    assert abs(a.count("dev") / len(a) - 0.6) < 0.04


def test_write_read_roundtrip(tmp_path):
    eio.write_episodes(tmp_path / "e.yaml", [VALID])
    assert eio.read_episodes(tmp_path / "e.yaml")[0]["goal"]["instances"] == ["chair_01"]


def test_registered_episode_files_validate():
    reg = registry.load()
    scenes = set(reg["worlds"])
    from x30_datasets.paths import datasets_dir
    n = 0
    for sid, rel in reg["episodes"].items():
        for ep in eio.read_episodes(datasets_dir() / rel):
            assert eio.schema_errors(ep) == [], (sid, ep["id"])
            assert ep["scene"] in scenes
            n += 1
    assert n > 0 or not reg["episodes"]
