import json
import math
from pathlib import Path

import pytest

from x30_modes.registry import SkillRegistry
from x30_modes.semantic import ObjectRecord, label_matches, parse_query_response, select_candidates

REGISTRY = Path(__file__).resolve().parents[4] / "configs" / "skills.yaml"
REQUIRED = {"goto_pose", "goto_object", "approach", "explore", "search", "turn_to", "move_relative", "scan", "verify", "stop"}


@pytest.fixture(scope="module")
def reg():
    return SkillRegistry.load(REGISTRY)


def test_required_skills_present(reg):
    assert REQUIRED <= set(reg.names())
    for n in REQUIRED:
        s = reg.skills[n]
        assert s.description and s.schema.get("type") == "object"


def test_valid_and_defaults(reg):
    args, msg = reg.parse_args("goto_pose", '{"x": 1, "y": 2}')
    assert msg == "" and args["tolerance_m"] == 0.5
    args, msg = reg.parse_args("move_relative", '{"distance_cm": 100}')
    assert msg == "" and args["direction"] == "forward"
    args, msg = reg.parse_args("stop", "")
    assert msg == "" and args == {}


@pytest.mark.parametrize("name,payload,needle", [
    ("goto_pose", '{"x": 1}', "missing required argument 'y'"),
    ("goto_pose", '{"x": "a", "y": 1}', "expected number"),
    ("goto_pose", '{"x": 1, "y": 1, "speed": 3}', "unknown argument 'speed'"),
    ("move_relative", '{"distance_cm": 5000}', "above the maximum"),
    ("move_relative", '{"distance_cm": 50, "direction": "up"}', "not one of"),
    ("goto_object", '{}', "give at least one of"),
    ("goto_object", '{"object_id": 3, "rejected_ids": ["a"]}', "expected integer"),
    ("turn_to", '{}', "give at least one of"),
    ("search", 'not json', "not valid JSON"),
    ("search", '[1]', "expected a JSON object"),
])
def test_invalid_args_have_helpful_messages(reg, name, payload, needle):
    args, msg = reg.parse_args(name, payload)
    assert args is None and needle in msg and "Example" in msg or "not valid JSON" in msg or "expected a JSON object" in msg


def test_anyof_forms(reg):
    assert reg.parse_args("turn_to", '{"bearing": "left"}')[1] == ""
    assert reg.parse_args("turn_to", '{"bearing": 45}')[1] == ""
    assert reg.parse_args("turn_to", '{"bearing": [1]}')[0] is None
    assert reg.parse_args("explore", '{"strategy": "bias", "bias": [1.0, 2.0]}')[1] == ""


def test_label_matching():
    assert label_matches("chair", "Chair")
    assert label_matches("fire extinguisher", "extinguisher")
    assert label_matches("chairs", "chair")
    assert label_matches("trash can", "trash_can")
    assert not label_matches("chair", "table")


def objs():
    return [
        ObjectRecord(1, "chair", 0.9, 1.0, 0.0),
        ObjectRecord(2, "chair", 0.5, 5.0, 0.0),
        ObjectRecord(3, "chair", 0.2, 0.5, 0.0),
        ObjectRecord(4, "chair", 0.9, 0.2, 0.0, bearing_only=True),
        ObjectRecord(5, "table", 0.9, 0.3, 0.0),
    ]


def test_select_candidates():
    c = select_candidates(objs(), "chair", min_confidence=0.3, robot_xy=(0, 0))
    assert [o.id for o in c] == [1, 2]
    c = select_candidates(objs(), "chair", exclude_ids=[1], min_confidence=0.3, robot_xy=(0, 0))
    assert [o.id for o in c] == [2]
    c = select_candidates(objs(), "chair", include_bearing_only=True, robot_xy=(0, 0))
    assert 4 in [o.id for o in c]


def test_parse_query_response_forms():
    a = {"objects": [{"id": 3, "label": "door", "confidence": 0.8, "position": [1, 2, 0.5], "size": [0.9, 0.1, 2.0]}]}
    r = parse_query_response(json.dumps(a))[0]
    assert (r.id, r.x, r.y, r.sx) == (3, 1.0, 2.0, 0.9)
    b = [{"id": 4, "label": "x", "position": {"x": 3, "y": 4, "z": 0}}]
    assert parse_query_response(json.dumps(b))[0].y == 4.0
    assert parse_query_response("{}") == []


def test_record_to_dict_distance_bearing():
    d = ObjectRecord(1, "chair", 0.9, 0.0, 2.0).to_dict((0.0, 0.0), 0.0)
    assert d["distance_m"] == 2.0 and d["bearing_deg"] == pytest.approx(90.0)


def test_cloud_xyz_drops_zero_intensity_points():
    import struct
    from types import SimpleNamespace as NS
    from x30_modes.cloud import cloud_xyz
    pts = [(1.0, 2.0, 3.0, 5.0), (0.0, 0.0, 0.0, 0.0), (4.0, 5.0, 6.0, 0.0)]
    data = b"".join(struct.pack("<ffff", *p) for p in pts)
    fields = [NS(name=n, offset=4 * i, datatype=7) for i, n in enumerate(["x", "y", "z", "intensity"])]
    msg = NS(fields=fields, width=3, height=1, point_step=16, data=data)
    assert cloud_xyz(msg).shape == (3, 3)
    out = cloud_xyz(msg, drop_zero_intensity=True)
    assert out.shape == (1, 3) and out[0].tolist() == [1.0, 2.0, 3.0]
