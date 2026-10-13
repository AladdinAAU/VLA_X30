from x30_common.toolschema import closest_tool_name, coerce_arguments, describe_tools_compact, is_fatal, make_tool

TOOL = make_tool("approach", "Walk to an object.", {
    "object_id": {"type": "integer"}, "standoff_m": {"type": "number", "minimum": 0.5, "maximum": 3},
    "mode": {"type": "string", "enum": ["close", "far"]}, "ids": {"type": "array", "items": {"type": "integer"}},
    "fast": {"type": "boolean"}}, ["object_id"])
SCHEMA = TOOL["function"]["parameters"]


def test_coerce_basic_and_hash_ids():
    out, prob = coerce_arguments(SCHEMA, {"object_id": "#12", "standoff_m": "1.5 m", "fast": "true"})
    assert out == {"object_id": 12, "standoff_m": 1.5, "fast": True} and prob == []


def test_float_integer_and_bounds():
    out, prob = coerce_arguments(SCHEMA, {"object_id": 3.0, "standoff_m": 9})
    assert out["object_id"] == 3 and any("above the maximum" in p for p in prob)


def test_enum_case_and_typo():
    assert coerce_arguments(SCHEMA, {"object_id": 1, "mode": "CLOSE"})[0]["mode"] == "close"
    assert coerce_arguments(SCHEMA, {"object_id": 1, "mode": "far "})[0]["mode"] == "far"
    out, prob = coerce_arguments(SCHEMA, {"object_id": 1, "mode": "sideways"})
    assert "mode" not in out and any("not one of" in p for p in prob)


def test_missing_required_and_unknown():
    out, prob = coerce_arguments(SCHEMA, {"foo": 1})
    assert any("missing required" in p for p in prob) and any(p.startswith("unknown") for p in prob)
    assert [p for p in prob if is_fatal(p)] == ["missing required argument 'object_id'"]


def test_array_from_scalar_and_string():
    assert coerce_arguments(SCHEMA, {"object_id": 1, "ids": 4})[0]["ids"] == [4]
    assert coerce_arguments(SCHEMA, {"object_id": 1, "ids": "[1, 2]"})[0]["ids"] == [1, 2]
    assert coerce_arguments(SCHEMA, {"object_id": 1, "ids": "1, 2 3"})[0]["ids"] == [1, 2, 3]


def test_bad_types():
    out, prob = coerce_arguments(SCHEMA, {"object_id": "abc"})
    assert "object_id" not in out and prob
    out, prob = coerce_arguments(SCHEMA, {"object_id": True})
    assert prob


def test_key_typo_mapped():
    out, _ = coerce_arguments(SCHEMA, {"object_idd": 5})
    assert out == {"object_id": 5}


def test_closest_tool_name():
    tools = [TOOL, make_tool("finish", "x")]
    assert closest_tool_name(tools, "Approach") == "approach"
    assert closest_tool_name(tools, "aproach") == "approach"
    assert closest_tool_name(tools, "teleport") is None


def test_describe():
    assert "approach(object_id: integer" in describe_tools_compact([TOOL])
