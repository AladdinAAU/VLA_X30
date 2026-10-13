import json

import pytest

from x30_common.jsonrepair import extract_text_tool_calls, loads_lenient, parse_arguments, split_think


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('{"a": 1,}', {"a": 1}),
    ("{'a': 'x', 'b': True, 'c': None}", {"a": "x", "b": True, "c": None}),
    ('{a: 1, b: "two"}', {"a": 1, "b": "two"}),
    ('```json\n{"label": "chair"}\n```', {"label": "chair"}),
    ('Sure! Here is the call: {"object_id": 12} hope it helps', {"object_id": 12}),
    ('{"object_id": 12', {"object_id": 12}),                              # truncated
    ('{"ids": [1, 2, 3', {"ids": [1, 2, 3]}),
    ('{"reason": "found it and', {"reason": "found it and"}),            # truncated mid string
    ('{"x": 1.5e2, "y": -3}', {"x": 150.0, "y": -3}),
    ('{"a": 1 "b": 2}', {"a": 1, "b": 2}),                                # missing comma
    ('{"a": 1} // comment', {"a": 1}),
    ('{"s": "line\\nbreak \\u00e9"}', {"s": "line\nbreak é"}),
])
def test_loads_lenient(text, expected):
    val, _ = loads_lenient(text)
    assert val == expected


def test_valid_json_not_flagged_repaired():
    assert loads_lenient('{"a": 1}') == ({"a": 1}, False)
    assert loads_lenient('{"a": 1,}')[1] is True


@pytest.mark.parametrize("bad", ["", "   ", "no json here at all"])
def test_loads_lenient_rejects_garbage(bad):
    with pytest.raises(ValueError):
        loads_lenient(bad)


def test_deep_nesting_does_not_crash():
    with pytest.raises(ValueError):
        loads_lenient("[" * 500)


def test_parse_arguments_variants():
    assert parse_arguments('{"a": 1}') == ({"a": 1}, False, "")
    assert parse_arguments({"a": 1}) == ({"a": 1}, False, "")                     # object instead of string
    assert parse_arguments(None) == ({}, False, "")
    assert parse_arguments("") == ({}, False, "")
    assert parse_arguments(json.dumps(json.dumps({"a": 1})))[0] == {"a": 1}       # double encoded
    args, rep, err = parse_arguments("{'a': 1,}")
    assert args == {"a": 1} and rep and not err
    args, rep, err = parse_arguments("[1, 2]")
    assert args == {} and err
    args, rep, err = parse_arguments("###")
    assert args == {} and err


def test_split_think():
    assert split_think("<think>hmm</think>answer") == ("answer", "hmm")
    assert split_think("no tags") == ("no tags", "")
    assert split_think("<think>cut off") == ("", "cut off")
    assert split_think("reasoning first</think>the answer") == ("the answer", "reasoning first")


def test_text_tool_call_json_and_xml():
    calls, rest = extract_text_tool_calls('Let me look.\n<tool_call>\n{"name": "search", "arguments": {"label": "chair"}}\n</tool_call>')
    assert calls[0]["name"] == "search" and calls[0]["arguments"] == {"label": "chair"}
    assert rest == "Let me look."
    xml = "<tool_call>\n<function=approach>\n<parameter=object_id>\n12\n</parameter>\n<parameter=note>\nhi\n</parameter>\n</function>\n</tool_call>"
    calls, _ = extract_text_tool_calls(xml)
    assert calls[0]["name"] == "approach" and calls[0]["arguments"] == {"object_id": 12, "note": "hi"}
    # unterminated tag, string-valued arguments
    calls, _ = extract_text_tool_calls('<tool_call>{"name": "scan", "arguments": "{\\"yaw_step_deg\\": 90}"}')
    assert calls[0]["arguments"] == {"yaw_step_deg": 90}
    assert extract_text_tool_calls("nothing here") == ([], "nothing here")
