"""Interpretation of what a person says during a guided tour (no ROS imports; the language model is injected as a callable).

`interpret(chat, utterance, context, last_place)` builds one chat request with four tools and returns an `Action`:
  save    name + kind (room = the whole room the person stands in, spot = this spot) + optional owner (a person the room belongs to), function, aliases
  rename  old_name -> new_name (corrections: 'no, that is Marco')
  forget  name
  ignore  the sentence does not teach a place
`chat(messages, tools)` must return an object with `.tool_calls` (list of objects with `.name` and `.arguments` dict), like x30_common.VLMClient.chat.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

SYSTEM = """You are the memory of a mobile robot (a quadruped) that a person is showing around a building, like a new worker's first day. While the person walks, the robot follows. When the person says something about the place they are standing in, you decide what to remember. You act ONLY by calling exactly one tool.

What people say: 'this is Marcus', "this is Marcus's office", 'Alexander', 'this is room 12', 'here we deliver stuff', 'this is where the nurses take their break', 'no, that is Marco', 'forget this place', or small talk that has nothing to do with places.
Rules:
- A person's first name ('this is Marcus') names a room or desk that belongs to that person: name = 'Marcus', owner = 'Marcus'. Keep the person's spelling.
- A bare name or short label with nothing else ('Alexander.', 'Room 12', 'The pharmacy') is the person naming the place they are standing in: save it (a first name alone means a room or desk of that person).
- A function ('here we deliver stuff', 'this is the pharmacy') gives a short noun phrase as the name: 'Delivery', 'Pharmacy', 'Nurses break room'. Put what the person said about it in `function`. Do not invent facts.
- kind: 'room' when the place is the whole room the person stands in (the context says they are inside a room and the person says 'this is X' / 'this room' / 'my office'). 'spot' when it is a single place inside a corridor or hall, a corner, a door, a desk, 'here we ...', or when the context says corridor or hall or no region.
- Corrections ('no, that is Marco', 'it is called X, not Y') -> rename_place; use the last taught place unless another is named.
- 'forget this', 'delete the pharmacy' -> forget_place.
- Anything that does not teach, correct or delete a place -> ignore.
- Names are short (at most four words), written the way a sign would write them. Never put the whole sentence in a name."""

TOOLS = [
    {"type": "function", "function": {
        "name": "save_place", "description": "Remember a place the person just named.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "short name, e.g. 'Marcus', 'Delivery', 'Room 12'"},
            "kind": {"type": "string", "enum": ["room", "spot"], "description": "room = the whole room the person is in, spot = just this place"},
            "owner": {"type": "string", "description": "a person this room or desk belongs to (first name), else empty"},
            "function": {"type": "string", "description": "what is done here, in a few words, else empty"},
            "aliases": {"type": "array", "items": {"type": "string"}, "description": "other names the person used"}},
            "required": ["name", "kind"]}}},
    {"type": "function", "function": {
        "name": "rename_place", "description": "Correct the name of a place that was taught before.",
        "parameters": {"type": "object", "properties": {"old_name": {"type": "string"}, "new_name": {"type": "string"}}, "required": ["old_name", "new_name"]}}},
    {"type": "function", "function": {
        "name": "forget_place", "description": "Delete a taught place.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "ignore", "description": "The sentence does not teach, correct or delete a place.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]


@dataclass
class Action:
    kind: str                                   # save | rename | forget | ignore | error
    name: str = ""
    place_kind: str = "spot"
    owner: str = ""
    function: str = ""
    aliases: list = field(default_factory=list)
    old_name: str = ""
    new_name: str = ""
    reason: str = ""


def region_phrase(ctx: dict) -> str:
    r = ctx.get("region")
    if not r:
        return "the person stands in unmapped space (no region)"
    area = float(r.get("area_m2", 0.0))
    if r.get("name") == "corridor" or (float(r.get("width_m", 9)) <= 2.2 and float(r.get("length_m", 0)) >= 2.5 * max(float(r.get("width_m", 1)), 0.1)):
        kind = "a corridor"
    elif area > 80.0:
        kind = f"a large hall or open area ({area:.0f} m2)"
    else:
        kind = f"inside a room ({area:.0f} m2, {r.get('width_m')} x {r.get('length_m')} m)"
    called = r.get("name") or ""
    return f"the person stands in {kind}" + (f", currently called '{called}'" if called else ", not named yet")


def build_messages(utterance: str, context: dict, last_place: str | None) -> list[dict]:
    objs = context.get("objects_nearby") or {}
    near = context.get("places") or []
    lines = ["Context: " + region_phrase(context) + ".",
             "Objects nearby: " + (", ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(objs.items(), key=lambda kv: -kv[1])[:8]) or "none seen") + ".",
             "Taught places near here: " + (", ".join(f"{p['name']} ({p['kind']}, {p['distance_m']} m)" for p in near[:4]) or "none") + ".",
             "Last taught place: " + (last_place or "none") + ".",
             f"The person said: {json.dumps(utterance, ensure_ascii=False)}"]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def _clean(s) -> str:
    s = " ".join(str(s or "").replace('"', "").split()).strip(" .,:;!?")
    return s[:40]


def parse(res) -> Action:
    calls = getattr(res, "tool_calls", None) or []
    if not calls:
        return Action("error", reason="the model made no tool call")
    c = calls[0]
    a = getattr(c, "arguments", None) or {}
    name = getattr(c, "name", "")
    if name == "save_place":
        nm = _clean(a.get("name"))
        if not nm:
            return Action("error", reason="save_place without a name")
        kind = a.get("kind") if a.get("kind") in ("room", "spot") else "spot"
        al = [_clean(x) for x in (a.get("aliases") or []) if _clean(x)]
        return Action("save", name=nm, place_kind=kind, owner=_clean(a.get("owner")), function=" ".join(str(a.get("function") or "").split())[:80], aliases=al)
    if name == "rename_place":
        o, n = _clean(a.get("old_name")), _clean(a.get("new_name"))
        return Action("rename", old_name=o, new_name=n) if o and n else Action("error", reason="rename_place needs both names")
    if name == "forget_place":
        n = _clean(a.get("name"))
        return Action("forget", name=n) if n else Action("error", reason="forget_place without a name")
    if name == "ignore":
        return Action("ignore", reason=str(a.get("reason") or "")[:120])
    return Action("error", reason=f"unknown tool {name!r}")


def interpret(chat, utterance: str, context: dict, last_place: str | None = None) -> Action:
    res = chat(build_messages(utterance, context, last_place), TOOLS)
    return parse(res)
