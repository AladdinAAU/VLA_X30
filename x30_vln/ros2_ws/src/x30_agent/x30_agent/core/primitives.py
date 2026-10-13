"""Language action space of the velocity_follow mode (NaVILA style) and its parser.

`move_forward 50`, `turn_left 30`, `stop`, ... are mapped onto the skills move_relative / turn_to / stop of x30_modes.
The text parser accepts the synonyms seen in MapNav/NaVILA outputs ("move forward 75 cm", "rotate right by 30 degrees")."""
from __future__ import annotations

import re
from dataclasses import dataclass

ACTIONS = ("move_forward", "move_backward", "strafe_left", "strafe_right", "turn_left", "turn_right", "stop")
TURNS = ("turn_left", "turn_right")

_SYN = [
    (r"move\s*(?:forward|ahead|straight)|go\s*(?:forward|ahead|straight)|walk\s*(?:forward|ahead)|proceed|advance", "move_forward"),
    (r"move\s*(?:backward|back)|go\s*back(?:ward)?|step\s*back|reverse", "move_backward"),
    (r"strafe\s*left|(?:move|step|sidestep)\s*(?:to\s*the\s*)?left", "strafe_left"),
    (r"strafe\s*right|(?:move|step|sidestep)\s*(?:to\s*the\s*)?right", "strafe_right"),
    (r"turn\s*(?:to\s*the\s*)?left|rotate\s*(?:to\s*the\s*)?left|veer\s*left|pivot\s*left", "turn_left"),
    (r"turn\s*(?:to\s*the\s*)?right|rotate\s*(?:to\s*the\s*)?right|veer\s*right|pivot\s*right", "turn_right"),
    (r"stop|halt|finish|done|arrived", "stop"),
]
_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(cm|centimet\w*|m\b|meters?|metres?|deg\w*|°)?", re.I)


@dataclass
class Primitive:
    action: str
    amount: float = 0.0          # cm for moves, degrees for turns, 0 for stop


def parse_action_text(text: str) -> Primitive | None:
    """First action found in free text. Distances in metres are converted to cm. None if no action word is present."""
    t = (text or "").lower().replace("_", " ")
    best = None
    for pat, name in _SYN:
        for m in re.finditer(pat, t):
            if best is None or m.start() < best[0]:
                best = (m.start(), m.end(), name)
    if best is None:
        return None
    _, end, name = best
    if name == "stop":
        return Primitive("stop", 0.0)
    nm = _NUM.search(t[end:end + 40])
    if not nm:
        return Primitive(name, 0.0)
    val = float(nm.group(1))
    unit = (nm.group(2) or "").lower()
    if unit.startswith("m") and not unit.startswith("cm") and not unit.startswith("cent"):
        val *= 100.0
    return Primitive(name, abs(val))


def clamp(p: Primitive, cfg) -> tuple[Primitive, str]:
    """Apply the amount limits; returns (primitive, note)."""
    if p.action == "stop":
        return p, ""
    if p.action in TURNS:
        lo, hi, default = cfg.turn_min_deg, cfg.turn_max_deg, cfg.default_turn_deg
    else:
        lo, hi, default = cfg.move_min_cm, cfg.move_max_cm, cfg.default_move_cm
    amt = p.amount if p.amount > 0 else default
    note = "" if p.amount > 0 else f"no amount given, used {default:g}"
    new = min(hi, max(lo, amt))
    if new != amt:
        note = f"amount {amt:g} clamped to {new:g}"
    return Primitive(p.action, float(new)), note


def to_skill(p: Primitive) -> tuple[str, dict]:
    """(skill name, args) for the x30_modes skill server."""
    if p.action == "stop":
        return "stop", {}
    if p.action == "turn_left":
        return "turn_to", {"degrees": p.amount}
    if p.action == "turn_right":
        return "turn_to", {"degrees": -p.amount}
    return "move_relative", {"direction": p.action.split("_", 1)[1], "distance_cm": p.amount}


def describe(p: Primitive) -> str:
    if p.action == "stop":
        return "stop"
    return f"{p.action} {p.amount:g}" + (" deg" if p.action in TURNS else " cm")
