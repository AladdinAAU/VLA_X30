"""Tool definitions in the OpenAI function-calling format, and argument validation/coercion against them.

Small models send `"12"` for an integer, `"#12"` for an object id, `"true"` for a boolean, a scalar where an array is
expected, or add keys that do not exist. `coerce_arguments` repairs what is unambiguous and reports the rest, so the
agent can answer the model with a precise error instead of failing silently.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

_INT = re.compile(r"^\s*[#\s]*(-?\d+)(?:\.0+)?\s*$")
_NUM = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def make_tool(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": list(required or [])}}}


def tool_name(tool: dict) -> str:
    return tool["function"]["name"]


def tool_schema(tool: dict) -> dict:
    return tool["function"].get("parameters") or {"type": "object", "properties": {}}


def find_tool(tools: list[dict], name: str) -> dict | None:
    for t in tools:
        if tool_name(t) == name:
            return t
    return None


def closest_tool_name(tools: list[dict], name: str) -> str | None:
    """Unambiguous near-miss of a tool name (case, dashes, a typo), or None."""
    names = [tool_name(t) for t in tools]
    norm = {n.lower().replace("-", "_"): n for n in names}
    key = (name or "").lower().replace("-", "_").replace(" ", "_")
    if key in norm:
        return norm[key]
    close = difflib.get_close_matches(key, list(norm), n=1, cutoff=0.8)
    return norm[close[0]] if close else None


def _coerce(value: Any, spec: dict, path: str, problems: list[str]) -> Any:
    typ = spec.get("type")
    enum = spec.get("enum")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)
    if typ == "integer":
        if isinstance(value, bool):
            problems.append(f"{path}: expected integer, got boolean")
            return None
        if isinstance(value, int):
            out = value
        elif isinstance(value, float) and value == int(value):
            out = int(value)
        elif isinstance(value, str) and _INT.match(value):
            out = int(_INT.match(value).group(1))
        elif isinstance(value, str) and re.search(r"#\s*(\d+)", value):
            out = int(re.search(r"#\s*(\d+)", value).group(1))
        else:
            problems.append(f"{path}: expected integer, got {value!r}")
            return None
        return _bounds(out, spec, path, problems)
    if typ == "number":
        if isinstance(value, bool):
            problems.append(f"{path}: expected number, got boolean")
            return None
        if isinstance(value, (int, float)):
            out = float(value)
        elif isinstance(value, str) and _NUM.fullmatch(value.strip()):
            out = float(value.strip())
        elif isinstance(value, str) and _NUM.search(value) and len(value) < 24:     # "50 cm"
            out = float(_NUM.search(value).group(0))
        else:
            problems.append(f"{path}: expected number, got {value!r}")
            return None
        return _bounds(out, spec, path, problems)
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"):
            return True
        if isinstance(value, str) and value.strip().lower() in ("false", "no", "0"):
            return False
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        problems.append(f"{path}: expected boolean, got {value!r}")
        return None
    if typ == "string":
        if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float, str)) for v in value):
            value = ", ".join(str(v) for v in value)          # e.g. bias=[2, 3] for a flattened string-or-array argument
        if isinstance(value, (dict, list)):
            problems.append(f"{path}: expected string, got {type(value).__name__}")
            return None
        out = value if isinstance(value, str) else str(value)
        if enum:
            for e in enum:
                if str(e).lower() == out.strip().lower():
                    return e
            close = difflib.get_close_matches(out.strip().lower(), [str(e).lower() for e in enum], n=1, cutoff=0.75)
            if close:
                return next(e for e in enum if str(e).lower() == close[0])
            problems.append(f"{path}: {out!r} is not one of {list(enum)}")
            return None
        return out
    if typ == "array":
        if isinstance(value, str):
            from .jsonrepair import loads_lenient
            try:
                parsed, _ = loads_lenient(value)
                value = parsed if isinstance(parsed, list) else [parsed]
            except ValueError:
                value = [p for p in re.split(r"[,\s;]+", value.strip("[] ")) if p]
        elif not isinstance(value, (list, tuple)):
            value = [value]
        item_spec = spec.get("items") or {}
        return [_coerce(v, item_spec, f"{path}[{i}]", problems) if item_spec else v for i, v in enumerate(value)]
    if typ == "object":
        if isinstance(value, dict):
            return value
        problems.append(f"{path}: expected object, got {type(value).__name__}")
        return None
    if enum and value not in enum:
        problems.append(f"{path}: {value!r} is not one of {list(enum)}")
        return None
    return value


def _bounds(value, spec: dict, path: str, problems: list[str]):
    lo, hi = spec.get("minimum"), spec.get("maximum")
    if lo is not None and value < lo:
        problems.append(f"{path}: {value} is below the minimum {lo}")
    if hi is not None and value > hi:
        problems.append(f"{path}: {value} is above the maximum {hi}")
    return value


def coerce_arguments(schema: dict, args: dict) -> tuple[dict, list[str]]:
    """Return (clean_args, problems). Unknown keys are dropped (and listed); missing required keys are problems."""
    props = (schema or {}).get("properties", {})
    required = (schema or {}).get("required", [])
    problems: list[str] = []
    clean: dict = {}
    for key, val in (args or {}).items():
        if key not in props:
            close = difflib.get_close_matches(key, list(props), n=1, cutoff=0.8)
            if close and close[0] not in args:
                key = close[0]
            else:
                problems.append(f"unknown argument '{key}' (ignored)")
                continue
        if val is None:
            continue
        out = _coerce(val, props[key], key, problems)
        if out is not None or isinstance(val, bool):
            clean[key] = out
    for key in required:
        if key not in clean:
            problems.append(f"missing required argument '{key}'")
    return clean, problems


def is_fatal(problem: str) -> bool:
    """Problems that mean the call cannot run (as opposed to notes like an ignored unknown key)."""
    return not problem.startswith("unknown argument")


def describe_tools_compact(tools: list[dict]) -> str:
    """One line per tool: name(arg: type, ...) - first sentence. For prompts and docs."""
    lines = []
    for t in tools:
        f = t["function"]
        props = tool_schema(t).get("properties", {})
        req = set(tool_schema(t).get("required", []))
        sig = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in props.items())
        first = f.get("description", "").split(". ")[0].strip()
        lines.append(f"{f['name']}({sig}) - {first}")
    return "\n".join(lines)
