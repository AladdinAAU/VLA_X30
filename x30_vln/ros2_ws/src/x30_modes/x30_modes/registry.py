"""Skill registry loader and a small JSON-schema argument validator (subset: type, enum, minimum, maximum, items, anyOf,
required, additionalProperties, default). No jsonschema dependency so it runs in every container."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import find_configs_dir

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


@dataclass
class SkillSpec:
    name: str
    description: str
    schema: dict
    one_of: list[list[str]] = field(default_factory=list)
    exclusive: bool = True
    default_timeout_s: float = 60.0
    typical_duration_s: tuple[float, float] = (1.0, 60.0)
    category: str = "motion"
    raw: dict = field(default_factory=dict)


def validate_value(value: Any, schema: dict, path: str) -> list[str]:
    errs: list[str] = []
    if "anyOf" in schema:
        if not any(not validate_value(value, s, path) for s in schema["anyOf"]):
            errs.append(f"{path}: value {value!r} does not match any allowed form")
        return errs
    t = schema.get("type")
    if t and not _TYPES[t](value):
        return [f"{path}: expected {t}, got {type(value).__name__} ({value!r})"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if t == "array" and "items" in schema:
        for i, it in enumerate(value):
            errs += validate_value(it, schema["items"], f"{path}[{i}]")
    return errs


def validate_args(spec: SkillSpec, args: dict) -> list[str]:
    """Return a list of human readable problems (empty = valid). Defaults are not applied here."""
    errs: list[str] = []
    props = spec.schema.get("properties", {})
    for k in spec.schema.get("required", []):
        if k not in args:
            errs.append(f"missing required argument '{k}'")
    if spec.schema.get("additionalProperties") is False:
        for k in args:
            if k not in props:
                errs.append(f"unknown argument '{k}' (allowed: {', '.join(props) or 'none'})")
    for k, v in args.items():
        if k in props:
            errs += validate_value(v, props[k], k)
    if spec.one_of and not any(all(a in args for a in group) for group in spec.one_of):
        errs.append("give at least one of: " + " | ".join("+".join(g) for g in spec.one_of))
    return errs


def apply_defaults(spec: SkillSpec, args: dict) -> dict:
    out = dict(args)
    for k, p in spec.schema.get("properties", {}).items():
        if k not in out and "default" in p:
            out[k] = p["default"]
    return out


class SkillRegistry:
    def __init__(self, data: dict):
        self.version = data.get("version", 1)
        self.bearing_words: dict[str, float] = {k: float(v) for k, v in (data.get("bearing_words") or {}).items()}
        self.skills: dict[str, SkillSpec] = {}
        for s in data.get("skills", []):
            td = s.get("typical_duration_s", [1, 60])
            self.skills[s["name"]] = SkillSpec(
                name=s["name"], description=s.get("description", "").strip(), schema=s.get("args_schema", {"type": "object"}),
                one_of=[list(g) for g in s.get("args_require_one_of", [])], exclusive=bool(s.get("exclusive", True)),
                default_timeout_s=float(s.get("default_timeout_s", 60.0)), typical_duration_s=(float(td[0]), float(td[1])),
                category=s.get("category", "motion"), raw=s)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SkillRegistry":
        if path in (None, ""):
            d = find_configs_dir()
            if d is None:
                raise FileNotFoundError("configs/skills.yaml not found; set X30_CONFIG_DIR or pass the skills_registry parameter")
            path = d / "skills.yaml"
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def names(self) -> list[str]:
        return list(self.skills)

    def parse_args(self, name: str, args_json: str) -> tuple[dict | None, str]:
        """Parse and validate. Returns (args with defaults, "") or (None, message for the reasoner)."""
        spec = self.skills[name]
        try:
            args = json.loads(args_json) if args_json and args_json.strip() else {}
        except json.JSONDecodeError as ex:
            return None, f"Invalid arguments for {name}: not valid JSON ({ex.msg}). Send a JSON object, e.g. {self.example(name)}"
        if not isinstance(args, dict):
            return None, f"Invalid arguments for {name}: expected a JSON object, got {type(args).__name__}."
        errs = validate_args(spec, args)
        if errs:
            return None, f"Invalid arguments for {name}: " + "; ".join(errs) + f". Example: {self.example(name)}"
        return apply_defaults(spec, args), ""

    def example(self, name: str) -> str:
        spec = self.skills[name]
        ex: dict[str, Any] = {}
        for k in spec.schema.get("required", []):
            ex[k] = _example_value(spec.schema["properties"][k])
        if not ex and spec.one_of:
            k = spec.one_of[0][0]
            ex[k] = _example_value(spec.schema["properties"][k])
        return json.dumps(ex)


def _example_value(p: dict) -> Any:
    if "default" in p:
        return p["default"]
    if "enum" in p:
        return p["enum"][0]
    return {"string": "chair", "number": 1.0, "integer": 1, "boolean": True, "array": [], "object": {}}.get(p.get("type"), 0)
