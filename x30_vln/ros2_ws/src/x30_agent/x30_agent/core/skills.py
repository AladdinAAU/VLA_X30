"""Skill registry (configs/skills.yaml, owned by x30_modes) -> tool-calling schemas for the reasoner.

The registry is the single source of truth for skill names and arguments. This module only reads it: it builds the
OpenAI-style tool list, flattens `anyOf` (not every server's grammar generator handles it), appends the
"give one of" constraints that JSON schema cannot express here, and keeps the full schema for argument validation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from x30_common.toolschema import make_tool


@dataclass
class SkillSpec:
    name: str
    summary: str = ""
    description: str = ""
    schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    require_one_of: list = field(default_factory=list)
    default_timeout_s: float = 60.0
    typical_duration_s: list = field(default_factory=list)
    category: str = "motion"
    exclusive: bool = True
    result_data: dict = field(default_factory=dict)


def flatten_schema(schema: dict) -> dict:
    """Copy of a JSON schema subset without anyOf/oneOf/default/additionalProperties (keeps type, enum, items, bounds)."""
    s = copy.deepcopy(schema)

    def walk(node):
        if not isinstance(node, dict):
            return node
        for key in ("anyOf", "oneOf"):
            if key in node:
                alts = node.pop(key)
                types = [a.get("type") for a in alts if isinstance(a, dict)]
                node["type"] = "string"
                node["description"] = (node.get("description", "") + f" (accepts: {', '.join(str(t) for t in types)}; write arrays as text like \"1.5, 2\")").strip()
        node.pop("default", None)
        node.pop("additionalProperties", None)
        for k, v in list(node.items()):
            if k == "properties" and isinstance(v, dict):
                for pk, pv in v.items():
                    v[pk] = walk(pv)
            elif k == "items":
                node[k] = walk(v)
        return node

    return walk(s)


def load_registry(path: str | Path) -> dict[str, SkillSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, SkillSpec] = {}
    for raw in data.get("skills", []):
        out[raw["name"]] = SkillSpec(
            name=raw["name"], summary=raw.get("summary", ""), description=" ".join(str(raw.get("description", "")).split()),
            schema=raw.get("args_schema") or raw.get("parameters") or {"type": "object", "properties": {}},
            require_one_of=raw.get("args_require_one_of") or [], default_timeout_s=float(raw.get("default_timeout_s", 60)),
            typical_duration_s=raw.get("typical_duration_s") or [], category=raw.get("category", "motion"),
            exclusive=bool(raw.get("exclusive", True)), result_data=(raw.get("result") or {}).get("data") or {})
    return out


def one_of_text(spec: SkillSpec) -> str:
    if not spec.require_one_of:
        return ""
    alts = " or ".join("+".join(group) for group in spec.require_one_of)
    return f" Give at least: {alts}."


def build_tool(spec: SkillSpec, mode: str = "summary", override: str = "") -> dict:
    """mode 'summary': one-line summary (+ override text); 'full': the registry description (long)."""
    if override:
        desc = override.strip()
    elif mode == "full" and spec.description:
        desc = spec.description
    else:
        desc = spec.summary or spec.description.split(". ")[0]
    desc = desc.rstrip(".") + "." + one_of_text(spec)
    schema = flatten_schema(spec.schema)
    # a tool schema without `required` plus an at-least-one constraint is described in the text above
    return make_tool(spec.name, desc, schema.get("properties", {}), schema.get("required", []))


def validate_one_of(spec: SkillSpec, args: dict) -> str:
    """'' if the at-least-one-of rule holds, else a message naming the options."""
    if not spec.require_one_of:
        return ""
    if any(all(k in args for k in group) for group in spec.require_one_of):
        return ""
    return "give at least: " + " or ".join("+".join(g) for g in spec.require_one_of)


def full_schema(spec: SkillSpec) -> dict:
    """Schema used to validate and coerce arguments (anyOf flattened, bounds kept)."""
    return flatten_schema(spec.schema)
