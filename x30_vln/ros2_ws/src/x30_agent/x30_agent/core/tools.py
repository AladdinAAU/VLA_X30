"""Tool list for the reasoner: registry skills plus the two harness tools (query_map, finish)."""
from __future__ import annotations

from dataclasses import dataclass, field

from x30_common.toolschema import make_tool, tool_name

from .skills import SkillSpec, build_tool, full_schema


def query_map_tool() -> dict:
    return make_tool(
        "query_map",
        "Read-only question about the semantic map, no movement. op=list (all instances of label), get (one object by id), "
        "relations (what is near/left/right of object id), describe (rank objects for a text description such as 'the red chair near the window').",
        {"op": {"type": "string", "enum": ["list", "get", "relations", "describe"]},
         "label": {"type": "string", "description": "for op=list"},
         "id": {"type": "integer", "description": "for op=get / relations"},
         "text": {"type": "string", "description": "for op=describe"}},
        ["op"])


def finish_tool() -> dict:
    return make_tool(
        "finish",
        "End the task. success=true only when the task is done and checked with verify; success=false when you give up. "
        "This is the only way to end the task.",
        {"success": {"type": "boolean"}, "reason": {"type": "string", "description": "one short sentence"}},
        ["success", "reason"])


def report_target_tool() -> dict:
    return make_tool(
        "report_target",
        "Report WHERE the target is; a path planner then drives the robot there and your task ends. Give object_id (an id from the map) "
        "or map coordinates x AND y (for a room you MUST pass x and y: its centre from the Rooms list). found=false only if the target is not in the map and "
        "nothing is left to explore.",
        {"object_id": {"type": "integer", "description": "id of the mapped object to go to"},
         "x": {"type": "number", "description": "map x in metres (with y)"}, "y": {"type": "number", "description": "map y in metres (with x)"},
         "found": {"type": "boolean", "description": "false = give up (default true)"},
         "reason": {"type": "string", "description": "one short sentence"}},
        ["reason"])


LOCAL_TOOLS = {"query_map": query_map_tool, "finish": finish_tool, "report_target": report_target_tool}


@dataclass
class ToolSet:
    tools: list[dict]
    schemas: dict[str, dict]
    skills: dict[str, SkillSpec] = field(default_factory=dict)       # only the exposed skills
    local: set[str] = field(default_factory=set)

    @property
    def names(self) -> list[str]:
        return [tool_name(t) for t in self.tools]


def build_toolset(registry: dict[str, SkillSpec], skill_names: list[str], local_names: list[str], description_mode: str = "summary",
                  overrides: dict | None = None) -> ToolSet:
    tools, schemas, skills = [], {}, {}
    for n in skill_names:
        if n not in registry:
            raise KeyError(f"policy lists skill '{n}' which is not in the skill registry (known: {sorted(registry)})")
        spec = registry[n]
        tools.append(build_tool(spec, description_mode, (overrides or {}).get(n, "")))
        schemas[n] = full_schema(spec)
        skills[n] = spec
    for n in local_names:
        t = LOCAL_TOOLS[n]()
        tools.append(t)
        schemas[n] = t["function"]["parameters"]
    return ToolSet(tools, schemas, skills, set(local_names))
