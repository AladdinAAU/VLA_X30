"""Next-step hints appended to every tool result, and compact rendering of skill results for the model.

A small model needs the usual next step spelled out ("Found #3. NEXT: goto_object(3), then verify"). Texts live in
configs/agent/*.yaml under `hints`; this module only selects which one applies and fills in the fields."""
from __future__ import annotations

from typing import Any

from .types import SkillResult


class _Safe(dict):
    def __missing__(self, key: str) -> str:
        return "?"


def _fmt(template: str, **fields: Any) -> str:
    try:
        return template.format_map(_Safe(**fields))
    except (ValueError, IndexError, KeyError):
        return template


def _r(v: Any, nd: int = 1) -> Any:
    return round(v, nd) if isinstance(v, float) else v


def compact_data(data: dict, max_items: int = 5, max_chars: int = 400) -> str:
    """key=value pairs; lists of dicts become 'a, b, c' with their main fields; floats rounded."""
    parts = []
    for k, v in (data or {}).items():
        if isinstance(v, dict):
            v = " ".join(f"{a}={_r(b)}" for a, b in list(v.items())[:6])
        elif isinstance(v, list):
            shown = []
            for item in v[:max_items]:
                if isinstance(item, dict):
                    ident = f"#{item['id']}" if "id" in item else ""
                    rest = " ".join(f"{a}={_r(b)}" for a, b in item.items() if a not in ("id", "caption") and not isinstance(b, (dict, list)))
                    shown.append(f"{ident} {rest}".strip())
                else:
                    shown.append(str(_r(item)))
            v = "[" + "; ".join(shown) + (f"; +{len(v) - max_items} more" if len(v) > max_items else "") + "]"
        else:
            v = _r(v)
        parts.append(f"{k}={v}")
    s = " ".join(parts)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def render_result(res: SkillResult, hint: str, max_chars: int = 700) -> str:
    head = "success" if res.success else "failure"
    lines = [f"{head}. {res.message}".strip()]
    if res.error:
        lines.append(f"error: {res.error}")
    d = compact_data(res.data)
    if d:
        lines.append(f"data: {d}")
    if hint:
        lines.append(hint)
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars - 1] + "…"


class HintEngine:
    def __init__(self, hints: dict):
        self.h = hints

    def text(self, key: str, **fields: Any) -> str:
        """Hint template by dotted key ('verify.yes'); '' if missing."""
        node: Any = self.h
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return ""
            node = node[part]
        return _fmt(node, **fields) if isinstance(node, str) else ""

    def for_result(self, name: str, args: dict, res: SkillResult, rejected: set, blocked_before: bool = False, others: list | None = None) -> str:
        d = res.data or {}
        ok = res.success
        oid = d.get("object_id", args.get("object_id"))
        if name == "search":
            cands = [c for c in (d.get("candidates") or []) if isinstance(c, dict) and c.get("id") not in rejected]
            if ok and cands:
                shown = ", ".join(f"#{c['id']} {c.get('label', '')} {c['distance_m']:.1f} m" if isinstance(c.get("distance_m"), (int, float)) else f"#{c['id']} {c.get('label', '')}"
                                  for c in cands[:3])
                return self.text("search.found", n=len(cands), cands=shown, best_id=cands[0]["id"])
            if ok and d.get("candidates"):
                return self.text("search.all_rejected")
            return self.text("search.exhausted") if d.get("exhausted") else self.text("search.fail")
        if name in ("goto_object", "approach"):
            if ok:
                return self.text(f"{name}.ok", object_id=oid)
            if name == "goto_object" and blocked_before:
                return self.text("goto_object.blocked_repeat", object_id=oid)
            return self.text(f"{name}.fail", object_id=oid)
        if name == "verify":
            ans = str(d.get("answer", "")).lower()
            if ans == "no" and others:
                return self.text("verify.no_others", object_id=oid, others=", ".join(f"#{i}" for i in others[:4]), first_other=others[0])
            if ans in ("yes", "no", "unsure"):
                return self.text(f"verify.{ans}", object_id=oid if oid is not None else "?")
            return self.text("verify.unknown")
        if name == "explore":
            found = d.get("found")
            if isinstance(found, dict) and found.get("id") is not None:
                return self.text("explore.found", found_id=found["id"])
            if d.get("exhausted"):
                return self.text("explore.exhausted")
            return self.text("explore.ok" if ok else "explore.fail")
        if name == "scan":
            new = d.get("new_objects") or []
            if ok and new:
                return self.text("scan.ok", new=", ".join(f"#{o.get('id')} {o.get('label')}" for o in new[:5] if isinstance(o, dict)))
            return self.text("scan.empty") if ok else self.text("generic.fail")
        return self.text("generic.ok" if ok else "generic.fail")
