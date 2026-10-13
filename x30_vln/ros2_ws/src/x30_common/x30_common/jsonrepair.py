"""Tolerant JSON parsing for small-model output, and extraction of tool calls written as text.

Small local models (and the llama.cpp chat templates around them) produce, now and then:
  * arguments with trailing commas, single quotes, python literals (True/None), unquoted keys;
  * output truncated by max_tokens (unterminated string / object);
  * JSON wrapped in markdown fences or prose;
  * tool calls that the server failed to parse and left in the content as
    `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` or `<function=name><parameter=a>1</parameter></function>`.

`loads_lenient` returns the parsed value and whether repair was needed. It never executes anything.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON|python)?\s*(.*?)```", re.S)
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_FUNCTION_TAG = re.compile(r"<function=([\w\-.]+)>\s*(.*?)\s*(?:</function>|$)", re.S)
_PARAM_TAG = re.compile(r"<parameter=([\w\-.]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_WORD = re.compile(r"[A-Za-z_][\w\-.]*")

_LITERALS = {"true": True, "false": False, "null": None, "none": None, "nan": None}


class _Parser:
    """Lenient recursive-descent JSON parser. Closes unterminated structures at end of input."""

    def __init__(self, text: str):
        self.s = text
        self.i = 0
        self.n = len(text)
        self.depth = 0

    def ws(self) -> None:
        while self.i < self.n:
            c = self.s[self.i]
            if c in " \t\r\n﻿":
                self.i += 1
            elif self.s.startswith("//", self.i):
                j = self.s.find("\n", self.i)
                self.i = self.n if j < 0 else j + 1
            elif self.s.startswith("/*", self.i):
                j = self.s.find("*/", self.i + 2)
                self.i = self.n if j < 0 else j + 2
            else:
                break

    def value(self) -> Any:
        self.ws()
        if self.i >= self.n:
            raise ValueError("unexpected end of input")
        self.depth += 1
        if self.depth > 50:
            raise ValueError("nesting too deep")
        try:
            c = self.s[self.i]
            if c == "{":
                return self.obj()
            if c == "[":
                return self.arr()
            if c in "\"'`“‘":
                return self.string()
            m = _NUMBER.match(self.s, self.i)
            if m and (m.end() >= self.n or not (self.s[m.end()].isalpha() or self.s[m.end()] == "_")):
                self.i = m.end()
                txt = m.group(0)
                return float(txt) if any(ch in txt for ch in ".eE") else int(txt)
            m = _WORD.match(self.s, self.i)
            if m:
                self.i = m.end()
                w = m.group(0)
                return _LITERALS.get(w.lower(), w)
            raise ValueError(f"unexpected character {c!r} at {self.i}")
        finally:
            self.depth -= 1

    def string(self) -> str:
        quote = self.s[self.i]
        closing = {"“": "”", "‘": "’"}.get(quote, quote)
        self.i += 1
        out = []
        while self.i < self.n:
            c = self.s[self.i]
            if c == "\\" and self.i + 1 < self.n:
                e = self.s[self.i + 1]
                if e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", self.s[self.i + 2:self.i + 6]):
                    out.append(chr(int(self.s[self.i + 2:self.i + 6], 16)))
                    self.i += 6
                    continue
                out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", "\\": "\\",
                            '"': '"', "'": "'"}.get(e, e))
                self.i += 2
                continue
            if c == closing:
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1
        return "".join(out)          # unterminated: truncated output

    def key(self) -> str:
        self.ws()
        c = self.s[self.i]
        if c in "\"'`“‘":
            return self.string()
        j = self.i
        while j < self.n and self.s[j] not in ":,}]" and not self.s[j].isspace():
            j += 1
        k = self.s[self.i:j]
        self.i = j
        return k

    def obj(self) -> dict:
        self.i += 1
        out: dict = {}
        while True:
            self.ws()
            while self.i < self.n and self.s[self.i] == ",":
                self.i += 1
                self.ws()
            if self.i >= self.n:
                return out
            if self.s[self.i] == "}":
                self.i += 1
                return out
            k = self.key()
            self.ws()
            if self.i < self.n and self.s[self.i] in ":=":
                self.i += 1
                self.ws()
                if self.i >= self.n or self.s[self.i] in ",}":
                    out[k] = None
                    continue
                out[k] = self.value()
            else:
                out[k] = None

    def arr(self) -> list:
        self.i += 1
        out: list = []
        while True:
            self.ws()
            while self.i < self.n and self.s[self.i] == ",":
                self.i += 1
                self.ws()
            if self.i >= self.n:
                return out
            if self.s[self.i] == "]":
                self.i += 1
                return out
            if self.s[self.i] == "}":          # stray closer: stop this array
                return out
            out.append(self.value())


def strip_fences(text: str) -> str:
    m = _FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def _first_json_start(text: str) -> int:
    idx = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    return min(idx) if idx else -1


def loads_lenient(text: str) -> tuple[Any, bool]:
    """Parse JSON from `text`. Returns (value, repaired). Raises ValueError if nothing JSON-like is found."""
    if text is None:
        raise ValueError("no text")
    t = text.strip()
    if not t:
        raise ValueError("empty text")
    try:
        return json.loads(t), False
    except (json.JSONDecodeError, ValueError):
        pass
    t = strip_fences(t)
    try:
        return json.loads(t), True
    except (json.JSONDecodeError, ValueError):
        pass
    start = _first_json_start(t)
    if start < 0:
        raise ValueError(f"no JSON object in {text[:80]!r}")
    p = _Parser(t[start:])
    val = p.value()
    return val, True


def parse_arguments(raw: Any) -> tuple[dict, bool, str]:
    """Normalise tool-call `arguments` to a dict.

    Accepts a JSON string (OpenAI standard), an object (some servers), None/'' (no arguments), or a
    double-encoded string. Returns (args, repaired, error); error is '' when args are trustworthy.
    """
    if raw is None:
        return {}, False, ""
    if isinstance(raw, dict):
        return raw, False, ""
    if isinstance(raw, (list, tuple)):
        return {}, False, f"arguments is a list, expected an object: {str(raw)[:80]}"
    if not isinstance(raw, str):
        return {}, False, f"arguments has type {type(raw).__name__}"
    if not raw.strip():
        return {}, False, ""
    try:
        val, repaired = loads_lenient(raw)
    except ValueError as ex:
        return {}, False, f"unparseable arguments {raw[:80]!r}: {ex}"
    if isinstance(val, str):                       # double-encoded: "\"{\\\"a\\\":1}\""
        try:
            val, _ = loads_lenient(val)
            repaired = True
        except ValueError:
            return {}, repaired, f"arguments decoded to a bare string {val[:60]!r}"
    if not isinstance(val, dict):
        return {}, repaired, f"arguments decoded to {type(val).__name__}, expected an object"
    return val, repaired, ""


def split_think(content: str) -> tuple[str, str]:
    """Separate <think>...</think> reasoning from the visible answer. An unclosed <think> counts as all reasoning."""
    if not content:
        return "", ""
    thoughts = [m.strip() for m in _THINK.findall(content)]
    visible = _THINK.sub("", content)
    if "<think>" in visible:
        head, _, tail = visible.partition("<think>")
        thoughts.append(tail.strip())
        visible = head
    if "</think>" in visible and not thoughts:       # template pre-opened <think>: everything before the closer is reasoning
        head, _, tail = visible.partition("</think>")
        thoughts.append(head.strip())
        visible = tail
    return visible.strip(), "\n".join(t for t in thoughts if t)


def extract_text_tool_calls(content: str) -> tuple[list[dict], str]:
    """Find tool calls the server left in the text. Returns ([{name, arguments(dict), repaired}], remaining_text)."""
    calls: list[dict] = []
    rest = content or ""
    for m in list(_TOOL_CALL_TAG.finditer(rest)):
        body = m.group(1)
        fm = _FUNCTION_TAG.search(body)
        if fm:
            calls.append(_function_tag_call(fm))
            continue
        try:
            val, repaired = loads_lenient(body)
        except ValueError:
            continue
        for item in (val if isinstance(val, list) else [val]):
            if isinstance(item, dict) and ("name" in item or "function" in item):
                fn = item.get("function") if isinstance(item.get("function"), dict) else item
                args, rep2, _ = parse_arguments(fn.get("arguments", fn.get("parameters")))
                calls.append({"name": str(fn.get("name", "")), "arguments": args, "repaired": repaired or rep2})
    if calls:
        rest = _TOOL_CALL_TAG.sub("", rest)
    else:
        for fm in _FUNCTION_TAG.finditer(rest):
            calls.append(_function_tag_call(fm))
        if calls:
            rest = _FUNCTION_TAG.sub("", rest)
    return calls, rest.strip()


def _function_tag_call(fm: re.Match) -> dict:
    args: dict = {}
    for pm in _PARAM_TAG.finditer(fm.group(2)):
        raw = pm.group(2).strip()
        try:
            args[pm.group(1)] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            args[pm.group(1)] = raw
    return {"name": fm.group(1), "arguments": args, "repaired": True}
