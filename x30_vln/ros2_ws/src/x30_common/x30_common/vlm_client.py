"""OpenAI-compatible chat client for llama.cpp / vLLM / hosted endpoints, with tool calling.

Only the standard library is used (http.client), so it runs in any container. A request can be cancelled from another
thread (`cancel` event) and has a hard total deadline; connection errors, timeouts, 429 and 5xx (including
llama.cpp's 503 "loading model") are retried with exponential backoff and jitter.

Normalisation done here, so callers see one shape (`ChatResult`):
  * tool-call `arguments` as JSON string, as object, malformed/truncated (repaired) or missing;
  * several tool calls in one reply, content mixed with calls;
  * tool calls the server left in `content` as <tool_call>...</tool_call> or <function=...> text;
  * `<think>` reasoning, or `reasoning_content`, separated from the visible answer;
  * token usage and llama.cpp `timings`.
"""
from __future__ import annotations

import asyncio
import http.client
import json
import random
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .config import ModelsConfig, VLMConfig, load_models_config
from .images import count_images
from .jsonrepair import extract_text_tool_calls, parse_arguments, split_think

# ------------------------------------------------------------------ errors


class VLMError(Exception):
    """Base class. `retryable` tells whether another attempt could succeed."""
    retryable = False


class VLMConnectionError(VLMError):
    retryable = True


class VLMTimeout(VLMError):
    retryable = True


class VLMHTTPError(VLMError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body
        self.retryable = status in (408, 409, 425, 429) or status >= 500


class VLMContextOverflow(VLMHTTPError):
    """Prompt does not fit the server context (llama.cpp: 400 exceed_context_size_error)."""

    def __init__(self, status: int, body: str, n_prompt: int | None = None, n_ctx: int | None = None):
        super().__init__(status, body)
        self.n_prompt, self.n_ctx = n_prompt, n_ctx
        self.retryable = False


class VLMCancelled(VLMError):
    pass


class VLMBadResponse(VLMError):
    retryable = True


# ------------------------------------------------------------------ results


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    repaired: bool = False
    error: str = ""            # non-empty: arguments could not be parsed (arguments == {})
    source: str = "native"     # native | text (recovered from content)

    def to_message(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": json.dumps(self.arguments, separators=(",", ":"))}}


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    timings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)      # quirks seen: "arguments_object", "text_tool_call", "repaired", ...
    raw: dict = field(default_factory=dict)

    def assistant_message(self) -> dict:
        m: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            m["tool_calls"] = [tc.to_message() for tc in self.tool_calls]
        return m

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CallRecord:
    role: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    n_images: int
    attempts: int
    ok: bool
    error: str = ""
    t: float = field(default_factory=time.time)
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0


class UsageStats:
    """Thread-safe token and latency accounting. `summary()` is JSON-able."""

    def __init__(self, keep: int = 5000):
        self._lock = threading.Lock()
        self.records: list[CallRecord] = []
        self.keep = keep

    def add(self, rec: CallRecord) -> None:
        with self._lock:
            self.records.append(rec)
            if len(self.records) > self.keep:
                del self.records[: len(self.records) - self.keep]

    def reset(self) -> None:
        with self._lock:
            self.records.clear()

    def summary(self, since: float | None = None) -> dict:
        with self._lock:
            recs = [r for r in self.records if since is None or r.t >= since]
        ok = [r for r in recs if r.ok]
        lat = sorted(r.latency_s for r in ok)

        def pct(p: float) -> float:
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 3) if lat else 0.0

        return {
            "calls": len(recs), "errors": len(recs) - len(ok), "retries": sum(max(0, r.attempts - 1) for r in recs),
            "prompt_tokens": sum(r.prompt_tokens for r in ok), "completion_tokens": sum(r.completion_tokens for r in ok),
            "images": sum(r.n_images for r in ok),
            "latency_mean_s": round(sum(lat) / len(lat), 3) if lat else 0.0, "latency_p50_s": pct(0.5),
            "latency_p95_s": pct(0.95), "latency_max_s": round(lat[-1], 3) if lat else 0.0,
            "latency_total_s": round(sum(lat), 3),
        }


# ------------------------------------------------------------------ client


class VLMClient:
    """Sync and async chat-completions client. One instance per endpoint; thread-safe (one connection per call)."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", max_tokens: int = 400, temperature: float = 0.2,
                 timeout_s: float = 120.0, max_retries: int = 3, backoff_s: float = 1.0, backoff_max_s: float = 20.0,
                 extra_body: dict | None = None, stats: UsageStats | None = None, role: str = "vlm",
                 context_tokens: int = 8192, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        self.extra_body = dict(extra_body or {})
        self.seed: int | None = None          # sampling seed sent with every request when set (llama.cpp "seed")
        self.stats = stats or UsageStats()
        self.role = role
        self.context_tokens = context_tokens
        self._sleep = sleep
        self._rng = random.Random(0)

    # --------------------------------------------------------- construction
    @classmethod
    def from_config(cls, role: str = "reasoner", config: ModelsConfig | VLMConfig | None = None, **overrides) -> "VLMClient":
        """Build from configs/models.yaml (+ env overrides). `role` is 'reasoner' or 'mapper'."""
        if config is None:
            config = load_models_config()
        c = config.vlm(role) if isinstance(config, ModelsConfig) else config
        kw = dict(base_url=c.base_url, model=c.model, api_key=c.api_key, max_tokens=c.max_tokens, temperature=c.temperature,
                  timeout_s=c.timeout_s, max_retries=c.max_retries, backoff_s=c.backoff_s, extra_body=c.extra_body,
                  context_tokens=c.context_tokens, role=role)
        kw.update(overrides)
        return cls(**kw)

    # --------------------------------------------------------- public API
    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | dict | None = "auto",
             max_tokens: int | None = None, temperature: float | None = None, cancel: threading.Event | None = None,
             timeout_s: float | None = None, extra: dict | None = None) -> ChatResult:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                   "max_tokens": max_tokens or self.max_tokens,
                                   "temperature": self.temperature if temperature is None else temperature}
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        payload.update(self.extra_body)
        if self.seed is not None:
            payload["seed"] = self.seed
        if extra:
            payload.update(extra)
        n_img = count_images(messages)
        t0 = time.time()
        attempts = 0
        last_err: Exception | None = None
        while True:
            attempts += 1
            try:
                data = self._post("/chat/completions", payload, timeout_s or self.timeout_s, cancel)
                res = parse_chat_response(data)
                res.latency_s = time.time() - t0
                res.attempts = attempts
                self._record(res, n_img, attempts, True)
                return res
            except VLMCancelled:
                self._record(None, n_img, attempts, False, "cancelled", time.time() - t0)
                raise
            except VLMError as ex:
                last_err = ex
                if not ex.retryable or attempts > self.max_retries:
                    self._record(None, n_img, attempts, False, f"{type(ex).__name__}: {ex}", time.time() - t0)
                    raise
                delay = min(self.backoff_max_s, self.backoff_s * (2 ** (attempts - 1))) * (0.75 + 0.5 * self._rng.random())
                if cancel is not None:
                    if cancel.wait(delay):
                        self._record(None, n_img, attempts, False, "cancelled", time.time() - t0)
                        raise VLMCancelled("cancelled during backoff") from last_err
                else:
                    self._sleep(delay)

    async def achat(self, *args, **kwargs) -> ChatResult:
        """Async wrapper: runs `chat` in a worker thread so the event loop never blocks. Pass `cancel=` to abort."""
        return await asyncio.to_thread(self.chat, *args, **kwargs)

    def health(self, timeout_s: float = 3.0) -> bool:
        """True if the server answers GET /health (llama.cpp) or GET /models (generic)."""
        for path, root in (("/health", True), ("/models", False)):
            try:
                self._get(path, timeout_s, root=root)
                return True
            except VLMError:
                continue
        return False

    def server_props(self, timeout_s: float = 3.0) -> dict:
        """llama.cpp GET /props (n_ctx, model path, modalities); {} if unavailable."""
        try:
            return json.loads(self._get("/props", timeout_s, root=True))
        except (VLMError, ValueError):
            return {}

    def config_dict(self) -> dict:
        return {"base_url": self.base_url, "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "timeout_s": self.timeout_s, "max_retries": self.max_retries, "context_tokens": self.context_tokens,
                "extra_body": self.extra_body}

    # --------------------------------------------------------- internals
    def _record(self, res: ChatResult | None, n_img: int, attempts: int, ok: bool, err: str = "", latency: float = 0.0) -> None:
        if res is not None:
            self.stats.add(CallRecord(self.role, res.latency_s, res.prompt_tokens, res.completion_tokens, n_img, attempts, True,
                                      prompt_ms=float(res.timings.get("prompt_ms", 0) or 0),
                                      predicted_ms=float(res.timings.get("predicted_ms", 0) or 0)))
        else:
            self.stats.add(CallRecord(self.role, latency, 0, 0, n_img, attempts, ok, err))

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key and self.api_key != "none":
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _split(self, path: str, root: bool):
        u = urlsplit(self.base_url)
        base_path = "" if root else u.path.rstrip("/")
        return u.scheme, u.hostname or "localhost", u.port, f"{base_path}{path}"

    def _connect(self, scheme: str, host: str, port: int | None, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        return cls(host, port, timeout=timeout)

    def _get(self, path: str, timeout_s: float, root: bool = False) -> str:
        scheme, host, port, full = self._split(path, root)
        conn = self._connect(scheme, host, port, timeout_s)
        try:
            conn.request("GET", full, headers=self._headers())
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", "replace")
        except (OSError, http.client.HTTPException) as ex:
            raise VLMConnectionError(f"GET {full}: {ex}") from ex
        finally:
            conn.close()
        if resp.status >= 400:
            raise VLMHTTPError(resp.status, body)
        return body

    def _post(self, path: str, payload: dict, timeout_s: float, cancel: threading.Event | None) -> dict:
        """POST in a worker thread; the caller polls for cancel/deadline and abandons (and shuts down) the socket if needed."""
        if cancel is not None and cancel.is_set():
            raise VLMCancelled("cancelled before sending")
        scheme, host, port, full = self._split(path, False)
        body = json.dumps(payload).encode("utf-8")
        conn = self._connect(scheme, host, port, min(timeout_s, 30.0))
        done = threading.Event()
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                conn.request("POST", full, body=body, headers=self._headers())
                if conn.sock is not None:
                    conn.sock.settimeout(timeout_s)
                resp = conn.getresponse()
                box["status"], box["raw"] = resp.status, resp.read()
            except socket.timeout as ex:
                box["error"] = VLMTimeout(f"no complete reply within {timeout_s:.0f} s: {ex}")
            except (OSError, http.client.HTTPException, AttributeError) as ex:
                box["error"] = VLMConnectionError(f"{type(ex).__name__}: {ex}")
            finally:
                done.set()
                try:
                    conn.close()
                except OSError:
                    pass

        threading.Thread(target=work, daemon=True, name="vlm-request").start()
        deadline = time.time() + timeout_s
        while not done.wait(0.05):
            reason = ""
            if cancel is not None and cancel.is_set():
                reason = "cancelled"
            elif time.time() > deadline:
                reason = "timeout"
            if reason:
                try:                      # unblocks the worker where the OS allows it; otherwise the daemon thread ends with its own timeout
                    if conn.sock is not None:
                        conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                if reason == "cancelled":
                    raise VLMCancelled("cancelled")
                raise VLMTimeout(f"no complete reply within {timeout_s:.0f} s")
        if "error" in box:
            raise box["error"]
        text = box["raw"].decode("utf-8", "replace")
        if box["status"] >= 400:
            raise _http_error(box["status"], text)
        try:
            return json.loads(text)
        except ValueError as ex:
            raise VLMBadResponse(f"reply is not JSON: {text[:200]!r}") from ex


def _http_error(status: int, text: str) -> VLMHTTPError:
    if status == 400 and ("exceed_context_size" in text or "context size" in text.lower() or "context length" in text.lower()):
        n_prompt = n_ctx = None
        try:
            err = json.loads(text).get("error", {})
            n_prompt, n_ctx = err.get("n_prompt_tokens"), err.get("n_ctx")
        except (ValueError, AttributeError):
            pass
        return VLMContextOverflow(status, text, n_prompt, n_ctx)
    return VLMHTTPError(status, text)


# ------------------------------------------------------------------ response parsing


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def parse_chat_response(data: dict) -> ChatResult:
    """Turn a chat-completions JSON body into a ChatResult, absorbing server/template quirks."""
    if not isinstance(data, dict) or not data.get("choices"):
        raise VLMBadResponse(f"no choices in reply: {str(data)[:200]}")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    res = ChatResult(raw=data, finish_reason=str(choice.get("finish_reason") or ""))
    usage = data.get("usage") or {}
    res.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    res.completion_tokens = int(usage.get("completion_tokens") or 0)
    res.timings = data.get("timings") or {}

    content = _content_text(msg.get("content"))
    reasoning = _content_text(msg.get("reasoning_content") or msg.get("reasoning"))
    content, think = split_think(content)
    res.reasoning = "\n".join(x for x in (reasoning, think) if x)
    if res.reasoning:
        res.notes.append("reasoning_text")

    raw_calls = msg.get("tool_calls") or []
    if not raw_calls and isinstance(msg.get("function_call"), dict):          # legacy single-call format
        raw_calls = [{"type": "function", "function": msg["function_call"]}]
        res.notes.append("legacy_function_call")
    for i, rc in enumerate(raw_calls):
        fn = rc.get("function") if isinstance(rc.get("function"), dict) else rc
        raw_args = fn.get("arguments", fn.get("parameters"))
        args, repaired, err = parse_arguments(raw_args)
        if isinstance(raw_args, dict):
            res.notes.append("arguments_object")
        if repaired:
            res.notes.append("arguments_repaired")
        if err:
            res.notes.append("arguments_unparseable")
        res.tool_calls.append(ToolCall(
            id=str(rc.get("id") or f"call_{uuid.uuid4().hex[:8]}"), name=str(fn.get("name") or ""), arguments=args,
            raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args) if raw_args is not None else "",
            repaired=repaired, error=err))
        if not rc.get("id"):
            res.notes.append("tool_call_without_id")
    if len(res.tool_calls) > 1:
        res.notes.append("multiple_tool_calls")

    if not res.tool_calls and content and ("<tool_call>" in content or "<function=" in content):
        calls, rest = extract_text_tool_calls(content)
        for c in calls:
            res.tool_calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=c["name"], arguments=c["arguments"],
                                           raw_arguments=json.dumps(c["arguments"]), repaired=c["repaired"], source="text"))
        if calls:
            res.notes.append("text_tool_call")
            content = rest
    if res.tool_calls and content:
        res.notes.append("content_with_tool_call")
    res.content = content
    if res.finish_reason == "length":
        res.notes.append("truncated")
    return res
