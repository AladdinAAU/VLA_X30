"""A scriptable fake OpenAI-compatible server (chat completions) for tests, including the quirks of llama.cpp --jinja.

    with FakeOpenAIServer() as srv:
        srv.queue(tool_call_reply("search", {"label": "chair"}))
        client = VLMClient(srv.base_url, "fake")
        client.chat(...)

Replies are consumed in order; when the queue is empty `handler(request, index)` is called (if given), else a plain
"no script left" text reply is returned. Every request body is kept in `srv.requests`.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


@dataclass
class Reply:
    body: dict | None = None
    status: int = 200
    raw_text: str | None = None       # sent as-is instead of JSON(body)
    delay_s: float = 0.0
    drop: bool = False                # close the connection without answering


def _usage(prompt: int = 100, completion: int = 20) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _wrap(message: dict, finish: str = "stop", prompt: int = 100, completion: int = 20, extra: dict | None = None) -> dict:
    body = {"id": "chatcmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion", "model": "fake",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": _usage(prompt, completion), "timings": {"prompt_ms": 12.0, "predicted_ms": 80.0}}
    if extra:
        body.update(extra)
    return body


def text_reply(content: str, finish: str = "stop", **kw) -> Reply:
    return Reply(_wrap({"role": "assistant", "content": content}, finish), **kw)


def tool_call_reply(name: str, args: dict | str | None = None, content: str | None = None, args_as: str = "string",
                    call_id: str | None = "call_1", **kw) -> Reply:
    """One tool call. args_as: 'string' (OpenAI), 'object' (some servers), or pass `args` as an already-broken string."""
    a: Any = args if args is not None else {}
    if isinstance(a, dict) and args_as == "string":
        a = json.dumps(a)
    call: dict = {"type": "function", "function": {"name": name, "arguments": a}}
    if call_id:
        call["id"] = call_id
    return Reply(_wrap({"role": "assistant", "content": content, "tool_calls": [call]}, "tool_calls"), **kw)


def multi_tool_call_reply(calls: list[tuple[str, dict]], content: str | None = None, **kw) -> Reply:
    tcs = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return Reply(_wrap({"role": "assistant", "content": content, "tool_calls": tcs}, "tool_calls"), **kw)


def text_tool_call_reply(name: str, args: dict, style: str = "json") -> Reply:
    """A tool call left in the content by a template/parser mismatch."""
    if style == "json":
        txt = f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'
    else:
        params = "".join(f"<parameter={k}>\n{json.dumps(v) if not isinstance(v, str) else v}\n</parameter>\n" for k, v in args.items())
        txt = f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"
    return text_reply(txt)


def reasoning_reply(content: str, reasoning: str, field_name: str = "reasoning_content", tool: tuple[str, dict] | None = None) -> Reply:
    msg: dict = {"role": "assistant", "content": content, field_name: reasoning}
    if tool:
        msg["tool_calls"] = [{"id": "call_r", "type": "function", "function": {"name": tool[0], "arguments": json.dumps(tool[1])}}]
    return Reply(_wrap(msg, "tool_calls" if tool else "stop"))


def error_reply(status: int, message: str = "error", **kw) -> Reply:
    return Reply({"error": {"code": status, "message": message, "type": "server_error"}}, status=status, **kw)


def context_overflow_reply(n_prompt: int = 9000, n_ctx: int = 8192) -> Reply:
    return Reply({"error": {"code": 400, "type": "exceed_context_size_error",
                            "message": "request (%d tokens) exceeds the available context size (%d tokens)" % (n_prompt, n_ctx),
                            "n_prompt_tokens": n_prompt, "n_ctx": n_ctx}}, status=400)


class FakeOpenAIServer:
    def __init__(self, handler: Callable[[dict, int], Reply | None] | None = None, port: int = 0, context_tokens: int | None = None):
        self.handler = handler
        self.replies: list[Reply] = []
        self.requests: list[dict] = []
        self.request_times: list[float] = []
        self.paths: list[str] = []
        self.context_tokens = context_tokens
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):          # silence
                pass

            def _send(self, status: int, text: str) -> None:
                data = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/health"):
                    self._send(200, '{"status":"ok"}')
                elif self.path.startswith("/props"):
                    self._send(200, json.dumps({"default_generation_settings": {"n_ctx": outer.context_tokens or 8192}}))
                elif self.path.endswith("/models"):
                    self._send(200, '{"data":[{"id":"fake"}]}')
                else:
                    self._send(404, "{}")

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                with outer._lock:
                    idx = len(outer.requests)
                    outer.requests.append(req)
                    outer.request_times.append(time.time())
                    outer.paths.append(self.path)
                    reply = outer.replies.pop(0) if outer.replies else None
                if reply is None and outer.handler is not None:
                    reply = outer.handler(req, idx)
                if reply is None:
                    reply = text_reply("no script left")
                if outer.context_tokens and reply.status == 200:
                    est = sum(len(json.dumps(m.get("content"))) for m in req.get("messages", [])) / 3.5
                    est += 300 * json.dumps(req.get("messages", [])).count('"image_url"')
                    est += len(json.dumps(req.get("tools", []))) / 3.5
                    if est > outer.context_tokens:
                        reply = context_overflow_reply(int(est), outer.context_tokens)
                if reply.delay_s:
                    time.sleep(reply.delay_s)
                if reply.drop:
                    self.connection.close()
                    return
                self._send(reply.status, reply.raw_text if reply.raw_text is not None else json.dumps(reply.body))

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def queue(self, *replies: Reply) -> None:
        with self._lock:
            self.replies.extend(replies)

    def start(self) -> "FakeOpenAIServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> "FakeOpenAIServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
