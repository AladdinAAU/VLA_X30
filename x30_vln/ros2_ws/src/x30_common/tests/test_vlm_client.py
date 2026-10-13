import asyncio
import json
import threading
import time

import numpy as np
import pytest

from x30_common import VLMCancelled, VLMClient, VLMContextOverflow, VLMHTTPError, VLMTimeout
from x30_common.config import load_models_config
from x30_common.images import user_message
from x30_common.testing.fake_openai import (FakeOpenAIServer, Reply, context_overflow_reply, error_reply, multi_tool_call_reply,
                                            reasoning_reply, text_reply, text_tool_call_reply, tool_call_reply)

TOOLS = [{"type": "function", "function": {"name": "search", "description": "d",
                                           "parameters": {"type": "object", "properties": {"label": {"type": "string"}}}}}]
MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture()
def srv():
    with FakeOpenAIServer() as s:
        yield s


def client(srv, **kw):
    kw.setdefault("backoff_s", 0.01)
    kw.setdefault("timeout_s", 5)
    return VLMClient(srv.base_url, "fake", **kw)


def test_seed_is_sent_only_when_set(srv):
    srv.queue(text_reply("a"), text_reply("b"))
    c = client(srv)
    c.chat(MSG)
    c.seed = 7
    c.chat(MSG)
    assert "seed" not in srv.requests[0] and srv.requests[1]["seed"] == 7


def test_plain_reply_and_request_shape(srv):
    srv.queue(text_reply("hello"))
    res = client(srv).chat(MSG, tools=TOOLS)
    assert res.content == "hello" and res.tool_calls == [] and res.prompt_tokens == 100 and res.completion_tokens == 20
    req = srv.requests[0]
    assert req["model"] == "fake" and req["tools"] == TOOLS and req["tool_choice"] == "auto" and req["stream"] is False
    assert res.timings["predicted_ms"] == 80.0


def test_tool_call_arguments_as_string_and_object(srv):
    srv.queue(tool_call_reply("search", {"label": "chair"}), tool_call_reply("search", {"label": "chair"}, args_as="object"))
    c = client(srv)
    a, b = c.chat(MSG, TOOLS), c.chat(MSG, TOOLS)
    assert a.tool_calls[0].arguments == {"label": "chair"} and "arguments_object" not in a.notes
    assert b.tool_calls[0].arguments == {"label": "chair"} and "arguments_object" in b.notes


def test_malformed_and_truncated_arguments(srv):
    srv.queue(tool_call_reply("search", "{'label': 'chair',}"), tool_call_reply("search", '{"label": "chai'),
              tool_call_reply("search", "garbage"))
    c = client(srv)
    r1, r2, r3 = c.chat(MSG, TOOLS), c.chat(MSG, TOOLS), c.chat(MSG, TOOLS)
    assert r1.tool_calls[0].arguments == {"label": "chair"} and r1.tool_calls[0].repaired
    assert r2.tool_calls[0].arguments == {"label": "chai"} and r2.tool_calls[0].repaired
    assert r3.tool_calls[0].error and r3.tool_calls[0].arguments == {}


def test_multiple_calls_and_mixed_content(srv):
    srv.queue(multi_tool_call_reply([("scan", {}), ("search", {"label": "door"})], content="I will scan first."))
    res = client(srv).chat(MSG, TOOLS)
    assert [t.name for t in res.tool_calls] == ["scan", "search"] and res.content == "I will scan first."
    assert "multiple_tool_calls" in res.notes and "content_with_tool_call" in res.notes
    assert res.tool_calls[0].id != res.tool_calls[1].id


def test_missing_tool_call_id_gets_one(srv):
    srv.queue(tool_call_reply("search", {}, call_id=None))
    res = client(srv).chat(MSG, TOOLS)
    assert res.tool_calls[0].id.startswith("call_") and "tool_call_without_id" in res.notes


def test_text_tool_calls_recovered(srv):
    srv.queue(text_tool_call_reply("search", {"label": "sofa"}), text_tool_call_reply("search", {"label": "tv"}, style="xml"))
    c = client(srv)
    a, b = c.chat(MSG, TOOLS), c.chat(MSG, TOOLS)
    assert a.tool_calls[0].arguments == {"label": "sofa"} and a.tool_calls[0].source == "text" and a.content == ""
    assert b.tool_calls[0].arguments == {"label": "tv"}


def test_reasoning_text_separated(srv):
    srv.queue(reasoning_reply("done", "step by step", tool=("search", {"label": "x"})), text_reply("<think>why</think>ok"))
    c = client(srv)
    a, b = c.chat(MSG, TOOLS), c.chat(MSG, TOOLS)
    assert a.reasoning == "step by step" and a.tool_calls[0].name == "search"
    assert b.content == "ok" and b.reasoning == "why"


def test_assistant_message_roundtrip(srv):
    srv.queue(tool_call_reply("search", {"label": "chair"}, content="thinking"))
    res = client(srv).chat(MSG, TOOLS)
    m = res.assistant_message()
    assert m["role"] == "assistant" and m["content"] == "thinking"
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"label": "chair"}


def test_image_parts_sent(srv):
    srv.queue(text_reply("ok"))
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    c = client(srv)
    c.chat([user_message("what?", [img])])
    part = srv.requests[0]["messages"][0]["content"][0]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert c.stats.summary()["images"] == 1


def test_retry_on_5xx_and_connection_drop(srv):
    srv.queue(error_reply(503, "loading model"), Reply(drop=True), text_reply("finally"))
    c = client(srv, max_retries=3)
    res = c.chat(MSG)
    assert res.content == "finally" and res.attempts == 3
    s = c.stats.summary()
    assert s["calls"] == 1 and s["retries"] == 2


def test_no_retry_on_400_and_gives_up_after_max(srv):
    srv.queue(error_reply(400, "bad request"))
    c = client(srv)
    with pytest.raises(VLMHTTPError) as ei:
        c.chat(MSG)
    assert ei.value.status == 400 and len(srv.requests) == 1
    srv.queue(*[error_reply(500)] * 5)
    c2 = client(srv, max_retries=2)
    with pytest.raises(VLMHTTPError):
        c2.chat(MSG)
    assert len(srv.requests) == 1 + 3
    assert c2.stats.summary()["errors"] == 1


def test_context_overflow_is_typed(srv):
    srv.queue(context_overflow_reply(9000, 8192))
    with pytest.raises(VLMContextOverflow) as ei:
        client(srv).chat(MSG)
    assert ei.value.n_prompt == 9000 and ei.value.n_ctx == 8192 and not ei.value.retryable


def test_timeout(srv):
    srv.queue(text_reply("late", delay_s=1.5))
    c = client(srv, timeout_s=0.4, max_retries=0)
    t0 = time.time()
    with pytest.raises(VLMTimeout):
        c.chat(MSG)
    assert time.time() - t0 < 1.2


def test_cancel_in_flight(srv):
    srv.queue(text_reply("late", delay_s=3.0))
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t0 = time.time()
    with pytest.raises(VLMCancelled):
        client(srv, max_retries=3).chat(MSG, cancel=ev)
    assert time.time() - t0 < 1.5


def test_cancel_before_send_and_during_backoff(srv):
    ev = threading.Event()
    ev.set()
    with pytest.raises(VLMCancelled):
        client(srv).chat(MSG, cancel=ev)
    assert srv.requests == []
    srv.queue(error_reply(503))
    ev2 = threading.Event()
    threading.Timer(0.2, ev2.set).start()
    with pytest.raises(VLMCancelled):
        client(srv, backoff_s=5.0, max_retries=3).chat(MSG, cancel=ev2)


def test_connection_refused_is_retryable_error():
    c = VLMClient("http://127.0.0.1:9/v1", "m", max_retries=1, backoff_s=0.01, timeout_s=1)
    with pytest.raises(Exception) as ei:
        c.chat(MSG)
    assert getattr(ei.value, "retryable", False)
    assert not c.health(timeout_s=0.5)


def test_health_and_props(srv):
    c = client(srv)
    assert c.health() and c.server_props()["default_generation_settings"]["n_ctx"] == 8192


def test_async(srv):
    srv.queue(text_reply("a"), text_reply("b"))
    c = client(srv)

    async def go():
        return await asyncio.gather(c.achat(MSG), c.achat(MSG))

    res = asyncio.run(go())
    assert sorted(r.content for r in res) == ["a", "b"]


def test_stats_accumulate(srv):
    srv.queue(text_reply("a"), text_reply("b"))
    c = client(srv)
    c.chat(MSG)
    c.chat(MSG)
    s = c.stats.summary()
    assert s["calls"] == 2 and s["prompt_tokens"] == 200 and s["completion_tokens"] == 40 and s["latency_p50_s"] >= 0


def test_bad_json_body(srv):
    srv.queue(Reply(raw_text="<html>oops</html>"), text_reply("ok"))
    assert client(srv).chat(MSG).content == "ok"


def test_from_config(tmp_path, srv):
    cfg = load_models_config(tmp_path / "none.yaml", env={"X30_REASONER_URL": srv.base_url, "X30_REASONER_TIMEOUT_S": "7"})
    c = VLMClient.from_config("reasoner", cfg)
    assert c.base_url == srv.base_url and c.timeout_s == 7.0 and c.role == "reasoner"
