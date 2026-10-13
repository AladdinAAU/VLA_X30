"""Fake OpenAI-compatible VLM server for the verify skill: answers yes when the question mentions the configured word."""
import json
import sys
import time

from x30_common.testing.fake_openai import FakeOpenAIServer, text_reply

YES_WORD = sys.argv[2] if len(sys.argv) > 2 else "chair"


def handler(request, index):
    text = json.dumps(request.get("messages", []))
    if "unsure_please" in text:
        return text_reply('{"answer": "unsure", "plausibility": 40, "reason": "too far to tell"}')
    if YES_WORD in text:
        return text_reply('{"answer": "yes", "plausibility": 92, "reason": "a chair with four legs is in the centre of the view"}')
    return text_reply('{"answer": "no", "plausibility": 85, "reason": "I see an empty floor and a wall"}')


with FakeOpenAIServer(handler=handler, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8123) as srv:
    print("fake vlm at", srv.base_url, flush=True)
    while True:
        time.sleep(1)
