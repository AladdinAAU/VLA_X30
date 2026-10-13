"""Mock model for the chain check against the REAL skill_server/mode_manager of x30_modes (with its fake robot):
requests with tools get the rule-based object-nav mock, tool-less requests (the verify skill) get a yes."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1]), str(HERE.parents[2] / "x30_common")]
from mock_models import ObjNavMock  # noqa: E402
from x30_common.testing.fake_openai import FakeOpenAIServer, text_reply  # noqa: E402

mock = ObjNavMock(seed=0, p_malformed=0, p_text=0, p_nocall=0, p_repeat=0, p_prefinish=0)


def handler(req, idx):
    if req.get("tools"):
        return mock(req, idx)
    return text_reply('{"answer": "yes", "plausibility": 92, "reason": "a chair is in the centre of the view"}')


with FakeOpenAIServer(handler=handler, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8123) as srv:
    print("chain vlm at", srv.base_url, flush=True)
    while True:
        time.sleep(1)
