import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent / "x30_common")]

from x30_common import VLMClient  # noqa: E402
from x30_common.testing.fake_openai import FakeOpenAIServer  # noqa: E402

from x30_agent.core.policy import load_policy  # noqa: E402
from x30_agent.core.reasoner import ReasonerAgent  # noqa: E402
from x30_agent.core.skills import load_registry  # noqa: E402

CONFIGS = HERE.parents[3] / "configs"


@pytest.fixture(scope="session")
def registry():
    return load_registry(CONFIGS / "skills.yaml")


@pytest.fixture()
def srv():
    with FakeOpenAIServer() as s:
        yield s


@pytest.fixture()
def make_agent(srv, registry, tmp_path):
    def _make(world, overrides=None, cancel=None, client_kw=None, policy="object_nav"):
        pol = load_policy(policy, CONFIGS / "agent", overrides)
        kw = dict(backoff_s=0.01, timeout_s=10, max_retries=1)
        kw.update(client_kw or {})
        client = VLMClient(srv.base_url, "fake", context_tokens=16384, **kw)
        return ReasonerAgent(client, world, registry, pol, trace_dir=str(tmp_path / "traces"), cancel=cancel)
    return _make
