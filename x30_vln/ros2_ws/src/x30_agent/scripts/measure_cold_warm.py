#!/usr/bin/env python3
"""Per-turn latency of the reasoner prompt with a cold prefix (llama.cpp prompt cache miss) and a warm one (same prefix again).
Adds {"cold_warm": {...}} to results/agent_live/live_eval/report.json.   X30_REASONER_URL=http://127.0.0.1:8000/v1 python measure_cold_warm.py"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests"), str(ROOT), str(ROOT.parent / "x30_common")]
import live_eval as L  # noqa: E402
from x30_common import VLMClient, load_models_config  # noqa: E402

from x30_agent.core.policy import load_policy  # noqa: E402
from x30_agent.core.prompts import build_system_prompt  # noqa: E402
from x30_agent.core.skills import load_registry  # noqa: E402
from x30_agent.core.tools import build_toolset  # noqa: E402

cfg = ROOT.parents[2] / "configs"
pol = load_policy("object_nav", cfg / "agent")
ts = build_toolset(load_registry(cfg / "skills.yaml"), list(pol.tools), list(pol.local_tools), pol.description_mode, pol.tool_descriptions)
system = build_system_prompt(pol)
client = VLMClient.from_config("reasoner", load_models_config())
cold, warm, pt, ct = [], [], [], []
for i, s in enumerate(L.make_states(random.Random(3), 3)[:12]):
    m = L.build_messages(s, f"[session {i}-{random.random()}] " + system, pol.hints)     # unique prefix: forces a prompt-cache miss
    r = client.chat(m, ts.tools, "auto", max_tokens=300, temperature=0.2)
    cold.append(r.latency_s); pt.append(r.prompt_tokens); ct.append(r.completion_tokens)
    warm.append(client.chat(m, ts.tools, "auto", max_tokens=300, temperature=0.2).latency_s)
out = {"n": len(cold), "cold_mean_s": round(statistics.mean(cold), 3), "cold_max_s": round(max(cold), 3), "warm_mean_s": round(statistics.mean(warm), 3),
       "prompt_tokens_mean": round(statistics.mean(pt)), "completion_tokens_mean": round(statistics.mean(ct), 1)}
p = ROOT.parents[2] / "results" / "agent_live" / "live_eval" / "report.json"
rep = json.loads(p.read_text())
rep["cold_warm"] = out
p.write_text(json.dumps(rep, indent=2))
print(json.dumps(out))
