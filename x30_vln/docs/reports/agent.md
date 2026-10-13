# Report: x30_agent and x30_common (language-model reasoning)

Packages `ros2_ws/src/x30_common` (VLM client, config; no ROS needed) and `ros2_ws/src/x30_agent` (reasoner, instruction follower, baselines, ROS nodes). Configs: `configs/models.yaml`, `configs/agent/*`.
How to run: `ros2_ws/src/x30_agent/README.md`. Everything below was measured in this work; what was not tested is listed in section 7.
Result files: `results/agent/*.json` (ROS checks), `results/agent_live/*` (live-model runs, traces, `summary.json` per run). Figures: `docs/figures/agent/` (captions there).

## 1. What exists

| Part | Files | Status |
|---|---|---|
| `VLMClient` (OpenAI-compatible, sync/async, cancel, retry, usage, tool-call normalisation, JSON repair, seed), models.yaml loader, image encoding | `x30_common/*` | 60 pytest pass |
| Reasoner (object_nav): tool-calling loop over the skill registry, hints, rejected-object memory, repetition refusal, guarded `finish`, token budget, traces | `x30_agent/core/reasoner.py, hints.py, memory.py, guard.py, tools.py, prompts.py, skills.py, trace.py`, `configs/agent/object_nav*.{yaml,md}` | tested (46 pytest with a mock server, fake world, live model) |
| Instruction follower (velocity_follow): subgoal planner, NaVILA-style primitives, loop detector, stop check, measured progress, auto-advance of metric subgoals | `core/follower.py, primitives.py`, `configs/agent/velocity_follow.yaml, follow_*.md` | tested as above; weak in the fake world (section 4.4) |
| Baselines for ablations: `ScriptedSearch` (no LLM decisions), `DeadReckon` (reference path from odometry) | `core/baselines.py` | pytest + through the ROS node |
| ROS nodes `reasoner_agent` (`/x30/agent/run`), `instruction_follower` (`/x30/agent/follow`), launch file, parameters | `ros/nodes.py, ros/ros_env.py`, `launch/agent.launch.py`, `config/agent.yaml` | built and run in the x30-framework image (section 3) |
| Test rigs: pure-python fake world + rule-based mock models; ROS fake stack (skill server, semantic-map services, TF, camera, mock model) | `tests/fake_world.py, mock_models.py`, `tests/ros/*` | used for everything below |
| Live scripts | `scripts/live_eval.py, measure_cold_warm.py, run_fake_world.py, make_figures.py` | used for section 4 |

## 2. Tests without ROS and GPU
`x30_common`: 60 tests (client retry/cancel/overflow/seed, JSON repair, tool schema coercion, config and images), 14 s. `x30_agent`: 46 tests (reasoner incl. guard, refusals, retries, cancel, token budget; follower incl. blocked feedback, stop check,
loop detector, metric auto-advance, cancel; baselines; prompts), 40 s. A regression test exists for a real bug found earlier (unquoted `yes:`/`no:` hint keys parsed as YAML booleans).

## 3. ROS verification (image `x30-framework:latest`, ROS 2 Humble, `ROS_DOMAIN_ID=60`, container `x30-agent-test` on `x30_net`)
`x30_msgs`, `x30_common`, `x30_agent` built with `colcon build --symlink-install` (11 s). `ros2 launch x30_agent agent.launch.py` starts both nodes; both action servers appear.

### 3.1 Fake stack, 19/19 checks (`tests/ros/run_ros_check.sh`, data `results/agent/ros_check.json`)
`tests/ros/fake_stack.py` wraps the fake world as ROS interfaces with the names and JSON shapes of the real system: action servers `/x30/skill/<name>` for the 10 skills of `configs/skills.yaml`
(motion skills exclusive, `stop` cancels, every skill honours cancel), services `/semantic_map/{summary,query,find,relations}` in the format of x30_semantic_map, TF `map->base_link`, camera and annotated-map topics,
and a rule-based mock model server (so no GPU). It is a plumbing test: mock model decisions say nothing about the real model.

| Check | Result |
|---|---|
| object_nav through `/x30/agent/run`, 3 worlds | 3/3 finished, ground truth within 1.3 m, 3-12 turns, feedback published (35 messages in the longest) |
| velocity_follow through `/x30/agent/follow`, 3 worlds | 3/3 finished with a result; 2 of 3 end within 1.5 m (mock follower with injected noise) |
| `params_json` `agent_kind=scripted_search` / follow with `scripted_search` | baseline works; the follower answers with an `error` result and a clear message |
| bad `params_json`; second goal while busy | error result in < 1 s; second goal rejected |
| **Cancel** during a model call (model delayed 8 s) / during a skill (20 s) / follower during a primitive | result after **0.21 / 0.18 / 0.24 s**, `stop_reason=aborted`, goal status canceled (limit was 5 s) |
| eval-runner keys: `trace_dir`, `seed` in `params_json`; give-up | trace file `<trace_dir>/eval_ep_1.jsonl` written; a request for an object that does not exist ends as `no_progress` (not `finished`) |
| node usable after a cancel | yes |

### 3.2 Real `mode_manager` + `skill_server` of x30_modes on its fake robot (`tests/ros/run_chain_check.sh`)
One goal "Find the chair and stop next to it." sent to `/x30/navigate` (mode_manager) was delegated to `/x30/agent/run`, executed with the real `skill_server` (search -> goto_object -> verify -> finish) on the fake robot of x30_modes,
with a mock model: success, `finished`, 4 agent turns, path 5.46 m, 49.3 s (the fake robot moves in real time). The delegation, feedback relay and result path work. Only one episode was run.

## 4. Live measurements with the real model
Model server: llama.cpp `server-cuda`, Qwen3-VL-8B-Instruct Q4_K_M, `-c 16384`, one slot, RTX 3090 Ti (container `x30-vlm`, started for this work and stopped afterwards). Client on the Windows host.
**Finding:** `http://localhost:8000` costs about 2 s per request on this host (IPv6 fallback); use `127.0.0.1`. All numbers below use `127.0.0.1` (a first run with `localhost` showed 2.6 s per turn, discarded). `configs/models.yaml` now says so.

### 4.1 Latency per turn (`scripts/live_eval.py`, `scripts/measure_cold_warm.py`; object_nav prompt, 3.37k tokens, 27 completion tokens)
| Case | Mean latency |
|---|---|
| Cold prompt cache (system prompt + tools not cached) | 1.28 s (max 1.42 s), n = 12 |
| Cached system prompt (normal case in an episode) | 0.28 s, n = 12 |
| No image / camera / map / map+camera attached (cached prefix, small fake-world images) | 0.39 / 0.33 / 0.32 / 0.33 s, n = 12 each |
Generation speed measured by the server: about 40 tokens/s; prefill of a cold 3.4k-token prompt about 0.6 s. Images added only 60-100 prompt tokens here because the fake-world images are small (a 448x336 test image costs 154 tokens); the latency
of images at the real camera resolution was **not** measured. Real episodes also pay a model call for `verify` inside the skill server (not measured here).

### 4.2 Tool-call format errors (72 invented states x 2 settings = 144 requests, raw first reply, no retries)
0 server errors, 0 replies without a tool call, 0 multiple calls, 0 unknown tool names, 0 unparseable arguments, in both `tool_choice=auto` and `required` (identical results, so `auto` stays the default).
The only format problem: 9 of 72 replies (12.5 %, all in the "arrived, now verify" state) called `verify` without its required `question`; the harness fills the question from the task, after which 0 invalid calls remain.
Over the 1642 model calls of the seven live fake-world runs of section 4.4 the harness counted 0 format retries, 0 invalid calls and 0 turns without a usable action (so in practice the retry path never fired with this model).

### 4.3 Decision quality on a hand-made set (n = 72 states: 8 types x 9, invented maps, one correct step per state; states of one type differ only in labels/numbers, so they are not independent)
| State type | Correct raw (of 9) | Note |
|---|---|---|
| A empty map -> search/scan/explore | 9 | |
| B target in map -> goto_object(id) | 9 | correct id every time |
| C arrived -> verify | 0 raw, 9 with harness | always chose `verify`, always omitted `question` |
| D verified yes and close -> finish(true) | 9 | |
| E rejected object, other candidate known -> goto the other id | 9 | |
| F map exhausted -> give up | 2 | 7 of 9 searched a synonym (the hint offers it); counts as acceptable ("with harness" 9/9) |
| G search failed, not exhausted -> explore/scan | 9 | |
| H unsure verify -> approach/turn_to | 9 | |
Raw 56/72 = 78 %, with the two harness-tolerated cases 72/72. This set was written by the author of the prompts, so it overstates generalisation; it shows the model follows the hint protocol, not how well it navigates.

### 4.4 Episodes in the FAKE world with the live model (`scripts/run_fake_world.py`; 2D three-room world, not the robot)
Success = the agent claimed success and ended within 1.5 m of the target in the fake world's ground truth.
| Run (n) | Success | Claimed | Stop reasons | Mean turns | Notes |
|---|---|---|---|---|---|
| object_nav, text only (20) | 20/20 | 20 | finished 20 | 4.3 | 85 model calls, 0 retries |
| object_nav, map + camera images (20) | 20/20 | 20 | finished 20 | 4.4 | |
| object_nav, verify skill wrong in 20 % of calls (20) | 18/20 | 19 | finished 20 | 4.5 | 1 false claim after a wrong verify |
| object_nav, requested object absent (10) | 10/10 gave up | 0 | no_progress 10 | 2.9 | all report "not found" after 2-3 turns (map fully revealed by the first scan) |
| velocity_follow, target only: "Walk to the red table in the kitchen" (20, first prompt version) | 1/20 | 5 | timeout 13, no_progress 2, finished 5 | 33 | task is a blind search without map; the 8B model wanders |
| velocity_follow, metric route description, **first prompt** (20, dev seeds) | 1/20 | 1 | no_progress 13, timeout 6, finished 1 | 28 | model ignored amounts ("1.5 meters" -> 250 cm) and never set `subgoal_done` |
| same, + "do exactly the stated amount" rule (dev / fresh seeds) | 1/20 / 0/20 | 4 / 3 | | 24 / 22 | |
| same, + measured-progress line and planner `done_when` (dev / fresh) | 1/20 / 2/20 | 1 / 3 | | 27 / 22 | |
| same, + **harness auto-advance** of subgoals that name an angle or forward distance (final; dev / **fresh seeds**) | 9/20 / **6/20** | 17 / 16 | finished 17 / 16 | 12 / 15 | 8 / 10 false claims |
The follower prompt and harness were changed three times while looking at the dev seeds (100-119); the fresh seeds (200-219) were run once with the final version and are the number to quote: **6/20 (30 %)**, with 10 episodes in which the
agent stopped 2-5 m from the target (claimed success, ground truth says no). Causes seen in traces: the fake world's route text is not always executable (a straight leg that clips a wall gives "Blocked"), and the 8B model
does not reliably set `subgoal_done`, hence the harness now advances metric subgoals itself (`prompt.auto_advance_metric`, switchable). The fake object_nav world is easy: every object is revealed by the first 360 degree scan, so 20/20 shows that the loop,
guards and hint protocol work with the live model, not that object search works. Nothing here is a robot result.

### 4.5 Prompt token budget (server context 16384; `reasoner_vlm.context_tokens` set to 16384)
object_nav prompt: system prompt + 10 skill tools + 2 local tools about 3.3k tokens (tools JSON 7.8k characters); per turn mean 3.75k, **max 4.6k** over 20 episodes of 3-10 turns (4.6k also with map + camera images), 3.86k max in the give-up runs.
So the budget used is at most 28 % of the context in these short episodes; longer episodes are bounded by `history_units` (5 call/result pairs) and the budgeter, but 30-turn episodes were not run. follower prompt: mean 1.2-1.4k, max 1.6k tokens (4 frames at 336 px).
The first object_nav run and the live_eval run read `context_tokens: 8192` (the yaml was changed during the work); prompts were far below both values, so the budgeter never trimmed.

### 4.6 Whole ROS path with the live model (fake stack, `tests/ros/run_ros_live.sh`, `results/agent/ros_live.json`)
The two nodes in the container talked to `x30-vlm` over `x30_net`: object_nav 4/5 (the failure: "white tv in the office", `no_progress` after 10 turns), velocity_follow with target-only instructions 0/5 (2 false claims, 2 timeouts, 1 no_progress). Samples are tiny; they only show that the full path works with the live model.

## 5. Bugs and findings during this work
* `ros_env.object_info` and `query_map(get)` sent `{"id": ...}` to `/semantic_map/query`, which has no id filter (it returned the first 10 objects). Now the client queries with `limit 500` and filters; `get`/`relations` use `/semantic_map/relations`. Request filed: `docs/requests/agent-to-semantic_map.md`.
* A fake world that teleports races with TF: the agent measured "moved 2 m since verify" because the pose came from before the jump. The fake skill now publishes TF before returning. (On the robot TF is continuous.)
* `finish(success=false)` and "not found" were reported as `finished`; the eval contract (`docs/requests/eval-to-agent.md`) says only a self-stop is `finished`. They now end as `no_progress`; model-server errors start with "LLM error" so the runner retries them as infrastructure failures; `params_json` keys `trace_dir` and `seed` are honoured (`VLMClient.seed` sends the llama.cpp `seed`).
* `localhost` costs 2 s per request on the Windows host (see section 4).
* 8B model weaknesses that needed harness help: omits `verify.question`; ignores metric amounts and never sets `subgoal_done` reliably in the follower; prefers synonym searches over giving up.

## 6. Step 3 (one real object_nav episode in the simulator): not done
Condition not met: `docs/reports/navigation.md` does not exist (navigation was still running its benchmark: `docs/reports/navigation-progress.md`), so there is no `/map`, Nav2 or `/costmap_cloud` for `goto_pose`/`goto_object`/`approach` in the simulator.
The simulator container (`x30-sim`) exists, but a full-stack episode also needs the perception server with a restricted vocabulary through `/perception/vocabulary`, the semantic mapper, x30_navigation and the skill server together with the VLM: bring-up would exceed the 30 minutes budget. Not attempted.
What is verified instead: the agents against the real `skill_server` on x30_modes' fake robot (3.2) and against the fake stack (3.1).

## 7. Not tested / limits / next steps
* No run on the robot or in the simulator; no real semantic map (only the fake and the `x30_modes` fake robot); real camera frames and real map images at full size were not sent to the model; latency with real-size images is unknown.
* Fake-world success rates say nothing about navigation quality. object_nav in the fake world is easy; the follower is weak (30 % on fresh seeds) and makes false stop claims; the real VLN benchmark (`x30_eval`) is the place to measure it.
* Only Qwen3-VL-8B Q4_K_M was tried; no comparison with a larger model, no fine-tuning (traces store messages and images for later export).
* Long episodes (30 turns, prompt growth), several consecutive episodes in one node process with the real skill server, and the `/x30/primitive` service (the follower uses `move_relative`/`turn_to`/`stop` skills instead) were not exercised.
* Cancel was measured with fast fake skills; a real motion skill must react to the `stop` the agent sends (the agent sends `stop` and cancels the skill goal, then returns).
* Seeds: `seed` is passed to the server, but llama.cpp sampling with `--parallel 1` and prompt caching is not guaranteed bit-reproducible across runs; not checked.
* Next: run the eval benchmark with `x30_eval`, compare `scripted_search` and `dead_reckon` baselines on the same episodes, test `description_mode: full` and `examples: []` ablations (implemented as policy keys, not measured), re-measure latency with real images.
