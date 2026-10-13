# x30_agent

Language-model reasoning for the X30 VLN framework. Two action servers, both `x30_msgs/action/Navigate`, delegated to by `mode_manager` (x30_modes):

| Node (executable) | Action | Mode | What it does |
|---|---|---|---|
| `reasoner_agent` | `/x30/agent/run` | `object_nav` | Tool-calling loop. Tools are generated from `configs/skills.yaml` (search, goto_object, approach, verify, explore, scan, goto_pose, turn_to, move_relative, stop) plus the local tools `query_map` and `finish`. One tool call per turn. |
| `instruction_follower` | `/x30/agent/follow` | `velocity_follow` | NaVILA-style: instruction split into subgoals, recent camera frames, one primitive per step (`move_forward 50`, `turn_left 30`, `stop`...) executed through `move_relative`/`turn_to`/`stop`; optional VLM stop check. |

Both nodes have the same plumbing (`ros/nodes.py`): one episode at a time (a second goal is rejected), feedback (`status`, `current_skill`, `agent_turn`), `Result.stop_reason` in `finished|timeout|aborted|safety|no_progress|error`, cancel interrupts the in-flight model request and the running skill and returns within about 0.3 s, a crash inside an episode still ends in an `error` result. The robot side is reached only through `x30_msgs` interfaces (`ros/ros_env.py`): skills `/x30/skill/<name>`, semantic map services `/semantic_map/{summary,query,find,relations}`, TF `map->base_link`, `/camera/color/image_raw`, `/semantic_map/annotated_map`.

## What the harness does besides asking the model
Rejected-object memory injected into every prompt; identical repeated calls and repeated failed calls refused; second `scan` near the first refused; every skill result is followed by a `NEXT:` hint; `verify` without a question gets one from the task; `finish(success=true)` is guarded (a fresh `verify` yes for the object, robot within 1.5 m, not moved since); malformed replies re-asked (`format_retries`); token budget against the server context (`reasoner_vlm.context_tokens`, old history is dropped first); every turn traced to `results/<run>/traces/<episode>.jsonl` (+ the images used). Everything is a number in `configs/agent/*.yaml`; the prompt files and the yaml are hashed and the hash is in every trace.

## Parameters and per-goal overrides
Node parameters (`config/agent.yaml`): `agent_kind` (`llm` | `scripted_search` | `dead_reckon`), `results_dir` (default `/x30/results`, env `X30_RESULTS`), `run_name`, `write_traces`, `config_dir` (else `X30_CONFIG_DIR`), `models_config`, topic names. `Navigate.params_json` may contain `agent_kind`, `max_turns`, `policy` (nested overrides of the yaml, e.g. `{"policy": {"images": {"camera": true}}}`), `reference_path` and `reference_frame`/`start` (dead_reckon), `use_verify`.
Baselines for ablations: `scripted_search` (no LLM decisions: explore until the map has the label, approach, verify, finish), `dead_reckon` (follow a reference path from odometry only).

## Run
```
# inside the x30-framework container (workspace built with colcon: x30_msgs, x30_common, x30_agent)
export X30_REASONER_URL=http://vlm:8000/v1        # default from configs/models.yaml
ros2 launch x30_agent agent.launch.py agent_kind:=llm run_name:=my_run
```
Model server: `x30.ps1 up models` (llama.cpp, Qwen3-VL-8B Q4_K_M, context 16384).

## Tests
* Pure python (no ROS, no GPU): `cd ros2_ws/src/x30_agent && python -m pytest -q` (reasoner, follower, baselines, prompts against a fake OpenAI server and a fake 2D world).
* ROS plumbing: `docker exec <framework container> bash /x30/ros2_ws/src/x30_agent/tests/ros/run_ros_check.sh` starts `tests/ros/fake_stack.py` (fake skill server, fake semantic map services, TF, camera, mock model) and both nodes and runs `ros_check.py` (17 checks). `run_chain_check.sh` does the same through the real `mode_manager` + `skill_server` of x30_modes on its fake robot.
* Live model in the fake world: `python scripts/run_fake_world.py --mode object_nav|velocity_follow --agent llm --n 20 --url http://127.0.0.1:8000/v1` and `python scripts/live_eval.py` (single-turn states with known best step; latency, format errors).
The fake world (`tests/fake_world.py`) is not a model of the robot: success rates measured there say nothing about the X30. See `docs/reports/agent.md`.
