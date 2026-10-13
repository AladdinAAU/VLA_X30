# Requests from EVAL to AGENT

The runner sends these keys inside `Navigate.params_json` (and `goal.episode_id` is already the unique trace id, e.g. `house1_chair_01_r0`):
`seed` (int, vary sampling/any randomness with it), `episode`, `repeat`, `trace_dir` (absolute directory, same as the agent `trace_dir`), `trace_id`, `trace_file`, `frames_dir`.
Please: (1) use `trace_dir` from `params_json` when present (the agent then writes `<trace_dir>/<safe_name(episode_id)>.jsonl` and `<id>_img/`, which is exactly what the exporters read);
(2) honour `seed`; (3) keep the trace row fields the evaluator reads: `kind=turn` rows with `turn, tool, args, parse_ok, refused, success, latency_s, prompt_tokens, completion_tokens, images, messages`,
`kind=finish_guard` rows with `allowed`. The SFT exporter needs `messages` (trace.messages: true) and `images` on every turn; image placeholders in `messages` must appear in the same order as `images`.
(4) `Navigate.Result.stop_reason`: the evaluator treats only `finished` as "the system stopped by itself"; `success` is the claim. Please never report `finished` for a give-up (use `aborted`/`no_progress`).
(5) Connection problems to the VLM should end with a message containing e.g. "LLM error"/"Connection refused" and stop_reason `error`; the runner then re-runs the episode as an infrastructure failure
(patterns in `configs/eval/eval.yaml: infra.error_patterns`).
