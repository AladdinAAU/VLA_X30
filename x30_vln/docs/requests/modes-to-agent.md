# modes -> agent

## 1. Skill registry is ready (2026-10-09)

`configs/skills.yaml` (version 1). Names and argument names are frozen; additions are fine, renames will be listed
here. Notes for the schema generator:

- Per skill: `name`, `description` (written for the LLM), `args_schema` (JSON schema; `additionalProperties: false`),
  `args_require_one_of` (list of argument-name lists, at least one entry of the list must be present; for
  `goto_object`, `approach`, `turn_to`, `verify`; mention it in the tool description, it is not in the schema),
  `result.data` (fields of `data_json`), `typical_duration_s`, `default_timeout_s`, `preconditions`.
- `bias` (explore) and `bearing` (turn_to) use `anyOf` string/number-array; if your provider cannot handle `anyOf`,
  flatten them to `type: string` (the skill server parses numbers and "x,y" strings too).
- The skill server returns `success`, `message` (short, ends with a "Next:" hint on most failures) and `data_json`.
  Unknown or invalid arguments are returned as `success=false` with the reason in `message`, never as a rejected goal.
- `stop` and `verify` may run while another skill runs; all other skills are exclusive (a second motion goal is
  rejected with a message). `stop` cancels the running skill.
- Rejections: `search` and `goto_object` accept `rejected_ids` so your harness can inject the remembered ones.

## 2. Delegation targets of the mode manager

`mode_manager` forwards `velocity_follow` goals to the action `/x30/agent/follow` and `object_nav` goals to
`/x30/agent/run`, both `x30_msgs/action/Navigate`, with the goal copied unchanged (instruction, target_label,
params_json, timeout_s). It relays feedback and uses your result's `success`, `stop_reason`, `message`,
`agent_turns`. It cancels your goal on mode switch, timeout, e-stop or teleop takeover and expects the result within
~5 s of the cancel. It adds `path_length_m` and `duration_s` from odometry itself.

## 3. Velocity primitives for the instruction follower

Service `/x30/primitive` (`x30_msgs/srv/Json`), blocking until the primitive finished:
request `{"action": "move_forward", "value": 75}` (actions: `move_forward`, `move_backward`, `strafe_left`,
`strafe_right` in cm; `turn_left`, `turn_right` in degrees; `stop`) or `{"text": "move forward 75 cm"}`
(NaVILA-style sentence). Response `{ok, response}` with `response` = JSON `{success, message, achieved, ...}`.
The same executor is available as skills `move_relative` / `turn_to`.
