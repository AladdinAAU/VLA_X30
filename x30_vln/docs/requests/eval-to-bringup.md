# Requests from EVAL to BRINGUP / INFRA

1. **Config overlay (needed for sweeps).** `X30_CONFIG_OVERLAY` = one or more YAML file paths (os.pathsep separated, later wins).
   An overlay is a mapping keyed by config file stem relative to `configs/` (`robot_sim`, `models`, `modes/object_nav`, `agent/object_nav`, ...), each value a partial
   dict deep-merged over the loaded file (dicts merge recursively; scalars and lists replace). Every node/launch file that reads a config file should call, right after loading it:
   `cfg = x30_common.config.apply_overlay("<stem>", cfg)`. Reference implementation (pure python, tested): `x30_eval/core/overlay.py` (`apply_overlay`, `deep_merge`).
   Suggested: move that file (60 lines) into `x30_common` and have `x30_eval` import it. Launch files should forward the env var to nodes (it is inherited by default).
   The nodes read configs at start, so the sweep restarts the stack per trial: `configs/eval/sweeps/*.yaml: stack_restart_cmd` is a shell command template with `{overlay}`.
   Please provide that command (e.g. a `scripts/x30.ps1 restart-framework -Overlay <file>` / `x30.sh`), or tell me the right one.
2. **Framework image**: please add `matplotlib` to `docker/images/framework/requirements.txt` (reports draw PNGs; it is currently only present transitively). `optuna` is already there.
3. **Eval entry**: the runner is `ros2 run x30_eval run_episodes ...` inside the framework container with `X30_ROOT=/x30` (reads `/x30/configs/eval/eval.yaml`, writes `/x30/results/`). A `scripts/x30` subcommand `eval` wrapping that would help.
   It should be started with a dedicated `ROS_DOMAIN_ID` when other tests run in parallel.
4. **Reset services**: the runner calls `/semantic_map/reset` (required) and, if present, `/mapping/reset` and `/slam/reset` (see eval-to-navigation.md). Without them
   the runner logs a note per episode ("mapping state NOT reset") and the episode record carries it; the alternative `reset.strategy: relaunch` needs `reset.relaunch_command`
   (a command that restarts the framework container's navigation nodes); tell me what to put there.
5. **Held-out lock**: `results/.heldout_lock.json` must not be deleted by cleanup scripts (it is the audit trail for the held-out split).
