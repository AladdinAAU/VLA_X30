# EVAL progress notes

- M1 (done): core library (geometry, grid/A*, metrics, scoring, failure classifier, stats, overlay, manifest, runlog, heldout guard), runner, fake env, RosEnv (untested in ROS), CLI, report module. 32 pytest tests pass on the host (fake env, no ROS).
- M2 (in progress): sweep harness written (space/evaluator/harness), needs tests + CLI. Then exports, package files, container test of RosEnv with a mock stack, real-sim episode, docs.
- M2 (done): sweep harness (Optuna, successive halving, confirm/finalize guards), exporters (SFT hf/llamafactory, NaVILA steps, perception), 55 pytest tests pass (fake env, host Python 3.12).
- M3 (next): package files done; ROS node test in the x30-framework container with a mock stack, real sim episode attempt, docs (EVALUATION.md, README, requests, eval.md), fine-tune recipe research.
- M3a (done): package files, mock ROS stack, RosEnv tested in x30-framework container (domain 60, localhost only): reset/alignment/ATE/infra retry/abort verified against the MOCK; sim is not running yet (sim agent at RTF tuning). Next: docs, requests, real-sim attempt.
