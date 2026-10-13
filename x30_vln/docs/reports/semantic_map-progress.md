# semantic_map progress notes

## Milestone 1 (done, verified by running on the host with the synthetic scenario)
- Core pure-python modules written in `ros2_ws/src/x30_semantic_map/x30_semantic_map/`: config, vocabulary, datatypes,
  attributes, semantic_map (association, merging, forgetting, dynamic objects, ray triangulation with weighted LS,
  persistence), gridtools, rooms, relations, finder, summary, annotated_map, synthetic, annotator, ros_convert.
- `configs/semantic_map.yaml` written (thermal disabled by default: the stock X30 has no thermal camera).
- Synthetic scenario (8 seeds): recall ~0.92, mean position error ~0.05 m, 0 duplicates, 0-2 false positives per run.
- ASM images rendered and looked at: `docs/reports/semantic_map_asm.png`, `semantic_map_asm_ego.png`.

## Remaining
- ROS node file, package files (setup.py, package.xml, launch), unit tests, fake-VLM tests, ROS integration test in the
  `x30-framework` image, live VLM measurements, README and final report.

## Milestone 2 (done)
- ROS node, package files, launch, unit tests: 58 tests pass on the host (core, geometry, annotator against a fake
  OpenAI-compatible server). Next: ROS integration test in `x30-framework`, live VLM, README, final report.

## Milestone 3 (done)
- ROS integration (31 checks), live Qwen3-VL measurement, sim chain run, README and final report: see semantic_map.md.
