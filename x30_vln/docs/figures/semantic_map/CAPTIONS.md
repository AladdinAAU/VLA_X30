# x30_semantic_map figures

* `asm_synthetic.png`: annotated semantic map (north-up) rendered from the pure-python synthetic apartment scenario after the full run: free/occupied/unknown grid, tinted named rooms, trajectory, `id label` footprints, robot arrow.
* `asm_synthetic_ego.png`: the same map robot-centred (egocentric view, as it would be given to a VLM), from the synthetic scenario.
* `asm_ros_integration.png`: annotated map published by the running `semantic_mapper` node (`/semantic_map/render`) in the x30-framework container after the scripted Detection3DArray replay (18 objects, 3 named rooms, a moved chair and a walking person handled, 0 duplicates).
* `sim_lab01_camera.jpg`: one `/camera/color/image_raw` frame of the simulator world `lab_01` (flat-shaded SDF primitives, robot in the corridor), showing why open-vocabulary detections on simulated frames are unreliable (see docs/reports/semantic_map.md, "Simulator chain").
