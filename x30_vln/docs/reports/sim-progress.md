# SIM progress notes

- 2026-10-09 milestone 1: image `x30-sim:latest` built (Ubuntu 24.04, Jazzy, Harmonic 8.15, upstream controller + patch in /opt/x30_ext). compose.sim.yml / compose.sim.gpu.yml written.
- milestone 2: package x30_sim skeleton (config, lidar_model, lidar_merge, robot_description, worlds, sdf_scene, xacro with 4 Livox + RGBD + thermal + IMU). Both URDFs (legged, kinematic) validate with `gz sdf -k`; headless ogre2 + llvmpipe loads the arena and all sensor topics appear.
- next: bridge + nodes (sim_state, kinematic_driver, thermal_processor, ground_truth, sim_services), sim.launch.py, kinematic smoke test, then legged, lidar comparison, report.
- milestone 3: all nodes + sim.launch.py written; kinematic launch brings up every contract topic. First measurement with FULL sensor load (4x20k-ray lidars, 1280x720 RGBD, 384x288 thermal) under llvmpipe: RTF ~0.002 (single sim thread at 100 %). Sensor budget tuning needed (scripts/sim/experiment.sh).
- milestone 4 (2026-10-09): **kinematic mode with the full sensor contract works and is smoke-tested** (`scripts/sim/smoke_test.sh kinematic 20`: 26/26 checks;
  lidar 10 Hz x 79872 pts in the exact contract layout, IMU 200 Hz, odom 50 Hz, RGB 640x480 10 Hz, depth, thermal 384x288 5 Hz with temperatures, TF complete,
  3 m drive odom==GT, /sim/set_robot_pose OK, RTF ~1.0). How to use: `docker compose -f docker/compose.sim.yml --profile sim up -d sim; docker exec x30-sim x30-build;
  docker exec -e ROS_DOMAIN_ID=10 x30-sim ros2 launch x30_sim sim.launch.py world:=default_arena locomotion:=kinematic` (README in ros2_ws/src/x30_sim).
  Worlds: `datasets/registry.yaml` ids (see docs/requests/sim-to-worlds.md for label/temperature conventions). Key finding: the upstream 1.3 M-vertex visual meshes
  cost ~100 ms per sensor frame in software rendering; they are decimated at launch (robot.visual_mesh_voxel_m), which brought RTF from 0.2 to ~1.0.
  Still to do: legged mode, lidar-vs-bag comparison, unit tests, image rebuild with the fixed x30-build, final report docs/reports/sim.md.
- milestone 5: lidar-vs-bag comparison works (`scripts/sim/capture_sim_scans.py` in the container + `scripts/sim/compare_lidar_to_bag.py` on the host); world `lab_room`
  (raised visual floor so the standing sim robot sees the same floor/ceiling spacing as the lying real robot). JS distance range 0.055 / elevation 0.016 / height 0.027;
  valid fraction 0.60 sim vs 0.47 real. 14 unit tests pass (`docker exec x30-sim x30-test`). Next: legged mode, rebuild image with fixed x30-build, report.
- milestone 6: **legged mode works too**; `scripts/sim/smoke_test.sh kinematic|legged 120` passes 29/29 checks in both modes (contract topics/rates/layout, TF, 3 m drive odom vs GT,
  set_robot_pose, e-stop hold, reset_world, GT objects). Measured (all sensors on, 640x480 RGBD + 4x20k-ray lidars + thermal 384x288@5): kinematic RTF 0.99, legged RTF 0.88;
  12-minute kinematic soak: RTF 0.98 mean, gz RSS 518 -> 516 MB (flat). Remaining: GPU-render test, image rebuild with fixed scripts, docs/reports/sim.md.
- milestone 7 (final): GPU render path verified (needs GALLIUM_DRIVER=d3d12 + WSLg display, RTF 1.0, Gazebo RSS +1.2 MB/s), image rebuilt with the fixed scripts, registry worlds apt_01/lab_01 run
  (kinematic RTF 0.82/0.74, legged lab_01 0.62, 29/29 checks), figures in docs/figures/sim, report docs/reports/sim.md, lidar comparison docs/reports/sim-lidar-compare.md.
