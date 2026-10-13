# modes progress notes

- 2026-10-09 M1: configs/skills.yaml (10 skills) published; docs/requests/modes-to-agent.md written.
- M2: pure logic (geometry, guard_logic, mux_logic, primitives, mode_logic, registry, semantic) + 78 pytest pass on host.
- M3: ROS nodes written (cmd_vel_mux, safety_guard, mode_manager, skill_server, skills/*), configs/modes/*.yaml, launch, package builds in x30-framework image (container x30-modes-test, build dir /tmp/ws).
- M4 (in progress): fake robot + stub agent under test/ros_integration; next: run integration scenarios, then README and docs/reports/modes.md.
- M4 (done): fake-robot integration in x30-framework container (ROS_DOMAIN_ID=30): mux 6/6, guard 9/9 (stop gaps 0.19-0.24 m at margin 0.20), hazards 5/5, primitives 19/19, skills 24/25 (fixed test), modes 14/15 (fixed stub). Full run in progress; then sim run (x30-sim image, kinematic) and reports.
- Note from sim: invalid lidar returns sit at sensor origins; the guard's self filter drops points inside the footprint, which covers them.
- M5 (done): simulator run (kinematic, domain 30): 14/14 checks, primitives <= 0.8 cm / 0.8 deg, guard stops 0.15-0.25 m from the wall (0.05 m short at 0.6 m/s). Fixes from the sim: self margin, leg zone, speckle filter, zero-intensity filter, coast_s calibration. goto_pose in the sim pending (no Nav2).
- M6 (done): README, docs/reports/modes.md, figures docs/figures/modes, requests to agent/navigation/semantic_map. Containers x30-modes-test and x30-sim stopped.
