# HW agent progress

- M1 (done): x30_hw/protocol.py, tools/fake_core.py, tools/fake_x30.py; pytest 38 passed (protocol golden vectors, speed mapping, fake state machine).
- M2 (written, not yet compiled): x30_nav_bridge fork (velocity_mapping.hpp, telemetry_utils.hpp, core/control/io .cpp, params). Next: gtests, launch, CHANGES.md, docker build.
- Next: x30_hw nodes, launches, tools, configs, runbook, integration run, hw.md.
- M3: nav_bridge compiles in x30-framework container, 15 gtests pass; found+fixed upstream bug: warmupControl spun ~1000 heartbeats/ms flood that dropped the command after it (fake_x30 now flags heartbeat floods).
- M4: x30_hw written: protocol, fake_x30, safety_gate, real_adapter, static_tf, camera/thermal ingest, replay_stub, launches (real, replay), configs/robot_real.yaml, ros1_bridge lists. Untested in ROS yet. Next: build x30_hw in container, integration, tools, docs.
- M5: dry-run integration in container (domain 50): 56/57 then fixed check; replay check passes (lidar 10 Hz sim time, contract fields, TF, stub). Reports in results/hw/. Next: tools (calibration, measure_*, clock_offset, preflight, x30ctl, record_telemetry), tests, runbook, CHANGES, hw.md.
