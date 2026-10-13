# INFRA progress notes (short; final report is docs/reports/infra.md)

Done and verified (2026-10-09):
- images x30-framework (7.49 GB, build ~5 min with ros1_bridge), x30-ros1-bridge (3.37 GB) built; perception image NOT built yet
- compose.yml / compose.models.yml / compose.real.yml, dds/*.xml, scripts/x30.ps1 (status, build work), x30.sh, x30-guard.ps1/.sh, check_real_network.ps1 (never run)
- (c) colcon build x30_msgs + `ros2 interface show` OK (12.9 s incl. container start)
- (a) Humble<->Jazzy DDS proven: cyclonedds.xml with AllowMulticast=spdp (multicast discovery, unicast data), 0 drops at 10 Hz 2.08 MB cloud, 15 Hz image, 50 Hz tf, 20 Hz Twist; latency 3-5 ms cloud; ~11-14 % of one core. Plain multicast=default gave ~5 % image drops best-effort. Results in results/infra/dds_*.json
- SocketReceiveBufferSize min must be <= kernel rmem_max*2 (VM rmem_max 4 MB): min=2MB

TODO: (b) bridge chain (Foxy cyclone rejects XML? bridge crashed with rmw handle invalid), (d) vlm+perception proofs, (e) guard run, docs/INFRA.md, requests to sim, final report; delete .env copy if unchanged
Helper scripts: docker/scripts/proof_dds.sh, proof_bridge.sh, dds_probe.py, count_topic.py, summarize_probe.py
- (b) DONE: bag->bridge->framework 177/178 msgs once (10.01 Hz, gaps>150ms:2), loop 355 msgs/35 s 10.02 Hz; Foxy needs docker/dds/cyclonedds-foxy.xml (no SocketReceiveBufferSize); roscore/rosbag need 'ros1' prefix in bridge image. results/infra/bridge_*
- (d) DONE: vlm (reused running x30-vlm from my compose): text 0.34s 112 tok/s gen, image 0.35-0.52s ~82 tok/s; perception image built (base drone-perception:latest, 10.9 s), placeholder /health ok cuda=true on :8101; total GPU 11.1GB with vlm+other agent perception
- (e) guard DONE (ps1+sh; uses MemTotal-MemAvailable of docker-desktop VM, vmmem includes 13 GB cache); launcher up/replay/down tested; bag_compat.py needed for Humble replay of v9 metadata; domain cross-talk with other agents on x30_net seen. TODO: kiss_icp functional, rosdep check, INFRA.md, requests infra-to-sim.md, final report, rm .env copy
