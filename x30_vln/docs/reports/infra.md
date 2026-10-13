# INFRA report (docker, launcher, proofs)

Date 2026-10-09. Operator manual: `docs/INFRA.md`. Raw numbers: `results/infra/` (dds_*.json, bridge_*.txt, guard-*.log, build_framework.log).
Re-runnable proofs: `docker/scripts/proof_dds.sh`, `proof_bridge.sh`, `proof_vlm.py`, `check_framework.sh`.

## What exists
* `docker/compose.yml` (framework, tools, replay, ros1_bridge, network `x30_net`, colcon volumes), `compose.models.yml` (vlm, vlm2 profile `two-vlm`, perception),
  `compose.real.yml` (host-network overlay), `docker/dds/*.xml` (4 DDS files), `.env.example`. `compose.sim.yml` belongs to the SIM agent; it merges cleanly
  with mine (`config -q` ok, also with the real overlay). `include:` was dropped on purpose: it cannot skip a missing file (tested, error on compose 2.40.3);
  the launcher builds the `-f` list instead and appends `compose.sim.yml` when it exists.
* Images: `docker/images/framework` (Humble), `perception`, `ros1_bridge`. Helper commands in the framework image: `x30-build`, `x30-test`, `x30-rosdep`.
* Launcher `scripts/x30.ps1` and `x30.sh` (docker-start, up, down [profile], status, shell, exec, build, test, rosdep, images [build lock], logs, guard, replay),
  `scripts/x30-guard.ps1|.sh`, `scripts/check_real_network.ps1` (parse-checked only, never executed, by design).
* `docs/INFRA.md`.

## Images (measured)
| Image | Size | Build time |
|---|---|---|
| x30-framework | 7.67 GB (7.49 GB before the python-apt fix) | cold with ros1_bridge: 4 min 58 s; apt-layer change rebuild 4 min 19 s; python-only change a few seconds |
| x30-ros1-bridge | 3.37 GB (base ros:foxy-ros1-bridge 3.35 GB) | about 1 min (+ cyclone apt); entrypoint change 2.8 s |
| x30-perception | 10.7 GB | 10.9 s because built with `PERCEPTION_BASE_IMAGE=drone-perception:latest` (already contains ultralytics, CLIP, fastapi from the same recipe). The default base `ultralytics/ultralytics:latest` was NOT built here (would pull about 8 GB); the Dockerfile is the ported recipe and is untested with that base |
apt and pip use BuildKit cache mounts; layer order is apt (stable) -> python requirements -> user/scripts.
Framework verified (`check_framework.sh`): numpy 1.26, scipy, OpenCV headless 4.11, open3d 0.19, optuna, trimesh, openai, httpx, `kiss_icp` 1.3.0 (a
synthetic room moved 0.3 m: odometry 0.36 m), cv_bridge, tf_transformations (needed `transforms3d>=0.4.2`; the apt one breaks on numpy 1.26), Nav2, slam_toolbox,
pcl_ros, robot_localization, pointcloud_to_laserscan, foxglove_bridge, mcap storage, rviz2; `x30-rosdep --check`: all satisfied. Runs as uid 1000 `x30` with sudo.

## Proofs
**(c) colcon x30_msgs**: `x30.ps1 build x30_msgs` in the framework image with named volumes: `Finished <<< x30_msgs [0.46s]`, 12.9 s including container start
and volume creation; `ros2 interface show x30_msgs/msg/Detection3DArray` prints the message from the overlay.

**(a) Humble framework <-> stock ros:jazzy** (only change to the stock image: `apt install ros-jazzy-rmw-cyclonedds-cpp`; both on `x30_net`, both directions at once, 20 s window, python probe publishing pre-serialised CDR and receiving raw bytes).
Shipped config `cyclonedds.xml` (SPDP multicast + unicast data), best-effort sensor QoS, per direction (the two directions agree within noise):
| Topic | Rate (nominal) | Drops | Latency mean / p95 / max (ms) |
|---|---|---|---|
| PointCloud2 80000 pts x 26 B = 2.08 MB | 10.00 Hz (10) | 0 / 198 | 3.7 / 4.4 / 6.0 (Humble->Jazzy), 3.5 / 4.4 / 7.2 (Jazzy->Humble) |
| Image 640x480 rgb8 | 15.00 Hz (15) | 0 / 297 | 1.4 / 1.8 / 2.6 |
| /tf (3 transforms) | 49.99 Hz (50) | 0 / 988 | 0.4 / 1.1 / 1.9 |
| Twist | 20.00 Hz (20) | 0 / 396 | 0.5 / 1.3 / 1.8 |
Data rate 20.9 MB/s cloud + 13.9 MB/s image per direction. CPU per probe process 11-12 % of one core; container `docker stats` 12-13 %; memory 53-124 MB.
Reliable QoS: also 0 drops, cloud latency 4.7 / 5.9 / 7.4 ms, CPU 14 %. Static-peers file `cyclonedds-peers.xml`: 0 drops both QoS.
Plain multicast (`AllowMulticast=default`): about 5 % of Image samples lost with best-effort (18/296, 11/299), reliable 0, cloud 0 loss. Latency is wall clock receive time minus header stamp
inside one host, so it is a one-way estimate including Python callback delay. An early version of the probe that deserialised messages in rclpy hit 100 % CPU on Humble
and 8 Hz: that was the tool (Humble rclpy converts 2 MB arrays slowly), not DDS; consumers in Python on Humble should use raw/serialized subscriptions or numpy views for clouds.
A `terminate/"string data is not null-terminated"` rcutils message appears once at probe shutdown; harmless, not investigated further.

**(b) real-robot chain without the robot**: Noetic `rosbag play` of `2026-10-06-17-05-52.bag` (370 MB) from the ros1_bridge image against a `roscore` in a sibling container -> `parameter_bridge`
(Foxy, Cyclone, `/lidar_points`) -> `/lidar_points` in the Humble framework container, all on `x30_net`. Single pass: 177 of 178 messages received (the first message precedes
the subscriber match; bag starts at once), mean 10.01 Hz over 17.6 s, mean gap 99.9 ms, p95 124 ms, max 156 ms, 2 gaps above 150 ms. Message shape intact: `frame_id=lidar_link`,
width about 80k, point_step 26, fields x,y,z,intensity,tag,line,timestamp, 2.08 MB. Looped (35 s): 355 messages, 10.02 Hz, max gap 429 ms (the loop restart), 9 gaps above 150 ms.
Bridge CPU 4-6 %, bag player 1-2 %, memory 29-41 MB. No 192.168.x.x address was contacted. Needed fixes found on the way: Foxy's Cyclone 0.7 rejects `SocketReceiveBufferSize`
(separate `cyclonedds-foxy.xml`); `roscore`/`rosbag` must run in a Noetic-only environment (`ros1` prefix in the bridge entrypoint).

**(d) models**: `x30-vlm` (my compose, default `VLM_CTX=16384`) was already running, started from my compose by another agent, so I reused it instead of starting a second VLM (brief rule).
`/v1/models` lists `qwen3-vl`. Text prompt: 0.34 s, 21 prompt tokens, 112 tokens/s generation, answer correct. Image prompt (PIL 640x480 red square + blue circle): 323 prompt tokens,
0.35-0.52 s, 81-84 tokens/s generation, description correct. VRAM: total GPU use was 11.1 GB while vlm and one other agent's perception container ran (about 0.9 GB desktop baseline); the VLM alone is
therefore roughly 7-8 GB (estimate; per-process VRAM is not reported under WSL). Container RAM 7.2 GB. Perception: `x30-perception` built and started (GPU, port 8101 because 8100 was taken by another
agent's container): `GET /health` -> `{"status":"placeholder","cuda":true,"device":"NVIDIA GeForce RTX 3090 Ti"}`, also reachable from the framework container by name; stopped and removed afterwards.
Not tested: the real perception server through my image (code did not exist yet), the `vlm2` mapper service (model not cached, download not allowed), starting `vlm` from scratch by me.

**(e) guard**: logs one line per 10 s (docker stats, nvidia-smi, WSL used memory, vmmem). Tested: dry run and a real kill of a throw-away container `x30-infra-victim` with limit 1 GB (exited 137). Finding:
`vmmemWSL` read 22.8-23.4 GB while the VM's real use (MemTotal-MemAvailable) was 9.5 GB because of 13.9 GB page cache, so the guard uses the latter (a vmmem limit of 21 GB would kill the sim constantly). Default limit 21 GB, victims `x30-sim,x30-framework`.
`x30-guard.sh` (Bash) tested once in Git Bash dry-run only.

**Launcher**: tested `status`, `build`, `replay` and `up replay` (framework + tools + replay started; replay needed `bag_compat.py`; Foxglove port 8765 answers the WebSocket upgrade with a Foxglove-specific protocol error, i.e. the server is alive; no Foxglove client was attached), `down`-style cleanup,
`images` path (via equivalent compose command; the lock logic of `x30.ps1 images` itself was not run end to end). `x30.sh`: `status`, unknown-command path, syntax check; other subcommands not run. `docker-start` not run (Docker was up).

## Failed / limits / known issues
* Real-robot networking is documented, not tested (no Linux host, no robot, `.wslconfig` untouched). `cyclonedds-host.xml` and `compose.real.yml` are untested.
* Cross-talk: all containers on `x30_net` share domain 0; during my replay test the simulator's `/lidar_points` was mixed in (3.3 Hz). Use different `ROS_DOMAIN_ID` per concurrent run.
* Humble `ros2 bag play` cannot read the converted lab bag as is (metadata version 9); handled by `bag_compat.py` in the replay service. The dataset owner could regenerate with older metadata.
* A bind mount of `ros2_ws/src/x30_perception/perception_server` makes Docker create the host directory if it is missing.
* Perception image default base not built (see above). `two-vlm` untested. Per-process VRAM unmeasurable on WSL.
* `compose.models.yml` publishes ports on 127.0.0.1 only; LAN clients cannot reach the VLM.
* `.env` is not created (defaults work); copy `.env.example` when you need to change something.

## Next steps
Run the real perception server through the image once it exists; try `up real --host-net` on a Linux box; decide per-agent domain ids; consider a Humble-friendly metadata in `datasets/real`.
