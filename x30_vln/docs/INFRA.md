# INFRA: operator manual for the X30 docker stack

Everything runs in Docker (Docker Desktop, WSL2 backend, RTX 3090 Ti 24 GB, WSL capped at 24 GB, swap off). The launcher
`scripts\x30.ps1` (PowerShell 5.1; `scripts/x30.sh` is the Bash twin) prints the docker command behind every subcommand and never prunes anything.
Measured numbers and what was not tested are in `docs/reports/infra.md`.

## 1. Containers, networks, volumes

| Container (`x30-<service>`) | Image | Profiles | Notes |
|---|---|---|---|
| `framework` | `x30-framework` (Humble, Nav2, SLAM Toolbox, python tools) | sim real replay | idle dev container (`sleep infinity`); you `exec`/`shell` into it and launch nodes |
| `tools` | same image | sim real replay | foxglove_bridge, `ws://localhost:8765` |
| `replay` | same image | replay | `ros2 bag play $REPLAY_BAG $REPLAY_ARGS` (default `--loop --clock`) |
| `ros1_bridge` | `x30-ros1-bridge` (Foxy + Noetic) | real | parameter bridge; topic list from `configs/ros1_bridge_topics.yaml` |
| `sim` | `x30-sim` (Jazzy, Gazebo) | sim | owned by the SIM agent: `docker/compose.sim.yml`, `docker/images/sim/` |
| `vlm` | llama.cpp `server-cuda` | models sim real | OpenAI API `127.0.0.1:8000`, alias `qwen3-vl` |
| `vlm2` | same | two-vlm | small mapper VLM on `127.0.0.1:8001` (first start downloads it; not cached yet) |
| `perception` | `x30-perception` | models sim real | `127.0.0.1:8100`, `GET /health` |

* Network: user-defined bridge `x30_net`. Service names are DNS names (`http://perception:8100`, `http://vlm:8000/v1`).
  Ports are published on `127.0.0.1` only.
* Volumes: `x30_ws_build`, `x30_ws_install`, `x30_ws_log` (framework colcon output, named volumes because NTFS bind mounts are slow);
  the sim has its own `x30_sim_ws_*`. Bind mounts: `ros2_ws/src` -> `/ws/src`, whole project -> `/x30`, `docker/dds` -> `/etc/x30` (ro).
  Model caches: `VLM_MODELS_DIR` (D:/vlm-models) -> `/models`, `PERCEPTION_CACHE_DIR` (D:/perception-cache) -> `/cache`.
* Configuration: copy `.env.example` to `.env` (all keys optional; defaults work). Compose is assembled from
  `compose.yml` + `compose.models.yml` + `compose.sim.yml` (if present) + `compose.real.yml` (only with `-HostNet`).
  Compose `include:` cannot skip a missing file, therefore the launcher builds the `-f` list. Raw equivalent from `docker/`:
  `docker compose --env-file ../.env -f compose.yml -f compose.models.yml -f compose.sim.yml --profile sim up -d`.
* All ROS 2 containers: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, `ROS_DOMAIN_ID` from `.env`,
  `CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml` (ros1_bridge: `cyclonedds-foxy.xml`).

## 2. Commands (`powershell -ExecutionPolicy Bypass -File scripts\x30.ps1 <cmd>`)

| Command | What it does |
|---|---|
| `docker-start` | if the engine does not answer: `docker-clean-start.ps1` (stale-socket workaround); otherwise says it is up |
| `up sim\|real\|replay\|models [-TwoVlm] [-HostNet] [-Build]` | `docker compose --profile ... up -d` |
| `down [profile]` | stop+remove the containers of the profile (default: everything incl. the models). Volumes stay. Think before running it while others use `vlm` |
| `status` | x30 containers, `docker stats`, GPU memory, WSL VM memory |
| `shell <svc>` / `exec <svc> <cmd>` | `docker exec` into `x30-<svc>` (login shell, ROS and overlay sourced) |
| `build [pkg...]` | `colcon build --symlink-install` of `/ws` in a throw-away framework container (named volumes). No args: all except `$X30_SKIP_PKGS` (`x30_sim`) |
| `test [pkg...]` | `colcon test` + result summary |
| `rosdep [--check\|--dry-run]` | `x30-rosdep`: rosdep for the package.xml files in `/ws/src`. Installs land in the throw-away container only: put what you need into `docker/images/framework/Dockerfile` |
| `images [framework\|perception\|ros1_bridge]` | `compose build` under the shared build lock (`.locks\docker-build`, waits up to 40 min) |
| `logs <svc> [-Tail n]` | follow logs |
| `guard [run\|start\|stop]` | memory/GPU guard, section 5 |
| `replay <bagdir>` | profile `replay` with `REPLAY_BAG` set to the directory (must be inside the project) |

Typical sessions:
```
x30.ps1 docker-start                       # only if Docker is down
x30.ps1 images                             # first time (about 5 min framework+bridge; perception uses the drone image as base, see report)
x30.ps1 build                              # colcon build of ros2_ws in the framework volumes
x30.ps1 up models                          # vlm + perception
x30.ps1 up replay                          # lab bag + framework + Foxglove bridge
x30.ps1 shell framework                    # ros2 topic hz /lidar_points ...
x30.ps1 down replay
```
Foxglove: open Foxglove Studio on Windows, "Open connection", `Foxglove WebSocket`, `ws://localhost:8765`.
Replaying the converted lab bag works because `docker/scripts/bag_compat.py` rewrites its metadata (version 9, written by the `rosbags` converter)
into a Humble-readable temporary copy; Humble's `ros2 bag play` aborts on the original (`yaml-cpp: bad conversion`).

`docker/scripts/` holds proofs you can re-run: `proof_dds.sh [best_effort|reliable] [s] [dds file]` (Humble<->Jazzy),
`proof_bridge.sh [bag] [once|loop]` (Noetic bag -> bridge -> Humble), `proof_vlm.py`, `check_framework.sh`, `dds_probe.py`, `count_topic.py`.

## 3. DDS design

CycloneDDS everywhere. `docker/dds/cyclonedds.xml`: `AllowMulticast=spdp` (discovery by multicast, which works inside one Docker bridge; data
by unicast), `MaxMessageSize=65500B`, `SocketReceiveBufferSize min=2MB`, `WhcHigh=1MB`.
Why not plain multicast: with `default` the best-effort Image stream lost about 5 % of samples; `spdp` and static peers lost none.
Files: `cyclonedds-peers.xml` (no multicast, peers = service names; fallback), `cyclonedds-foxy.xml` (Cyclone 0.7 in the ros1_bridge image
rejects `SocketReceiveBufferSize`), `cyclonedds-host.xml` (host network, untested).
Limits to know: the WSL VM has `net.core.rmem_max` 4 MB, so `min` must stay below 8 MB (a 10 MB request makes node creation fail with
"failed to increase socket receive buffer"). Containers of different agents on `x30_net` with the same `ROS_DOMAIN_ID` see each other
(observed: the replay `/lidar_points` was mixed with the simulator's). Give concurrent experiments different `ROS_DOMAIN_ID`s in `.env` or per `docker exec -e`.

Adding a ROS 2 service in another compose file: join `x30_net`, set `RMW_IMPLEMENTATION`, `ROS_DOMAIN_ID`, `CYCLONEDDS_URI` as in `compose.yml`,
mount `./dds:/etc/x30:ro`, and install `ros-<distro>-rmw-cyclonedds-cpp` in its image (the stock `ros:jazzy` image does not contain it).

## 4. Real robot networking (ROS 1 master on the robot)

For live `/lidar_points` with the robot's `/tf` and `/tf_static`, see [LIVOX_LIVE.md](LIVOX_LIVE.md).
It uses the ROB7 bridge architecture with a dedicated Compose overlay and ROS 2 consumer domain;
the robot supplies the transforms for viewing and recording.

ROS 1 is a callback protocol: after contacting the master, the master and each publisher open new TCP connections back to the address
a node advertises (`ROS_IP`) on random ports. Docker Desktop's NAT hides the container behind the VM, so the robot cannot call back: topics
are listed but no data flows. (This is why the bag test runs on `x30_net` with a Noetic container instead of the robot.)

Options, best first:
1. **Linux workstation with native Docker Engine** (the Rob7 setup): `x30.sh up real --host-net` adds `compose.real.yml`
   (`network_mode: host` for ros1_bridge, framework, tools; models stay on `x30_net` and are reached on `127.0.0.1:8000/8100`).
   Set in `.env`: `ROS_MASTER_URI=http://<robot master>:11311`, `ROS_IP=<this machine's address on the robot network>`,
   `X30_DDS_FILE=cyclonedds-host.xml` (and a `cyclonedds-foxy` copy for the bridge, see the comment in `cyclonedds-host.xml`). Not tested here.
2. **Windows 11 + WSL2 mirrored networking + Docker Engine inside the WSL distro** (not Docker Desktop's engine). Facts from the docs:
   mirrored mode needs Windows 11 22H2+ and WSL 2.0.4+, mirrors the Windows interfaces into Linux (WSL gets the Windows IP, LAN can reach WSL,
   multicast supported); `--network host` in an engine running inside the distro then really is the Windows address. Exact steps (not executed by the infra agent):
   1. Edit `%UserProfile%\.wslconfig`, keep the current `memory=24GB`, `swap=0`, `processors=24` and add `networkingMode=mirrored` under `[wsl2]`.
   2. PowerShell (admin): `Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow` (or a narrower
      `New-NetFirewallHyperVRule` for TCP from the robot subnet) so the robot can connect to the bridge.
   3. `wsl --shutdown` (stops ALL containers and the Docker Desktop engine), restart WSL.
   4. Install docker-ce in the Ubuntu distro, run the stack from there with `x30.sh up real --host-net`.
   5. `scripts\check_real_network.ps1 -MasterHost <ip>` (for you to run; it only pings and does one TCP connect to port 11311).
   Risks: it changes networking for the whole WSL VM and for Docker Desktop (a Docker Desktop 4.40-4.41 report: image pulls fail with mirrored
   mode plus a system proxy); the Docker Desktop daemon and a second docker-ce daemon must not be mixed in one shell (use contexts); VPNs/firewall
   rules can interfere; a wrong `.wslconfig` can break Docker startup (back it up; the file was lowered on 2026-10-08 after a memory runaway, do not raise it).
3. Docker Desktop's own "Enable host networking" (4.34+, Settings > Resources > Network) is an L4 proxy: Docker documents that containers cannot
   bind the host's addresses. Forum reports say it does not combine with mirrored mode as hoped. Do not rely on it for ROS 1 callbacks.

Do not point any test at 192.168.x.x. The reference values (Rob7: master 192.168.1.106, workstation 192.168.1.131) are only examples in `.env.example`.
The bridge topic list is a configs file in the same format as Rob7's `topics.yaml` (default in the image: `/lidar_points`, `/imu/data`, `/cmd_vel`;
parameter-bridge topics are bidirectional, so avoid `/tf` unless you accept ROS 2 transforms flowing back to ROS 1). `BRIDGE_MODE=dynamic` runs the dynamic bridge instead.

## 5. Guard

`x30.ps1 guard` (foreground) or `guard start|stop`: every 10 s one line in `results\infra\guard-<date>.log` with `docker stats`, `nvidia-smi` and WSL memory.
The limit metric is the WSL VM's `MemTotal - MemAvailable` (page cache excluded). The Windows process `vmmemWSL` is also logged but it counts the
file cache: it read 22.8 GB while the real use was 9.5 GB, so a vmmem-based limit would fire all the time. Above `GUARD_LIMIT_GB` (21) it runs
`docker kill` on the largest container of `GUARD_KILL` (`x30-sim,x30-framework`) and logs it; it never removes anything.

## 6. Troubleshooting

* **Docker will not start / `unix://...sock: file cannot be accessed`**: stale sockets; `x30.ps1 docker-start` (it refuses while Docker Desktop is still running: quit it first).
* **Memory creeps up** (Gazebo leak): run the guard; `docker stats`; `wsl -d docker-desktop -e cat /proc/meminfo`. Restart the sim container. Page cache from big files is harmless and reclaimable.
* **Build lock**: `mkdir .locks\docker-build` before any image build; remove it when done (`rmdir`). `x30.ps1 images` does this. A lock older than 90 min may be removed.
* **colcon "build directory ... was created with a different path / install permissions"**: the Humble framework and Jazzy sim must not share volumes (they do not). To start clean:
  `docker volume rm x30_ws_build x30_ws_install x30_ws_log` (only these, only when no container uses them).
* **GPU not visible in a container**: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`; check Docker Desktop's GPU support and `nvidia-smi` on Windows. The compose files use `deploy.resources.reservations.devices`.
* **`ros2 topic list` shows topics of other people's containers**: same domain on `x30_net`; change `ROS_DOMAIN_ID`.
* **Node creation fails "failed to increase socket receive buffer size"**: `SocketReceiveBufferSize min` is above what the VM allows; keep 2MB.
* **`ros2 bag play` "yaml-cpp: bad conversion"**: bag metadata is newer than Humble; use `x30.ps1 replay` or `docker/scripts/bag_compat.py`.
* **perception container serves only `/health` with status "placeholder"**: `ros2_ws/src/x30_perception/perception_server/server.py` is missing. Careful: compose creates that host directory if it does not exist when the container is started.
* **vlm unhealthy for minutes**: first start with an empty cache downloads the GGUF (about 5 GB) into `VLM_MODELS_DIR`; `x30.ps1 logs vlm`.
* **`bridge: master not reachable`**: wrong `ROS_MASTER_URI`, robot off, or NAT (section 4).
* **Windows paths with spaces**: compose gets native paths from the launcher; in Git Bash set `MSYS_NO_PATHCONV=1` for raw `docker exec` commands.
