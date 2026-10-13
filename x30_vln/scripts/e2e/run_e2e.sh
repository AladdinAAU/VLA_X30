#!/usr/bin/env bash
# Bring up the full end-to-end stack: sim (Jazzy) + framework nodes (Humble) + perception + VLM. Reproducible, idempotent.
#   bash scripts/e2e/run_e2e.sh [world]        (Git Bash on the Windows host, from anywhere)
# Containers: x30-vlm, x30-perception (compose, models profile), e2e-sim (Jazzy), e2e-fw (Humble). ROS_DOMAIN_ID=80, GZ_PARTITION=e2e.
# Logs: results/e2e/logs/*.log (inside containers /x30/results/e2e/logs). Stop everything with scripts/e2e/stop_e2e.sh.
set -u
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; P="$(cd "$ROOT" && pwd -W 2>/dev/null || pwd)"
WORLD="${1:-replica_frl_apartment_0}"; DOM=80
mkdir -p "$ROOT/results/e2e/logs"
LOGD=/x30/results/e2e/logs
COMMON="-d --network x30_net --init --shm-size 2gb -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e ROS_DOMAIN_ID=$DOM -e CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml -e X30_ROOT=/x30 -e X30_CONFIGS=/x30/configs"
fw() { docker exec e2e-fw bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=$DOM; $*"; }
fwd() { docker exec -d e2e-fw bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=$DOM; $*"; }
up() { docker ps --format '{{.Names}}' | grep -qx "$1"; }

echo "[1] models (VLM :8000, perception :8100)"
up x30-vlm && up x30-perception || bash "$ROOT/scripts/x30.sh" up models
for i in $(seq 60); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && curl -sf http://127.0.0.1:8100/health >/dev/null && break; sleep 5; done
curl -s http://127.0.0.1:8000/health; curl -s http://127.0.0.1:8100/health | head -c 80; echo

echo "[2] simulator container + launch (kinematic locomotion, $WORLD)"
up e2e-sim || docker run $COMMON --name e2e-sim -e GZ_PARTITION=e2e -e RENDER=sw -v "$P/ros2_ws/src:/ws/src" -v "$P:/x30" \
  -v x30_sim_ws_build:/ws/build -v x30_sim_ws_install:/ws/install -v x30_sim_ws_log:/ws/log -v "$P/docker/dds:/etc/x30:ro" x30-sim:latest sleep infinity >/dev/null
docker exec -d e2e-sim bash -lc ". /usr/local/bin/x30-env; export ROS_DOMAIN_ID=$DOM GZ_PARTITION=e2e; ros2 launch x30_sim sim.launch.py world:=$WORLD locomotion:=kinematic gui:=false render:=sw > $LOGD/sim.log 2>&1"

echo "[3] framework container + colcon build"
up e2e-fw || docker run $COMMON --name e2e-fw -v "$P/ros2_ws/src:/ws/src" -v "$P:/x30" -v e2e_build:/ws/build -v e2e_install:/ws/install \
  -v e2e_log:/ws/log -v "$P/docker/dds:/etc/x30:ro" x30-framework:latest sleep infinity >/dev/null
fw "cd /ws && colcon build --symlink-install --packages-skip x30_sim x30_worlds x30_hw x30_nav_bridge 2>&1 | tail -2"

echo "[4] wait for /clock + camera from the sim"
for i in $(seq 90); do fw "timeout 4 ros2 topic echo --once /gt/pose >/dev/null 2>&1" && break; sleep 5; done

echo "[5] known-map navigation (/map, map->odom identity, navigate_to_pose, /nav/follow_waypoints, /nav/next_frontier)"
fwd "ros2 launch x30_navigation known_map_nav.launch.py world:=$WORLD relay:=false > $LOGD/nav.log 2>&1"
sleep 8
echo "[6] x30_modes (cmd_vel_mux, safety_guard, mode_manager, skill_server), profile e2e (sim + raw lidar to the guard)"
fwd "ros2 launch x30_modes modes.launch.py params_file:=/x30/configs/modes/e2e.yaml use_sim_time:=true > $LOGD/modes.log 2>&1"
echo "[7] perception_client (server http://perception:8100) + vocabulary"
fwd "ros2 launch x30_perception perception.launch.py use_sim_time:=true > $LOGD/perception.log 2>&1"
echo "[8] semantic_mapper"
fwd "ros2 launch x30_semantic_map semantic_mapper.launch.py use_sim_time:=true vlm_enabled:=false > $LOGD/semmap.log 2>&1"
echo "[9] agent (reasoner_agent + instruction_follower), live VLM http://vlm:8000/v1"
fwd "ros2 launch x30_agent agent.launch.py use_sim_time:=true run_name:=e2e results_dir:=/x30/results/e2e > $LOGD/agent.log 2>&1"
sleep 12
fw "ros2 node list | sort | tr '\n' ' '; echo; ros2 service call /perception/vocabulary x30_msgs/srv/Json \"{request: '{\\\"op\\\": \\\"set\\\", \\\"labels\\\": [\\\"sofa\\\", \\\"chair\\\", \\\"tv\\\", \\\"bicycle\\\"]}'}\" 2>&1 | tail -2"
echo "stack up. Episodes: python scripts/e2e/run_episodes.py (host) ; stop: bash scripts/e2e/stop_e2e.sh"
