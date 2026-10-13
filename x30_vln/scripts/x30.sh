#!/usr/bin/env bash
# x30.sh - launcher for the X30 VLN docker stack (Linux / Git Bash / WSL). Counterpart of scripts/x30.ps1; prints every command.
# Never runs a docker prune or removes images/volumes.
#
#   docker-start                      Windows only: docker-clean-start.ps1 when the engine is down
#   up <sim|real|replay|models> [--two-vlm] [--host-net] [--build]
#   down [profile] | status | shell <svc> | exec <svc> <cmd...> | logs <svc>
#   build [pkg...] | test [pkg...] | rosdep [--check|--dry-run]
#   images [framework|perception|ros1_bridge ...]     builds under the shared build lock
#   guard [run|start|stop]            memory/GPU guard (scripts/x30-guard.sh)
#   replay <bagdir>                   bagdir inside the project
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="$ROOT/docker"
ENV_FILE="$ROOT/.env"
LOCK_DIR="$ROOT/.locks/docker-build"
CLEAN_START='C:\Users\LAB USER 1\Documents\ros2-docker-setup\docker-clean-start.ps1'
export MSYS_NO_PATHCONV=1   # Git Bash: do not rewrite container paths

# Git Bash passes POSIX paths to docker.exe; compose wants native paths there
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

show() { printf '>> %s\n' "$*"; }
run() { show "$*"; "$@"; }

TWO_VLM=0; HOST_NET=0; BUILD=0; ARGS=()
for a in "$@"; do
    case "$a" in
        --two-vlm) TWO_VLM=1 ;;
        --host-net) HOST_NET=1 ;;
        --build) BUILD=1 ;;
        *) ARGS+=("$a") ;;
    esac
done
CMD="${ARGS[0]:-help}"
REST=("${ARGS[@]:1}")

compose_args() {   # $1 = comma separated profiles
    C=(docker compose)
    [ -f "$ENV_FILE" ] && C+=(--env-file "$(native "$ENV_FILE")")
    C+=(-f "$(native "$DOCKER_DIR/compose.yml")" -f "$(native "$DOCKER_DIR/compose.models.yml")")
    [ -f "$DOCKER_DIR/compose.sim.yml" ] && C+=(-f "$(native "$DOCKER_DIR/compose.sim.yml")")
    [ "$HOST_NET" = 1 ] && C+=(-f "$(native "$DOCKER_DIR/compose.real.yml")")
    local p; IFS=',' read -ra P <<< "${1:-}"
    for p in "${P[@]}"; do [ -n "$p" ] && C+=(--profile "$p"); done
}

engine_up() { timeout 15 docker version --format '{{.Server.Version}}' >/dev/null 2>&1; }
need_engine() { engine_up || { echo "Docker engine does not answer. Run: scripts/x30.sh docker-start" >&2; exit 2; }; }
svc() { case "$1" in x30-*) echo "$1" ;; "") echo "service name required" >&2; exit 2 ;; *) echo "x30-$1" ;; esac; }

take_lock() {
    local waited=0
    mkdir -p "$ROOT/.locks"
    until mkdir "$LOCK_DIR" 2>/dev/null; do
        if [ -n "$(find "$LOCK_DIR" -maxdepth 0 -mmin +90 2>/dev/null)" ]; then echo "lock older than 90 min, removing"; rmdir "$LOCK_DIR"; continue; fi
        [ "$waited" -ge 2400 ] && { echo "build lock still held after 40 min" >&2; exit 3; }
        echo "build lock held by someone else; waiting 60 s ..."; sleep 60; waited=$((waited + 60))
    done
    show "build lock taken: $LOCK_DIR"
}
release_lock() { rmdir "$LOCK_DIR" 2>/dev/null && show "build lock released"; }

in_framework() {
    need_engine; compose_args sim
    run "${C[@]}" run --rm --no-deps -T framework "$@"
}

case "$CMD" in
    help) sed -n '2,12p' "${BASH_SOURCE[0]}" ;;
    docker-start)
        if engine_up; then echo "Docker engine is already up."; else
            run powershell.exe -ExecutionPolicy Bypass -File "$CLEAN_START"
        fi ;;
    up)
        need_engine
        PROF="${REST[0]:-}"
        case "$PROF" in sim|real|replay|models) ;; *) echo "usage: up <sim|real|replay|models>" >&2; exit 2 ;; esac
        P="$PROF"; [ "$TWO_VLM" = 1 ] && P="$P,two-vlm"
        [ "$PROF" = sim ] && [ ! -f "$DOCKER_DIR/compose.sim.yml" ] && echo "WARNING: docker/compose.sim.yml missing; starting without the simulator" >&2
        if [ "$PROF" = real ]; then
            grep -q '^ROS_MASTER_URI=' "$ENV_FILE" 2>/dev/null || echo "WARNING: ROS_MASTER_URI not set in .env (falls back to localhost)" >&2
            [ "$HOST_NET" = 1 ] || echo "WARNING: real profile without --host-net: the robot cannot call back into a NATed container (docs/INFRA.md)" >&2
        fi
        export X30_PROFILE="$PROF"
        compose_args "$P"
        UP=(up -d); [ "$BUILD" = 1 ] && UP+=(--build)
        run "${C[@]}" "${UP[@]}" ;;
    down)
        need_engine; compose_args "${REST[0]:-sim,real,replay,models,two-vlm}"
        run "${C[@]}" down ;;
    status)
        need_engine
        run docker ps -a --filter name=x30- --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        run docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'
        run nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
        command -v free >/dev/null && free -h ;;
    shell)
        need_engine; N="$(svc "${REST[0]:-}")"
        show "docker exec -it $N bash -l"; docker exec -it "$N" bash -l || docker exec -it "$N" sh ;;
    exec)
        need_engine; N="$(svc "${REST[0]:-}")"
        run docker exec -i "$N" bash -lc "${REST[*]:1}" ;;
    build)  in_framework x30-build "${REST[@]}" ;;
    test)   in_framework x30-test "${REST[@]}" ;;
    rosdep) in_framework x30-rosdep "${REST[@]}" ;;
    images)
        need_engine
        T=("${REST[@]}"); [ ${#T[@]} -eq 0 ] && T=(framework ros1_bridge perception)
        take_lock; trap release_lock EXIT
        export DOCKER_BUILDKIT=1; compose_args sim,real,models
        run "${C[@]}" build "${T[@]}" ;;
    logs)
        need_engine; run docker logs -f --tail 100 "$(svc "${REST[0]:-}")" ;;
    guard)
        SUB="${REST[0]:-run}"; G="$ROOT/scripts/x30-guard.sh"; PIDF="$ROOT/results/infra/guard.pid"
        case "$SUB" in
            run) show "$G"; bash "$G" ;;
            start) mkdir -p "$ROOT/results/infra"; show "nohup $G &"; nohup bash "$G" >/dev/null 2>&1 & echo $! > "$PIDF"; echo "guard pid $(cat "$PIDF")" ;;
            stop) [ -f "$PIDF" ] && kill "$(cat "$PIDF")" && rm -f "$PIDF" && echo "guard stopped" ;;
            *) echo "guard [run|start|stop]" >&2; exit 2 ;;
        esac ;;
    replay)
        need_engine
        [ -n "${REST[0]:-}" ] || { echo "usage: replay <bagdir inside the project>" >&2; exit 2; }
        FULL="$(cd "${REST[0]}" && pwd)"
        case "$FULL" in "$ROOT"/*) ;; *) echo "bag directory must be inside $ROOT" >&2; exit 2 ;; esac
        export REPLAY_BAG="/x30/${FULL#"$ROOT"/}"; show "REPLAY_BAG=$REPLAY_BAG"
        export X30_PROFILE=replay; compose_args replay
        run "${C[@]}" up -d --force-recreate replay ;;
    *) echo "unknown command '$CMD'" >&2; sed -n '2,12p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
