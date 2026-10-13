#!/bin/bash
# Run the mounted perception server if present, otherwise the placeholder health server.
set -e
PORT="${PERCEPTION_PORT:-8100}"
if [ -f "${PERCEPTION_DIR}/server.py" ]; then
    echo "[perception] serving ${PERCEPTION_APP} from ${PERCEPTION_DIR} on :${PORT}"
    exec uvicorn "${PERCEPTION_APP}" --app-dir "${PERCEPTION_DIR}" --host 0.0.0.0 --port "${PORT}" "$@"
fi
echo "[perception] WARNING: ${PERCEPTION_DIR}/server.py not found; starting PLACEHOLDER (GET /health only)"
exec uvicorn placeholder_server:app --app-dir /opt/x30 --host 0.0.0.0 --port "${PORT}" "$@"
