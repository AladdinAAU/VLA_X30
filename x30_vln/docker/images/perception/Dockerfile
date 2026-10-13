# syntax=docker/dockerfile:1.7
# Open-vocabulary segmentation server (GPU). Recipe ported from the drone project (Dockerfile.perception).
# The server code is NOT baked in: it is bind-mounted from ros2_ws/src/x30_perception/perception_server at /app/perception_server
# (compose.models.yml). If that directory has no server.py, entrypoint.sh starts placeholder_server.py (health only).
# Weights and the CLIP text model are downloaded on first use into /cache (host: D:\perception-cache).
ARG BASE_IMAGE=ultralytics/ultralytics:latest
FROM ${BASE_IMAGE}

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastapi "uvicorn[standard]" python-multipart \
    && pip install "git+https://github.com/ultralytics/CLIP.git"

COPY placeholder_server.py /opt/x30/placeholder_server.py
COPY entrypoint.sh /usr/local/bin/x30-perception-entrypoint
RUN chmod +x /usr/local/bin/x30-perception-entrypoint

WORKDIR /cache
ENV YOLO_CONFIG_DIR=/cache/config \
    PERCEPTION_DIR=/app/perception_server \
    PERCEPTION_APP=server:app
EXPOSE 8100
ENTRYPOINT ["/usr/local/bin/x30-perception-entrypoint"]
