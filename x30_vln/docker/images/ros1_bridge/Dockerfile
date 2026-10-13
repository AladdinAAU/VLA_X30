# syntax=docker/dockerfile:1.7
# ROS 1 Noetic <-> ROS 2 bridge for the real X30 (profile "real"). Based on the official ros:foxy-ros1-bridge image as in
# Rob7_Gr760_P7. The ROS 2 side is Foxy; it talks to the Humble framework container over CycloneDDS (RTPS interop).
# The image also contains the ROS 1 tools (roscore, rosbag, rostopic) so it doubles as the Noetic side of the replay test.
FROM ros:foxy-ros1-bridge

ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-foxy-rmw-cyclonedds-cpp iputils-ping net-tools netcat-openbsd

COPY topics.default.yaml /opt/x30/topics.default.yaml
COPY bridge-entrypoint.sh /usr/local/bin/x30-bridge-entrypoint
RUN chmod +x /usr/local/bin/x30-bridge-entrypoint && mkdir -p /etc/x30

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    BRIDGE_MODE=parameter \
    BRIDGE_TOPICS=/x30/configs/ros1_bridge_topics.yaml
ENTRYPOINT ["/ros_entrypoint.sh", "/usr/local/bin/x30-bridge-entrypoint"]
CMD ["bridge"]
