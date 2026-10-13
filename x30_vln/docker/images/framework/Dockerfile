# syntax=docker/dockerfile:1.7
# X30 VLN framework image: ROS 2 Humble + Nav2 + SLAM + python tooling. No CUDA (models live in their own containers).
# The workspace source is bind-mounted at /ws/src; /ws/build, /ws/install, /ws/log are named volumes (see docker/compose.yml).
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base-jammy

ARG ROS_DISTRO
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=x30
ARG HOST_UID=1000
ARG HOST_GID=1000
ENV ROS_DISTRO=${ROS_DISTRO} LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

# apt: keep downloaded .debs in a BuildKit cache mount so a changed package list does not re-download everything
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Layer 1: ROS packages (stable, big). Edit the list below to add ROS dependencies permanently.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates sudo nano less tmux iputils-ping net-tools iproute2 netcat-openbsd \
        python3-pip python3-dev python3-venv python3-requests python3-scipy python3-matplotlib python3-yaml python3-colcon-common-extensions python3-rosdep python3-vcstool \
        libeigen3-dev libtbb-dev libgl1 libgomp1 \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp ros-${ROS_DISTRO}-cyclonedds \
        ros-${ROS_DISTRO}-navigation2 ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-slam-toolbox \
        ros-${ROS_DISTRO}-robot-localization \
        ros-${ROS_DISTRO}-pointcloud-to-laserscan \
        ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-pcl-conversions ros-${ROS_DISTRO}-pcl-msgs \
        ros-${ROS_DISTRO}-tf2-tools ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-tf2-ros-py ros-${ROS_DISTRO}-tf2-eigen \
        ros-${ROS_DISTRO}-tf2-geometry-msgs ros-${ROS_DISTRO}-tf2-sensor-msgs ros-${ROS_DISTRO}-tf-transformations \
        ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-compressed-image-transport \
        ros-${ROS_DISTRO}-sensor-msgs-py ros-${ROS_DISTRO}-message-filters ros-${ROS_DISTRO}-vision-msgs \
        ros-${ROS_DISTRO}-robot-state-publisher ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-diagnostic-msgs \
        ros-${ROS_DISTRO}-teleop-twist-keyboard ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-foxglove-bridge \
        ros-${ROS_DISTRO}-ros2bag ros-${ROS_DISTRO}-rosbag2-storage-default-plugins ros-${ROS_DISTRO}-rosbag2-storage-mcap \
        ros-${ROS_DISTRO}-launch-testing ros-${ROS_DISTRO}-launch-testing-ament-cmake ros-${ROS_DISTRO}-ament-lint-auto \
        ros-${ROS_DISTRO}-demo-nodes-cpp ros-${ROS_DISTRO}-demo-nodes-py \
        ros-${ROS_DISTRO}-rqt-graph ros-${ROS_DISTRO}-rqt-tf-tree

# Layer 2: python deps. numpy stays below 2 because ROS cv_bridge (apt) is built against numpy 1.x.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r /tmp/requirements.txt
# open3d is optional (large); a failed install must not break the image but must be visible in the build log
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install "open3d==0.19.0" "numpy<2" || echo "WARNING: open3d did not install; continuing without it"

# Layer 3: non-root user, rosdep database, workspace dirs owned by that user (so the named volumes inherit ownership)
RUN groupadd -g ${HOST_GID} ${USERNAME} \
    && useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${USERNAME} \
    && usermod -aG video,dialout ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p /ws/src /ws/build /ws/install /ws/log /x30 /etc/x30 \
    && chown -R ${USERNAME}:${USERNAME} /ws \
    && (rosdep init || true)

COPY x30-env /usr/local/bin/x30-env
COPY entrypoint.sh /usr/local/bin/x30-entrypoint
COPY x30-build x30-rosdep x30-test /usr/local/bin/
RUN chmod +x /usr/local/bin/x30-* \
    && cp /usr/local/bin/x30-env /etc/profile.d/x30.sh \
    && echo '. /usr/local/bin/x30-env' >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
RUN rosdep update --rosdistro ${ROS_DISTRO} || echo "WARNING: rosdep update failed (offline build?); run 'x30-rosdep' later"

WORKDIR /ws
ENTRYPOINT ["/usr/local/bin/x30-entrypoint"]
CMD ["bash"]
