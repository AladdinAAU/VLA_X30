# syntax=docker/dockerfile:1.7
# X30 simulation image: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (via ros_gz), ros2_control, and the two pinned upstream
# repositories used by the earlier prework (legged controller + X30 model). Software rendering (llvmpipe) is the default.
#
# Layout inside the image
#   /opt/x30_ext/install   upstream packages (leg_pd_controller, unitree_guide_controller, x30_description, ...) already built
#   /opt/x30_ext/src       upstream sources at the pinned commits (patched)
#   /ws                    workspace overlay: source bind-mounted at /ws/src, build/install/log in named volumes (docker/compose.sim.yml)
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base-noble

ARG ROS_DISTRO
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=x30
ARG HOST_UID=1000
ARG HOST_GID=1000
# Pinned upstream commits (same as the prework scripts/setup_external.sh)
ARG QUADRUPED_REF=5434c5810d1a7fe223bcfd04550e9d3bfdd4b458
ARG NAVIGATION_REF=3c12fabf51614695ea763f8015e51b3b3d75c822
ENV ROS_DISTRO=${ROS_DISTRO} LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Layer 1: Gazebo Harmonic + bridge, ros2_control, rendering stack (Mesa llvmpipe over EGL), python tooling.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates sudo nano less tmux procps psmisc iputils-ping net-tools iproute2 \
        python3-pip python3-dev python3-colcon-common-extensions python3-rosdep python3-vcstool \
        python3-numpy python3-scipy python3-yaml python3-opencv python3-pil python3-matplotlib python3-pytest python3-requests \
        libeigen3-dev \
        libgl1 libegl1 libglx-mesa0 libgl1-mesa-dri libegl-mesa0 libgbm1 mesa-utils mesa-utils-extra xvfb x11-utils \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-ros-gz ros-${ROS_DISTRO}-gz-ros2-control \
        ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers ros-${ROS_DISTRO}-controller-manager \
        ros-${ROS_DISTRO}-imu-sensor-broadcaster ros-${ROS_DISTRO}-joint-state-broadcaster ros-${ROS_DISTRO}-joint-state-publisher \
        ros-${ROS_DISTRO}-hardware-interface ros-${ROS_DISTRO}-controller-interface ros-${ROS_DISTRO}-backward-ros \
        ros-${ROS_DISTRO}-kdl-parser ros-${ROS_DISTRO}-pluginlib ros-${ROS_DISTRO}-realtime-tools \
        ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-robot-state-publisher ros-${ROS_DISTRO}-diagnostic-msgs \
        ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-tf2-ros-py ros-${ROS_DISTRO}-tf2-tools ros-${ROS_DISTRO}-tf2-geometry-msgs \
        ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-sensor-msgs-py ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-ros2bag ros-${ROS_DISTRO}-rosbag2-storage-default-plugins \
        ros-${ROS_DISTRO}-launch-testing ros-${ROS_DISTRO}-launch-testing-ament-cmake ros-${ROS_DISTRO}-ament-lint-auto \
        ros-${ROS_DISTRO}-teleop-twist-keyboard

# Layer 2: pure-python helpers for the lidar comparison script (reads ROS 2 bags without ROS). PEP 668: apt python, hence the flag.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages rosbags

# Layer 3: user, workspace dirs. Ubuntu 24.04 images ship a user "ubuntu" with uid 1000: remove it first.
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && groupadd -g ${HOST_GID} ${USERNAME} \
    && useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${USERNAME} \
    && usermod -aG video,dialout ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p /ws/src /ws/build /ws/install /ws/log /x30 /etc/x30 /opt/x30_ext \
    && chown -R ${USERNAME}:${USERNAME} /ws /opt/x30_ext

# Layer 4: upstream repositories at the pinned commits, controller patch applied, built into /opt/x30_ext/install.
# quadruped_navigation_ros2 is cloned for reference only (its navigation stack is replaced by the framework container).
USER ${USERNAME}
WORKDIR /opt/x30_ext
COPY --chown=${USERNAME}:${USERNAME} patches/ /opt/x30_ext/patches/
RUN mkdir -p src && cd src \
    && git clone https://github.com/legubiao/quadruped_ros2_control.git \
    && git -C quadruped_ros2_control checkout -q ${QUADRUPED_REF} \
    && git clone https://github.com/mil-as/quadruped_navigation_ros2.git \
    && git -C quadruped_navigation_ros2 checkout -q ${NAVIGATION_REF} \
    && git -C quadruped_ros2_control apply /opt/x30_ext/patches/unitree_guide_x30.patch \
    && sed -i '/<joint name="imu_joint"/,/<\/joint>/ s#xyz="0 0 0"#xyz="0.13045 -0.001 0.0149"#' \
        quadruped_ros2_control/descriptions/deep_robotics/x30_description/xacro/imu_link.xacro \
    && test "$(grep -c '0.13045' quadruped_ros2_control/descriptions/deep_robotics/x30_description/xacro/imu_link.xacro)" = 1
# (the sed moves the body IMU joint to the documented position, manual p.67: (130.45, -1, 14.9) mm; it must change exactly one line)

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && cd /opt/x30_ext \
    && colcon --log-base /tmp/x30_ext_log build --merge-install --install-base /opt/x30_ext/install --build-base /tmp/x30_ext_build \
        --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        --base-paths src/quadruped_ros2_control \
        --packages-up-to control_input_msgs controller_common leg_pd_controller unitree_guide_controller x30_description \
    && rm -rf /tmp/x30_ext_build /tmp/x30_ext_log

USER root
COPY x30-env /usr/local/bin/x30-env
COPY entrypoint.sh /usr/local/bin/x30-entrypoint
COPY x30-build x30-test /usr/local/bin/
RUN chmod +x /usr/local/bin/x30-* \
    && cp /usr/local/bin/x30-env /etc/profile.d/x30.sh \
    && echo '. /usr/local/bin/x30-env' >> /home/${USERNAME}/.bashrc \
    && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /ws
ENV RENDER=sw
ENTRYPOINT ["/usr/local/bin/x30-entrypoint"]
CMD ["bash"]
