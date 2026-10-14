FROM x30-framework:latest
USER root
SHELL ["/bin/bash", "-c"]
ARG BUILD_JOBS=2

COPY livox_bridge/interfaces/ros2/ /tmp/livox_humble/src/livox_ros_driver2/
COPY livox_bridge/interfaces/msg/ /tmp/livox_humble/src/livox_ros_driver2/msg/
RUN source /opt/ros/humble/setup.bash \
    && cd /tmp/livox_humble \
    && CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} colcon build --merge-install --install-base /opt/livox/humble \
       --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
    && cp /usr/local/bin/x30-env /usr/local/bin/x30-env-base

COPY livox_bridge/humble-env /usr/local/bin/x30-env
COPY livox_bridge/humble-env /etc/profile.d/x30.sh
USER x30
