ARG BASE_IMAGE=x30-ros1-bridge:latest
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-c"]
ARG BUILD_JOBS=2

COPY livox_bridge/interfaces/ros1/ /tmp/livox_ros1/src/livox_ros_driver2/
COPY livox_bridge/interfaces/msg/ /tmp/livox_ros1/src/livox_ros_driver2/msg/
RUN unset PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH ROS_DISTRO \
    && source /opt/ros/noetic/setup.bash \
    && cd /tmp/livox_ros1 \
    && catkin_make -j${BUILD_JOBS} -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_INSTALL_PREFIX=/opt/livox/ros1 install

COPY livox_bridge/interfaces/ros2/ /tmp/livox_foxy/src/livox_ros_driver2/
COPY livox_bridge/interfaces/msg/ /tmp/livox_foxy/src/livox_ros_driver2/msg/
RUN source /opt/ros/foxy/setup.bash \
    && cd /tmp/livox_foxy \
    && CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} colcon build --merge-install --install-base /opt/livox/ros2 \
       --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

COPY ros1_bridge/bridge-entrypoint.sh /usr/local/bin/x30-bridge-entrypoint
RUN chmod +x /usr/local/bin/x30-bridge-entrypoint

# Source is prepared once outside Docker, so robot-Wi-Fi builds require no GitHub access.
COPY livox_bridge/upstream/ros1_bridge/ /tmp/livox_bridge/src/ros1_bridge/
COPY livox_bridge/build-bridge.sh /usr/local/bin/build-livox-bridge
RUN BUILD_JOBS=${BUILD_JOBS} bash /usr/local/bin/build-livox-bridge
