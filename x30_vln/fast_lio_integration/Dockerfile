FROM x30-livox-framework:latest
USER root
ARG FAST_LIO_REF=2fffc570a25d0df172720bac034fbdb6a13d2162
RUN git clone --recursive https://github.com/Ericsii/FAST_LIO_ROS2.git /opt/fast_lio/src/fast_lio \
    && cd /opt/fast_lio/src/fast_lio \
    && git checkout "${FAST_LIO_REF}" \
    && git submodule update --init --recursive \
    && . /opt/ros/humble/setup.sh \
    && . /opt/livox/humble/local_setup.sh \
    && cd /opt/fast_lio \
    && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
COPY launch.py /opt/fast_lio/install/share/fast_lio/launch/x30_mapping.launch.py
COPY select_lidar.py /opt/fast_lio/select_lidar.py
USER x30
