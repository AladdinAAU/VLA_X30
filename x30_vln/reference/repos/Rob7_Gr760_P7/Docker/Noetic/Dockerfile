FROM ros:noetic

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Install ROS desktop + development tools
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    build-essential \
    git \
    vim \
    nano \
    python3-pip \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init 2>/dev/null || true
RUN rosdep update

# Create catkin workspace
RUN mkdir -p /catkin_ws/src

WORKDIR /catkin_ws

# Automatically source ROS and workspace
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
    && echo "source /catkin_ws/devel/setup.bash 2>/dev/null || true" >> /root/.bashrc

CMD ["bash", "-c", "source /opt/ros/noetic/setup.bash && roscore"]

