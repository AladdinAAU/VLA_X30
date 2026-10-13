FROM ros:humble-ros-base-jammy

# Install useful dev tools
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository universe && \
    apt update && \
    apt install -y \
    	python3-colcon-common-extensions \
    	python3-rosdep \
	    iputils-ping \
	    net-tools \
    	nano \
    	cmake \
        pip \
    	libpcl-dev \
        libeigen3-dev \
    	ros-humble-pcl-conversions \
        ros-humble-pcl-msgs \
        ros-humble-pcl-ros \
	    ros-humble-tf2-ros \
	    ros-humble-tf2-eigen \
	    ros-humble-rviz2 \
        ros-humble-teleop-twist-keyboard \
        ros-humble-bondcpp \
        ros-humble-test-msgs \
    	wget \
    && rm -rf /var/lib/apt/lists/*


# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# Copy only package manifests so rosdep can resolve dependencies
# COPY ros2_ws/src /tmp/ws/src
# WORKDIR /tmp/ws

# # Install all ROS dependencies from package.xml files
# RUN apt-get update && \
#     rosdep install --from-paths src --ignore-src -r -y\
#     --from-paths ./src \
#     --ignore-src \
#     --skip-keys="gazebo_ros gazebo_ros_pkgs"

# # Clean up (since you'll mount the real workspace at runtime)
# RUN rm -rf /tmp/ws

# Auto-source ROS
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc

# Install Livox SDK
#WORKDIR /root
#RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
#    mkdir -p Livox-SDK2/build && \
#    cd Livox-SDK2/build && \
#    cmake .. && \
#    make -j$(nproc) && \
#    make install && \
#    ldconfig

WORKDIR /root/ros2_ws

CMD ["bash"]