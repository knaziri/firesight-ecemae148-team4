FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pyserial

RUN pip3 install git+https://github.com/LiamBindle/PyVESC.git

WORKDIR /firetruck_ws
COPY src/ src/

RUN /bin/bash -c \
    "source /opt/ros/humble/setup.bash && \
     rosdep update && \
     rosdep install --from-paths src --ignore-src -r -y && \
     colcon build --symlink-install"

RUN echo 'source /opt/ros/humble/setup.bash'       >> ~/.bashrc && \
    echo 'source /firetruck_ws/install/setup.bash' >> ~/.bashrc && \
    echo 'export ROS_DOMAIN_ID=42'                 >> ~/.bashrc

CMD ["/bin/bash"]
