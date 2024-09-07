# FROM ros:noetic as base
FROM ros:noetic as base

# minimal dep
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    -y wget apt-utils git openssh-server nano bc \
    python3                  \
    python3-wstool             \
    python3-pip                \
    python3-psutil             \
    python-is-python3          \
    python3-catkin-tools       \
    && rm -rf /var/lib/apt/lists/*

# nvidia 
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}all

# zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t agnoster \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting 
    
# color bash
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' ~/.bashrc

# fzf
WORKDIR /root/third_parties
RUN git clone https://github.com/junegunn/fzf.git
WORKDIR /root/third_parties/fzf
RUN /bin/zsh -c "./install"
WORKDIR /root

# -----------------------------------------
# ros  dependencies
FROM base as ros_dep

RUN apt-get update && apt-get install -y ros-noetic-cv-bridge \
                                         ros-noetic-vision-opencv\
                                         ros-noetic-std-msgs\
                                         ros-noetic-sensor-msgs\
                                         ros-noetic-geometry-msgs \
                                         ros-noetic-rospy \
                                         ros-noetic-rqt-image-view \
                                         ros-noetic-tf2 \
                                         ros-noetic-tf2-geometry-msgs \
                                         ros-noetic-tf
                                         
# ----------------------------------------
# python dependencies
FROM ros_dep as python_dep
WORKDIR /root/third_parties
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# install open3d
RUN pip3 install --ignore-installed PyYAML
RUN pip3 install open3d

# install cpu monitor and catkin build
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/alspitz/cpu_monitor.git

# catkin build
WORKDIR /root/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/$ROS_DISTRO/setup.bash; \
                    catkin_init_workspace;'

RUN echo "source /opt/ros/$ROS_DISTRO/setup.zsh" >> ~/.zshrc 
RUN echo "source /root/catkin_ws/devel/setup.zsh" >> ~/.zshrc 

#entrypoint
WORKDIR /
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/entrypoint.sh" ]
SHELL ["/bin/zsh", "-c"]
CMD ["zsh"]
