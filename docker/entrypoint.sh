#!/bin/bash

# go to workspace
cd /root/catkin_ws/src

# build workspace 
cd /root/catkin_ws
if [ ! -f build ]; then
	source /opt/ros/$ROS_DISTRO/setup.bash
	catkin build -j6
fi

cd /root/catkin_ws/src/target_detection_2
chmod +x target-loc.py

cd /root/catkin_ws/
echo "Setup completed!"

exec "$@"