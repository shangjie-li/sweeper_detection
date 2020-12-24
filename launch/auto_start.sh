#!/bin/bash

source /opt/ros/melodic/setup.bash
source ~/wendao/sweeper/ros_ws/devel/setup.bash

roslaunch ~/wendao/sweeper/ros_ws/src/my_image_publisher/launch/img_publisher.launch camera_id:=1 &
sleep 5

roslaunch ~/wendao/sweeper/ros_ws/src/transmiter/launch/transmiter.launch &
sleep 5

#!/usr/bin/env python3

cd ~/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/
rosparam load param.yaml
python3 detection_dual.py
sleep 5
