#!/bin/bash

source /opt/ros/melodic/setup.bash
source /home/seucar/wendao/sweeper/ros_ws/devel/setup.bash

roslaunch /home/seucar/wendao/sweeper/ros_ws/src/my_image_publisher/launch/img_publisher.launch camera_id:=1 &
sleep 5

roslaunch /home/seucar/wendao/sweeper/ros_ws/src/transmiter/launch/transmiter.launch &
sleep 5

#!/usr/bin/env python3

cd /home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/
rosparam load param.yaml
python3 detection_v6.py
sleep 5
