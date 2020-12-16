#!/bin/bash

source /opt/ros/kinetic/setup.bash
source ~/wendao/sweeper/ros_ws/devel/setup.bash

#!/usr/bin/env python3

cd ~/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/
rosparam load param.yaml
python3 detection_dual.py
