# -*- coding: UTF-8 -*-
#!/usr/bin/env python3

"""将ROS节点管理器发布的图像话题记录为视频
"""

# For computer seucar.
seucar_switch = False

import rospy
from sensor_msgs.msg import Image
import numpy as np

import os
import sys
if not seucar_switch:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

out_path = 'video_out.mp4'
target_fps = 30
frame_height = 480
frame_width = 640
video_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height), True)

def image_callback(image_data):
    cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
    video_out.write(cv_image)
    cv2.imshow("cv_image", cv_image)
    if cv2.waitKey(1) == 27:
        video_out.release()
        print("Save video.")
        cv2.destroyAllWindows()
        # 按下Esc键停止python程序
        rospy.signal_shutdown("It's over.")

if __name__ == '__main__':
    print('Waiting for node...')
    rospy.init_node("record")
    rospy.Subscriber('/image_rectified', Image, image_callback, queue_size=1, buff_size=52428800)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
