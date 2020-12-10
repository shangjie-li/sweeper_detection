#coding=utf-8

import cv2

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from region_divide_test import RegionDivide

g_regionDivide = RegionDivide()
g_cv_bridge = CvBridge()

IMG_WINDOW_NAME = "image"


def image_callback(rosImage):
    try:
        frame = g_cv_bridge.imgmsg_to_cv2(rosImage, "bgr8")
    except CvBridgeError as e:
        print(e)
        return 
    
    frame = g_regionDivide.draw(frame)
    
    cv2.imshow(IMG_WINDOW_NAME,frame)
    cv2.waitKey(1)


def main():

    rospy.init_node('region_divide_demo')
    
    image_sub = rospy.Subscriber("/image_rectified",Image,image_callback, queue_size=1)
        
    if(not g_regionDivide.loadCameraInfo("right.yaml")):
        return
    #设置区域划分参数L1,L2,L3,L4,L5
    #g_regionDivide.setRegionParams(6,10,10,3.5,3)
    g_regionDivide.setRegionParams(5,3,3,3,2)
    cv2.namedWindow(IMG_WINDOW_NAME,0)

    g_regionDivide.openMouseCapture(IMG_WINDOW_NAME)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()
    
    
    
    

    
