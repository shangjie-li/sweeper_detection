#coding=utf-8

import cv2
from math import *
import math
import numpy as np


def deg2rad(deg):
    return deg/180.0*pi
    
#假定路面为平面
#根据投影关系计算像素点对应的位置坐标
#相机坐标系(向前为Z,向右为X,向下为Y)
#像素坐标系(向右为u,向下为v)
#colCnt,rowCnt 为相机分辨率
#h 摄像头安装高度
#theta 摄像头安装向下倾斜角度(rad)
#fx,fy 相机归一化焦距
#cx,cy 相机光心坐标
def pixel2disTable(colCnt,rowCnt,h,theta,fx,fy,cx,cy):
    table = np.zeros((colCnt,rowCnt,2))
    for row in range(rowCnt):
        try:
            z = (h*fy*math.cos(theta) - h*(row-cy)*math.sin(theta)) / (fy*math.sin(theta) + (row-cy)*math.cos(theta))
        except ZeroDivisionError:
            continue
        if z > 0:
            for col in range(colCnt):
                x = (z*(col-cx)*math.cos(theta) + h*(col-cx)*math.sin(theta)) / fx
                table[col][row][0] = x
                table[col][row][1] = z
    return table

def roll2Matrix(roll):
    return np.matrix([[1,0,0],
                  [0,cos(roll),-sin(roll)],
                  [0,sin(roll), cos(roll)]])
                  
def pitch2Matrix(pitch):
    return np.matrix([[cos(pitch),0,sin(pitch)],
                  [0,1,0],
                  [-sin(pitch),0,cos(pitch)]])
                  
def yaw2Matrix(yaw):
    return np.matrix([[cos(yaw),-sin(yaw),0],
                  [sin(yaw), cos(yaw),0],
                  [0,0,1]])

#欧拉角转旋转矩阵 绕固定系的旋转
def RPY2Matrix(roll,pitch,yaw):
    Rx = roll2Matrix(roll)
    Ry = pitch2Matrix(pitch)
    Rz = yaw2Matrix(yaw)
    return Rz*Ry*Rx

#欧拉角转旋转矩阵 绕自身系的旋转
def RPY2Matrix2(roll,pitch,yaw):
    Rx = roll2Matrix(roll)
    Ry = pitch2Matrix(pitch)
    Rz = yaw2Matrix(yaw)
    return Rx*Ry*Rz

#世界坐标系到相机坐标系的旋转矩阵
#theta为相机的安装俯仰角，向下倾斜为正
def world2cameraMatrix(theta):
    #按固定坐标系旋转
    Rx = roll2Matrix(-pi/2)
    Rz = yaw2Matrix(-pi/2)
    Ry = pitch2Matrix(theta)
    return Ry*Rz*Rx

#世界坐标点转到相机坐标系
#~theta:相机的安装俯仰角
#~xyz: 世界坐标点
def world2cameraPoint(theta,xyz):
    point = np.mat([[xyz[0]],[xyz[1]],[xyz[2]]])
    R = world2cameraMatrix(theta)
    point = R.I*point
    return (point[0,0],point[1,0],point[2,0])
    
#相机坐标转像素坐标
def xyz2pixel(xyz,fx,fy,cx,cy):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    u = x/z*fx+cx
    v = y/z*fy+cy
    return(int(u),int(v))


class RegionDivide:
    def __init__(self):
        self.mask = None
        pass
    
    #设置相机参数
    def setCameraParams(self,fx,fy,cx,cy,size,h,theta):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.size = size
        self.height = h  #摄像头安装高度
        self.theta = theta#摄像头安装倾斜角,下倾为正
    
    #从文件载入相机参数
    def loadCameraInfo(self,file_name):
        print("---loading camera info---")
        fs = cv2.FileStorage(file_name, cv2.FileStorage_READ)
        if(not fs.isOpened()):
            print("No file: %s" %file_name)
            return False
            
        CameraMat = fs.getNode('CameraMat').mat()
        self.fx = int(CameraMat[0,0])
        self.fy = int(CameraMat[1,1])
        self.cx = int(CameraMat[0,2])
        self.cy = int(CameraMat[1,2])
        print("fx,fy,cx,cy",self.fx, self.fy, self.cx, self.cy)
        
        value = fs.getNode('ImageSize')
        default = []
        for i in range(value.size()):
            default.append(int(value.at(i).real()))
        
        self.size = default
        print("image size:", self.size)
        
        if(fs.getNode('Height').empty()):
            print("No 'Height' param in %s" %file_name)
            return False
            
        self.height = fs.getNode('Height').real()
        
        if(fs.getNode('Theta').empty()):
            print("No 'Theta' param in %s" %file_name)
            return False
            
        self.theta = fs.getNode('Theta').real()
        print("camera height:%.2fm\t theta:%.2fdeg" %(self.height,self.theta))
        self.theta = self.theta*math.pi/180.0
        
        fs.release()
        print("---load camera info ok ---")
        return True
        
        
    
    #设置区域划分参数
    def setRegionParams(self,L1,L2,L3,L4,L5):
        self.L1 = L1 = L1*1.0
        self.L2 = L2 = L2*1.0
        self.L3 = L3 = L3*1.0
        self.L4 = L4 = L4*1.0
        self.L5 = L5 = L5*1.0
        
        #x,y,z,区域分割点世界坐标
        A1 = [L1,L4/2,-self.height]
        A2 = [L1,-L4/2,-self.height]
        A3 = [L1+L2,L4/2,-self.height]
        A4 = [L1+L2,-L4/2,-self.height]
        A5 = [L1+L2+L3,L4/2,-self.height]
        A6 = [L1+L2+L3,-L4/2,-self.height]

        #print("world: A* ",A1,A2,A3,A4,A5,A6)
        B1 = [L1,L4/2+L5,-self.height]
        B2 = [L1,-L4/2-L5,-self.height]
        B3 = [L1+L2,L4/2+L5,-self.height]
        B4 = [L1+L2,-L4/2-L5,-self.height]
        B5 = [L1+L2+L3,L4/2+L5,-self.height]
        B6 = [L1+L2+L3,-L4/2-L5,-self.height]
        
        #print("world: A* ",A1,A2,A3,A4,A5,A6)
        
        #区域分割点相机坐标
        A1 = world2cameraPoint(self.theta,A1)
        A2 = world2cameraPoint(self.theta,A2)
        A3 = world2cameraPoint(self.theta,A3)
        A4 = world2cameraPoint(self.theta,A4)
        A5 = world2cameraPoint(self.theta,A5)
        A6 = world2cameraPoint(self.theta,A6)
        #print("camera: A* ",A1,A2,A3,A4,A5,A6)
        
        B1 = world2cameraPoint(self.theta,B1)
        B2 = world2cameraPoint(self.theta,B2)
        B3 = world2cameraPoint(self.theta,B3)
        B4 = world2cameraPoint(self.theta,B4)
        B5 = world2cameraPoint(self.theta,B5)
        B6 = world2cameraPoint(self.theta,B6)
        
        #区域分割点像素坐标
        self.a1 = xyz2pixel(A1,self.fx,self.fy,self.cx,self.cy)
        self.a2 = xyz2pixel(A2,self.fx,self.fy,self.cx,self.cy)
        self.a3 = xyz2pixel(A3,self.fx,self.fy,self.cx,self.cy)
        self.a4 = xyz2pixel(A4,self.fx,self.fy,self.cx,self.cy)
        self.a5 = xyz2pixel(A5,self.fx,self.fy,self.cx,self.cy)
        self.a6 = xyz2pixel(A6,self.fx,self.fy,self.cx,self.cy)
        
        self.m1 = ((self.a1[0]+self.a2[0])//2,self.a1[1])
        self.m2 = ((self.a5[0]+self.a6[0])//2,self.a5[1])
        
        
        self.b1 = xyz2pixel(B1,self.fx,self.fy,self.cx,self.cy)
        self.b2 = xyz2pixel(B2,self.fx,self.fy,self.cx,self.cy)
        self.b3 = xyz2pixel(B3,self.fx,self.fy,self.cx,self.cy)
        self.b4 = xyz2pixel(B4,self.fx,self.fy,self.cx,self.cy)
        self.b5 = xyz2pixel(B5,self.fx,self.fy,self.cx,self.cy)
        self.b6 = xyz2pixel(B6,self.fx,self.fy,self.cx,self.cy)
        
    
    #空间位置求区域
    def whatArea(self,x,z):
        if(x>=-self.L4/2-self.L5 and x<-self.L4/2): #1,5
            if(z>=self.L1 and z<self.L1+self.L2):
                return 1
            elif(z>=self.L1+self.L2 and z <self.L1+self.L2+self.L3):
                return 5
            else:
                return 0
        elif(x>=-self.L4/2 and x<0): #2,6
            if(z>=self.L1 and z<self.L1+self.L2):
                return 2
            elif(z>=self.L1+self.L2 and z <self.L1+self.L2+self.L3):
                return 6
            else:
                return 0
        elif(x>=0 and x<self.L4/2): #3,7
            if(z>=self.L1 and z<self.L1+self.L2):
                return 3
            elif(z>=self.L1+self.L2 and z <self.L1+self.L2+self.L3):
                return 7
            else:
                return 0
        elif(x>=self.L4/2 and x<self.L4/2+self.L5): #4,8
            if(z>=self.L1 and z<self.L1+self.L2):
                return 4
            elif(z>=self.L1+self.L2 and z <self.L1+self.L2+self.L3):
                return 8
            else:
                return 0
        else:
            return 0
        

    #区域划线
    def drawLine(self,img,color=(0,255,0),w=2):
        cv2.line(img,self.b1,self.b2,color,w)
        cv2.line(img,self.b3,self.b4,color,w)
        cv2.line(img,self.b5,self.b6,color,w)
        
        cv2.line(img,self.a1,self.a5,color,w)
        cv2.line(img,self.m1,self.m2,color,w)
        cv2.line(img,self.a2,self.a6,color,w)
        
        print(self.b1,self.b2,self.b3,self.b4,self.b5,self.b6)
        print(self.m1,self.m2)
        return img

    def draw(self,img):
        if(self.mask is None):
            size = (self.size[1],self.size[0],3)
            self.mask = np.zeros(size,dtype='uint8')
            self.drawLine(self.mask,(0,255,0))
        img = cv2.addWeighted(img,1.0,self.mask,0.2,0)
        return img

    #捕获鼠标按键
    def openMouseCapture(self,windowName):
        self._pixel2disTable = pixel2disTable(self.size[0],self.size[1],self.height,
                                              self.theta,self.fx,self.fy,self.cx,self.cy)

        cv2.setMouseCallback(windowName, self.onMouse)
        
    #地面像素点转空间距离和区域信息
    def onMouse(self,event,x,y,flags,param):
        if(event == cv2.EVENT_LBUTTONDOWN):
            loc = self._pixel2disTable[x,y] #像素求空间位置
            print("像素坐标uv(%4d,%4d)  位置坐标xz(%4.1fm,%4.1fm)" %(x,y,loc[0],loc[1]))
            print("区域： %d\n" %self.whatArea(loc[0],loc[1])) #空间位置求区域


def main(): 
    regionDivide = RegionDivide()
    #regionDivide.setCameraParams(721.5,721.5,609.5,172.85,(1242,375),1.65,0.0)
    if(not regionDivide.loadCameraInfo("1.yaml")):
        return
        
    regionDivide.setRegionParams(10,5,5,3,3)
    
    img = cv2.imread("a.png")
    if img is None:
        print("No image!")
        exit()
    
    img = regionDivide.draw(img)
    windowName = "image"
    cv2.namedWindow(windowName,0)
    regionDivide.openMouseCapture(windowName)
    cv2.imshow(windowName,img)
    
    cv2.waitKey()
    

if __name__ == '__main__':
    main()
    pass
    
