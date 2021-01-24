# -*- coding: UTF-8 -*-
#!/usr/bin/env python3

"""
修改注释_v1(20210115)：
    1.同时订阅单目相机图像话题和双目相机点云话题
    2.实现目标实例掩膜和特征点云的数据融合
    3.双目相机点云可视化
    4.将垃圾、行人目标水平面投影轮廓拟合为垂直矩形
    5.将植被目标水平面投影轮廓拟合为凸包多边形
    6.以特征点云对目标进行定位及参数估计
    7.垃圾目标显示选项：2D/3D边界框、掩膜、置信度
    8.植被目标显示选项：2D/3D边界框、掩膜、置信度
    9.行人目标显示选项：2D/3D边界框、掩膜、置信度
    10.分别存储垃圾、植被、行人目标的检测结果
    11.在param.yaml文件中增加若干控制选项
    12.终端显示各部分计算过程的耗时情况
    
TODO：
    1.使用新的权重文件
    2.优化数据融合机制，提高运行速度
    3.增加双目相机点云密度及连续性
"""

# For computer seucar.
seucar_switch = False

# For set /display_mode dynamically.
display_switch = False
# For set /record_mode dynamically.
record_switch = False

record_initialized = False
video_raw = None
video_result = None

# Dynamically set information to display.
realtime_control = False

show_result_r = True
box3d_mode_r = True
show_mask_r = True
show_score_r = True

show_result_v = True
box3d_mode_v = True
show_mask_v = True
show_score_v = True

show_result_p = True
box3d_mode_p = True
show_mask_p = True
show_score_p = True

show_pointcloud = True
show_output = True
show_time = True

print_stamp = True
print_xx = True

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import String

from transmiter.msg import AreaInfo
from transmiter.msg import AreasInfo
from transmiter.msg import EnvInfo
from transmiter.msg import Object
from transmiter.msg import Objects

import message_filters
import numpy as np
np.set_printoptions(suppress=True)
import time
import math

import os
import sys
if not seucar_switch:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import random
import cProfile
import pickle
import json
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

from data_custom import COLORS
from utils_custom import timer
from utils_custom.functions import SavePath

from yolact_custom import Yolact_custom
from utils_custom.augmentations_custom import FastBaseTransform_custom
from layers_custom.output_utils_custom import postprocess_custom
from data_custom import cfg_custom, set_cfg_custom

from yolact_coco import Yolact_coco
from utils_coco.augmentations_coco import FastBaseTransform_coco
from layers_coco.output_utils_coco import postprocess_coco
from data_coco import cfg_coco, set_cfg_coco

from numpy_pc2 import pointcloud2_to_xyz_array
from calib import Calib
from locat import Locat
from jet_color import Jet_Color

if seucar_switch:
    sys.path.append('/home/seucar/wendao/sweeper/region_divide')
else:
    sys.path.append('/home/lishangjie/wendao/sweeper/region_divide')
from region_divide import *

def project_pointcloud(xyz, projection_xyz_to_xyz, projection_xyz_to_uv, height, width):
    # 功能：将lidar的3D点云投影至图像平面
    # 输入：xyz <class 'numpy.ndarray'> (n, 4) 表示lidar坐标系下点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      projection_xyz_to_xyz <class 'numpy.ndarray'> (4, 4) 将lidar齐次坐标转换至camera齐次坐标的投影矩阵
    #      projection_xyz_to_uv <class 'numpy.ndarray'> (3, 4) 将lidar齐次坐标转换至uv齐次坐标的投影矩阵
    #      height <class 'int'> 图像高度
    #      width <class 'int'> 图像宽度
    # 输出：camera_xyz <class 'numpy.ndarray'> (n, 4) 表示camera坐标系下点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      camera_uv <class 'numpy.ndarray'> (n, 2) 表示像素坐标系下点云的坐标[u, v]，n为点的数量
    
    cam_xyz = projection_xyz_to_xyz.dot(xyz.T).T
    cam_uv = projection_xyz_to_uv.dot(xyz.T).T
    cam_uv = np.true_divide(cam_uv[:, :2], cam_uv[:, [-1]])
    
    camera_xyz = cam_xyz[(cam_uv[:, 0] >= 0) & (cam_uv[:, 0] < width) & (cam_uv[:, 1] >= 0) & (cam_uv[:, 1] < height)]
    camera_uv = cam_uv[(cam_uv[:, 0] >= 0) & (cam_uv[:, 0] < width) & (cam_uv[:, 1] >= 0) & (cam_uv[:, 1] < height)]
    
    return camera_xyz, camera_uv
    
def pointcloud_display(img, camera_xyz, camera_uv):
    jc = Jet_Color()
    depth = np.sqrt(np.square(camera_xyz[:, 0]) + np.square(camera_xyz[:, 1]) + np.square(camera_xyz[:, 2]))
    
    for pt in range(0, camera_uv.shape[0]):
        cv_color = jc.get_jet_color(depth[pt] * jet_color)
        cv2.circle(img, (int(camera_uv[pt][0]), int(camera_uv[pt][1])), 1, cv_color, thickness=-1)
        
    return img
    
def draw_mask(img, mask, color):
    # 功能：绘制掩膜
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      mask <class 'torch.Tensor'> torch.Size([frame_height, frame_width]) 掩膜
    #      color <class 'tuple'> 掩膜颜色
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0
    
    # 改变mask的维度 <class 'torch.Tensor'> torch.Size([480, 640, 1])
    mask = mask[:, :, None]
    
    # color_tensor <class 'torch.Tensor'> torch.Size([3])
    color_tensor = torch.Tensor(color).to(img_gpu.device.index).float() / 255.
    
    # alpha为透明度，置1则不透明
    alpha = 0.45
    
    mask_color = mask.repeat(1, 1, 3) * color_tensor * alpha
    inv_alph_mask = mask * (- alpha) + 1
    img_gpu = img_gpu * inv_alph_mask + mask_color
    
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    return img_numpy
    
def sort_polygon(polygon):
    # 功能：将多边形各顶点按逆时针方向排序
    # 输入：polygon <class 'numpy.ndarray'> (n, 1, 3) n为轮廓点数
    # 输出：polygon <class 'numpy.ndarray'> (n, 1, 3) n为轮廓点数
    
    num = polygon.shape[0]
    center_x = np.mean(polygon[:, 0, 0])
    center_z = np.mean(polygon[:, 0, 2])
    angles = []
    for i in range(num):
        dx = polygon[i, 0, 0] - center_x
        dz = polygon[i, 0, 2] - center_z
        # math.atan2(y, x)返回值范围(-pi, pi]
        angle = math.atan2(dz, dx)
        # angle范围[0, 2pi)
        if angle < 0:
            angle += 2 * 3.14159
        angles.append(angle)
    idxs = list(np.argsort(angles))
    
    return polygon[idxs]
    
def project_inside_camera(camera_xyz, projection_xyz_to_uv, height, width, cut_mode=False):
    # 功能：将camera坐标系中的点投影至图像平面
    # 输入：camera_xyz <class 'numpy.ndarray'> (n, 4) 表示camera坐标系下点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      projection_xyz_to_uv <class 'numpy.ndarray'> (3, 4) 将camera齐次坐标转换至uv齐次坐标的投影矩阵
    #      height <class 'int'> 图像高度
    #      width <class 'int'> 图像宽度
    #      cut_mode <class 'bool'> 是否滤除图像边界以外的点
    # 输出：camera_uv <class 'numpy.ndarray'> (n, 2) 表示像素坐标系下点云的坐标[u, v]，n为点的数量
    
    uv = projection_xyz_to_uv.dot(camera_xyz.T).T
    uv = np.true_divide(uv[:, :2], uv[:, [-1]])
    
    if cut_mode:
        camera_uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)]
    else:
        camera_uv = uv
        
    return camera_uv
    
def draw_box3d(img, polygon, height, projection_xyz_to_uv, color=(0, 0, 0), thickness=1):
    # 功能：绘制目标三维边界框
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      polygon <class 'numpy.ndarray'> (n, 1, 3) n为轮廓点数 代表目标底面多边形轮廓
    #      height <class 'float'> 代表目标高度
    #      projection_xyz_to_uv <class 'numpy.ndarray'> (3, 4) 将camera齐次坐标转换至uv齐次坐标的投影矩阵
    #      color <class 'tuple'> 边界框颜色
    #      thickness <class 'int'> 边界框宽度
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    img_raw = img.copy()
    frame_height = img.shape[0]
    frame_width = img.shape[1]
    
    # 将多边形各顶点按逆时针方向排序
    polygon = sort_polygon(polygon)
    
    # 计算多边形各顶点相对坐标原点的极角，范围[0, 2pi)
    num = polygon.shape[0]
    angles = []
    for j in range(num):
        # math.atan2(y, x)返回值范围(-pi, pi]
        angle = math.atan2(polygon[j, 0, 2], polygon[j, 0, 0])
        # angle范围[0, 2pi)
        if angle < 0:
            angle += 2 * 3.14159
        angles.append(angle)
    angles_array = np.array(angles)
    min_idx = np.where(angles == angles_array.min())[0][0]
    max_idx = np.where(angles == angles_array.max())[0][0]
    
    # 绘制box3d（STEP1）
    # 绘制目标侧面的竖边，由于视线遮挡，只绘制靠近坐标原点一侧的竖边
    k = max_idx
    draw_num = min_idx + num if max_idx > min_idx else min_idx
    while k <= draw_num:
        # 底面点
        x = polygon[k % num, 0, 0]
        y = polygon[k % num, 0, 1]
        z = polygon[k % num, 0, 2]
        xyz = np.array([[x, y, z, 1]])
        uv_1 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
        
        # 顶面点
        x = polygon[k % num, 0, 0]
        y = polygon[k % num, 0, 1] - height
        z = polygon[k % num, 0, 2]
        xyz = np.array([[x, y, z, 1]])
        uv_2 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
        
        pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
        pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
        cv2.line(img, pt_1, pt_2, color, thickness)
        k += 1
        
    # 绘制box3d（STEP2）
    # 绘制目标底面的横边，由于视线遮挡，只绘制靠近坐标原点一侧的横边
    k = max_idx
    draw_num = min_idx + num if max_idx > min_idx else min_idx
    while k < draw_num:
        # 第一点
        x = polygon[k % num, 0, 0]
        y = polygon[k % num, 0, 1]
        z = polygon[k % num, 0, 2]
        xyz = np.array([[x, y, z, 1]])
        uv_1 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
        
        # 第二点
        x = polygon[(k + 1) % num, 0, 0]
        y = polygon[(k + 1) % num, 0, 1]
        z = polygon[(k + 1) % num, 0, 2]
        xyz = np.array([[x, y, z, 1]])
        uv_2 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
        
        pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
        pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
        cv2.line(img, pt_1, pt_2, color, thickness)
        k += 1
        
    # 绘制box3d（STEP3）
    # 绘制目标顶面的横边，绘制靠近坐标原点一侧的横边
    k = max_idx
    draw_num = min_idx + num if max_idx > min_idx else min_idx
    while k < draw_num:
        # 第一点
        x = polygon[k % num, 0, 0]
        y = polygon[k % num, 0, 1] - height
        z = polygon[k % num, 0, 2]
        xyz = np.array([[x, y, z, 1]])
        uv_1 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
        
        # 第二点
        x = polygon[(k + 1) % num, 0, 0]
        y = polygon[(k + 1) % num, 0, 1] - height
        z = polygon[(k + 1) % num, 0, 2]
        xyz = np.array([[x, y, z, 1]])
        uv_2 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
        
        pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
        pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
        cv2.line(img, pt_1, pt_2, color, thickness)
        k += 1
    
    # 绘制box3d（STEP4）
    # 绘制目标顶面的横边，判断是否需要绘制远离坐标原点一侧的横边，如果是则绘制
    draw_flag = False
    x = polygon[max_idx, 0, 0]
    y = polygon[max_idx, 0, 1] - height
    z = polygon[max_idx, 0, 2]
    xyz = np.array([[x, y, z, 1]])
    uv_left = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
    
    x = polygon[min_idx, 0, 0]
    y = polygon[min_idx, 0, 1] - height
    z = polygon[min_idx, 0, 2]
    xyz = np.array([[x, y, z, 1]])
    uv_right = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
    
    # slope = dv / du
    slope = (uv_right[0, 1] - uv_left[0, 1]) / (uv_right[0, 0] - uv_left[0, 0])
    
    # 从靠近坐标原点一侧的点判断
    k = max_idx
    draw_num = min_idx + num if max_idx > min_idx else min_idx
    while k < draw_num:
        if (k % num) != max_idx:
            x = polygon[k % num, 0, 0]
            y = polygon[k % num, 0, 1] - height
            z = polygon[k % num, 0, 2]
            xyz = np.array([[x, y, z, 1]])
            uv = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
            du = uv[0, 0] - uv_left[0, 0]
            dv = uv[0, 1] - uv_left[0, 1]
            if dv > slope * du:
                draw_flag = True
        k += 1
        
    # 从远离坐标原点一侧的点判断
    k = min_idx
    draw_num = max_idx if max_idx > min_idx else max_idx + num
    while k < draw_num:
        if (k % num) != min_idx:
            x = polygon[k % num, 0, 0]
            y = polygon[k % num, 0, 1] - height
            z = polygon[k % num, 0, 2]
            xyz = np.array([[x, y, z, 1]])
            uv = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
            du = uv[0, 0] - uv_left[0, 0]
            dv = uv[0, 1] - uv_left[0, 1]
            if dv < slope * du:
                draw_flag = True
        k += 1
        
    if draw_flag:
        k = min_idx
        draw_num = max_idx if max_idx > min_idx else max_idx + num
        while k < draw_num:
            # 第一点
            x = polygon[k % num, 0, 0]
            y = polygon[k % num, 0, 1] - height
            z = polygon[k % num, 0, 2]
            xyz = np.array([[x, y, z, 1]])
            uv_1 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
            
            # 第二点
            x = polygon[(k + 1) % num, 0, 0]
            y = polygon[(k + 1) % num, 0, 1] - height
            z = polygon[(k + 1) % num, 0, 2]
            xyz = np.array([[x, y, z, 1]])
            uv_2 = project_inside_camera(xyz, projection_xyz_to_uv, frame_height, frame_width)
            
            pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
            pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
            cv2.line(img, pt_1, pt_2, color, thickness)
            k += 1
            
    return img
    
def draw_box2d(img, box, color, font_thickness):
    # 功能：绘制目标二维边界框
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      box <class 'numpy.ndarray'> (4,) [x1, x2, y1, y2] 图像左上角坐标x1 y1，右下角坐标x2 y2
    #      color <class 'tuple'> 边界框颜色
    #      thickness <class 'int'> 边界框宽度
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    x1, y1, x2, y2 = box[:]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, font_thickness)
    
    return img
    
def target_display(img, masks, classes, scores, boxes, polygons, heights, positions, box3d_mode=True, show_mask=True, show_score=True):
    num_target = classes.shape[0]
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    font_thickness = 1
    
    # 三维边界框模式
    if box3d_mode:
        # 计算目标距离
        target_dd = []
        for i in range(num_target):
            # 排除positions为(0, 0)的目标
            if positions[i, 0] or positions[i, 1]:
                dd = positions[i, 0] ** 2 + positions[i, 1] ** 2
            else:
                dd = float('inf')
            target_dd.append(dd)
        target_idxs = list(np.argsort(target_dd))
        
        # 按距离从远到近的顺序显示目标信息
        for i in reversed(target_idxs):
            # 排除positions为(0, 0)的目标
            if positions[i, 0] or positions[i, 1]:
                # 绘制掩膜及边界框
                color = COLORS[i]
                if show_mask:
                    img = draw_mask(img, masks[i], color)
                img = draw_box3d(img, polygons[i], heights[i], calib.projection, color, font_thickness)
                
                # 在三维边界框远处上方顶点处，显示目标信息
                polygon = polygons[i]
                num = polygon.shape[0]
                max_idx = 0
                dd_max = 0
                for j in range(num):
                    dd = polygon[j, 0, 0] ** 2 + polygon[j, 0, 2] ** 2
                    if dd > dd_max:
                        dd_max = dd
                        max_idx = j
                        
                x = polygon[max_idx, 0, 0]
                y = polygon[max_idx, 0, 1] - heights[i]
                z = polygon[max_idx, 0, 2]
                xyz = np.array([[x, y, z, 1]])
                uv = project_inside_camera(xyz, calib.projection, img.shape[0], img.shape[1])
                
                u, v = int(uv[0, 0]), int(uv[0, 1])
                text_str = '%s: %.2f' % (classes[i], scores[i]) if show_score else classes[i]
                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
                
                # 图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度
                cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
        return img
        
    # 二维边界框模式
    else:
        # 按置信度从低到高的顺序显示目标信息
        for i in reversed(range(num_target)):
            # 绘制掩膜及边界框
            color = COLORS[i]
            if show_mask:
                img = draw_mask(img, masks[i], color)
            img = draw_box2d(img, boxes[i], color, font_thickness)
            
            # 在二维边界框左上方顶点处，显示目标信息
            box = boxes[i]
            u, v = int(box[0]), int(box[1])
            text_str = '%s: %.2f' % (classes[i], scores[i]) if show_score else classes[i]
            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
            
            # 图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度
            cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
        return img
        
def output_display(img):
    global region_output
    # format格式化函数
    # {:.0f} 不带小数，{:.2f} 保留两位小数，{:>3} 右对齐且宽度为3，{:<3} 左对齐且宽度为3
    for i in range(8):
        out_str = "ID:{:.0f} R:{:.0f} V:{:.0f} P:{:.0f} w:{:>3} s:{:>6} v:{:>6} x:{:>6} z:{:>6} dx:{:>6} dz:{:>6}".format(
                    region_output[i, 3], region_output[i, 0], region_output[i, 1], region_output[i, 2], 
                    int(region_output[i, 4]), round(region_output[i, 5], 4), round(region_output[i, 6], 4), 
                    round(region_output[i, 7], 2), round(region_output[i, 8], 2), round(region_output[i, 9], 2), round(region_output[i, 10], 2))
        cv2.putText(img, out_str, (15, 40 + 12 * i), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    return img
    
def detection(img):
    with torch.no_grad():
        # 目标检测（网络一）
        if modal_custom:
            frame = torch.from_numpy(img).cuda().float()
            batch_custom = FastBaseTransform_custom()(frame.unsqueeze(0))
            preds_custom = net_custom(batch_custom)
            
        # 目标检测（网络二）
        if modal_coco:
            frame = torch.from_numpy(img).cuda().float()
            batch_coco = FastBaseTransform_coco()(frame.unsqueeze(0))
            preds_coco = net_coco(batch_coco)
            
        # 保留的检测类别
        remain_list = []
        rubbish_items_1 = ['branch', 'leaves', 'metal_bottle', 'paper_box', 'peel']
        rubbish_items_2 = ['plastic_bag', 'plastic_bottle', 'solid_clod', 'solid_crumb']
        rubbish_items_3 = ['paper_scraps']
        vegetation_items = ['grass', 'shrub', 'flower']
        person_items = ['person']
        other_items = []
        
        # 针对不同目标类别设置不同top_k
        rubbish_top_k = 20
        vegetation_top_k = 4
        person_top_k = 4
        other_top_k = 2
        
        # 针对不同目标类别设置不同score_threshold
        rubbish_score_threshold_1 = 0.05
        rubbish_score_threshold_2 = 0.15
        rubbish_score_threshold_3 = 0.25
        vegetation_score_threshold = 0.40
        person_score_threshold = 0.20
        other_score_threshold = 0.30
        
        if modal_custom:
            total_top_k_custom = rubbish_top_k + vegetation_top_k + other_top_k
            min_score_threshold_custom = min([rubbish_score_threshold_1, rubbish_score_threshold_2, rubbish_score_threshold_3,
                                            vegetation_score_threshold, other_score_threshold])
            
            # 建立每个目标的掩膜masks、类别classes、置信度scores、边界框boxes的一一对应关系（网络一）
            h, w, _ = frame.shape
            with timer.env('Postprocess_custom'):
                save = cfg_custom.rescore_bbox
                cfg_custom.rescore_bbox = True
                # 检测结果
                t = postprocess_custom(preds_custom, w, h, visualize_lincomb = False, crop_masks = True,
                                                score_threshold = min_score_threshold_custom)
                cfg_custom.rescore_bbox = save
            with timer.env('Copy_custom'):
                idx = t[1].argsort(0, descending=True)[:total_top_k_custom]
                masks_custom = t[3][idx]
                classes_custom, scores_custom, boxes_custom = [x[idx].cpu().numpy() for x in t[:3]]
                
        if modal_coco:
            total_top_k_coco = person_top_k
            min_score_threshold_coco = person_score_threshold
            
            # 建立每个目标的掩膜masks、类别classes、置信度scores、边界框boxes的一一对应关系（网络二）
            h, w, _ = frame.shape
            with timer.env('Postprocess_coco'):
                save = cfg_coco.rescore_bbox
                cfg_coco.rescore_bbox = True
                # 检测结果
                t = postprocess_coco(preds_coco, w, h, visualize_lincomb = False, crop_masks = True,
                                                score_threshold = min_score_threshold_coco)
                cfg_coco.rescore_bbox = save
            with timer.env('Copy_coco'):
                idx = t[1].argsort(0, descending=True)[:total_top_k_coco]
                masks_coco = t[3][idx]
                classes_coco, scores_coco, boxes_coco = [x[idx].cpu().numpy() for x in t[:3]]
                
        # 合并检测结果
        if modal_custom and modal_coco:
            if classes_custom.shape[0] and not classes_coco.shape[0]:
                masks = masks_custom
                scores = scores_custom
                boxes = boxes_custom
                classes = []
                for i in range(classes_custom.shape[0]):
                    det_name = cfg_custom.dataset.class_names[classes_custom[i]]
                    classes.append(det_name)
                classes = np.array(classes)
            elif classes_coco.shape[0] and not classes_custom.shape[0]:
                masks = masks_coco
                scores = scores_coco
                boxes = boxes_coco
                classes = []
                for i in range(classes_coco.shape[0]):
                    det_name = cfg_coco.dataset.class_names[classes_coco[i]]
                    classes.append(det_name)
                classes = np.array(classes)
            else:
                masks = torch.cat([masks_custom, masks_coco], dim=0)
                scores = np.append(scores_custom, scores_coco, axis=0)
                boxes = np.append(boxes_custom, boxes_coco, axis=0)
                classes = []
                for i in range(classes_custom.shape[0]):
                    det_name = cfg_custom.dataset.class_names[classes_custom[i]]
                    classes.append(det_name)
                for i in range(classes_coco.shape[0]):
                    det_name = cfg_coco.dataset.class_names[classes_coco[i]]
                    classes.append(det_name)
                classes = np.array(classes)
                
        elif modal_custom and not modal_coco:
            masks = masks_custom
            scores = scores_custom
            boxes = boxes_custom
            classes = []
            for i in range(classes_custom.shape[0]):
                det_name = cfg_custom.dataset.class_names[classes_custom[i]]
                classes.append(det_name)
            classes = np.array(classes)
            
        elif modal_coco and not modal_custom:
            masks = masks_coco
            scores = scores_coco
            boxes = boxes_coco
            classes = []
            for i in range(classes_coco.shape[0]):
                det_name = cfg_coco.dataset.class_names[classes_coco[i]]
                classes.append(det_name)
            classes = np.array(classes)
            
        # 按top_k和score_threshold提取检测结果
        num_rubbish = 0
        num_vegetation = 0
        num_person = 0
        num_other = 0
        for j in range(classes.shape[0]):
            if classes[j] in rubbish_items_1:
                if num_rubbish < rubbish_top_k and scores[j] > rubbish_score_threshold_1:
                    remain_list.append(j)
                    num_rubbish += 1
            elif classes[j] in rubbish_items_2:
                if num_rubbish < rubbish_top_k and scores[j] > rubbish_score_threshold_2:
                    remain_list.append(j)
                    num_rubbish += 1
            elif classes[j] in rubbish_items_3:
                if num_rubbish < rubbish_top_k and scores[j] > rubbish_score_threshold_3:
                    remain_list.append(j)
                    num_rubbish += 1
            elif classes[j] in vegetation_items:
                if num_vegetation < vegetation_top_k and scores[j] > vegetation_score_threshold:
                    remain_list.append(j)
                    num_vegetation += 1
            elif classes[j] in person_items:
                if num_person < person_top_k and scores[j] > person_score_threshold:
                    remain_list.append(j)
                    num_person += 1
            elif classes[j] in other_items:
                if num_other < other_top_k and scores[j] > other_score_threshold:
                    remain_list.append(j)
                    num_other += 1
                    
        return masks[remain_list], classes[remain_list], scores[remain_list], boxes[remain_list]
        
def get_recthull(xs, zs):
    # 功能：以垂直包络矩形作为投影轮廓
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      zs <class 'numpy.ndarray'> (n,) 代表纵坐标
    # 输出：hull <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    
    min_x = xs.min()
    max_x = xs.max()
    min_z = zs.min()
    max_z = zs.max()
    
    p1 = np.array([[min_x, min_z]])
    p2 = np.array([[max_x, min_z]])
    p3 = np.array([[max_x, max_z]])
    p4 = np.array([[min_x, max_z]])
    
    # 水平面投影轮廓 <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    hull = np.array([p1, p2, p3, p4])
    
    return hull
    
def get_convexhull(xs, zs):
    # 功能：以凸包作为投影轮廓
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      zs <class 'numpy.ndarray'> (n,) 代表纵坐标
    # 输出：hull <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    
    xs = xs * 100
    zs = zs * 100
    xs = xs.astype(np.int)
    zs = zs.astype(np.int)
    
    pts = np.array((xs, zs)).T
    hull = cv2.convexHull(pts)
    
    # 水平面投影轮廓 <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    hull = hull / 100.0
    
    return hull
    
def fusion(camera_xyz, camera_uv, target_masks, target_classes, target_scores, target_boxes):
    global region_output
    
    # 分别存储垃圾目标、植被目标、行人目标，并进行结果后处理
    rubbish_remain_list = []
    vegetation_remain_list = []
    person_remain_list = []
    
    rubbish_items = ['branch', 'ads', 'cigarette_butt', 'firecracker', 'glass bottle',
        'leaves', 'metal_bottle', 'paper_box', 'paper_scraps', 'peel', 'plastic_bag',
        'plastic_bottle', 'solid_clod', 'solid_crumb']
    vegetation_items = ['grass', 'shrub', 'flower']
    person_items = ['person']
    
    check_k = 0
    while check_k < target_classes.shape[0]:
        if target_classes[check_k] in rubbish_items:
            rubbish_remain_list.append(check_k)
        if target_classes[check_k] in vegetation_items:
            vegetation_remain_list.append(check_k)
        if target_classes[check_k] in person_items:
            person_remain_list.append(check_k)
        check_k += 1
        
    rubbsih_num = len(rubbish_remain_list)
    vegetation_num = len(vegetation_remain_list)
    person_num = len(person_remain_list)
    
    # x_masks     <class 'torch.Tensor'>  torch.Size([N, frame_height, frame_width]) N为目标数量
    # x_classes   <class 'numpy.ndarray'> (N,) N为目标数量
    # x_scores    <class 'numpy.ndarray'> (N,) N为目标数量
    # x_boxes     <class 'numpy.ndarray'> (N, 4) N为目标数量
    # x_polygons  <class 'list'> 列表长度为N，N为目标数量，列表元素为polygon或None <class 'numpy.ndarray'> (n, 1, 3) n为轮廓点数
    # x_heights   <class 'list'> 列表长度为N，N为目标数量，列表元素为height或None <class 'float'>
    # x_positions <class 'numpy.ndarray'> (N, 2) N为目标数量 代表最近点横纵方向坐标或(0, 0)
    
    r_masks = target_masks[rubbish_remain_list, :, :]
    r_classes = target_classes[rubbish_remain_list]
    r_scores = target_scores[rubbish_remain_list]
    r_boxes = target_boxes[rubbish_remain_list, :]
    r_polygons = []
    r_heights = []
    r_positions = np.zeros((rubbsih_num, 2))
    
    v_masks = target_masks[vegetation_remain_list, :, :]
    v_classes = target_classes[vegetation_remain_list]
    v_scores = target_scores[vegetation_remain_list]
    v_boxes = target_boxes[vegetation_remain_list, :]
    v_polygons = []
    v_heights = []
    v_positions = []
    v_positions = np.zeros((vegetation_num, 2))
    
    p_masks = target_masks[person_remain_list, :, :]
    p_classes = target_classes[person_remain_list]
    p_scores = target_scores[person_remain_list]
    p_boxes = target_boxes[person_remain_list, :]
    p_polygons = []
    p_heights = []
    p_positions = np.zeros((person_num, 2))
    
    # 针对垃圾目标的处理
    if rubbsih_num > 0:
        # 在CPU上操作掩膜
        r_masks_cpu = r_masks.byte().cpu().numpy()
        
        # 为不同垃圾分配质量系数(单位g/m3)
        rubbish_weight_coefficient_list = [160, 80, 200, 200, 8000,
            80, 1050, 80, 80, 6000, 80,
            775, 15750, 4000]
        assert len(rubbish_weight_coefficient_list) == len(rubbish_items)
        
        # 遍历每个目标
        for i in range(rubbsih_num):
            target_xs = []
            target_ys = []
            target_zs = []
            
            # 提取掩膜中的特征点
            mask = r_masks_cpu[i]
            for pt in range(camera_xyz.shape[0]):
                if mask[int(camera_uv[pt][1]), int(camera_uv[pt][0])]:
                    target_xs.append(camera_xyz[pt][0])
                    target_ys.append(camera_xyz[pt][1])
                    target_zs.append(camera_xyz[pt][2])
                    
            # 如果掩膜中包含特征点云
            if len(target_xs):
                pts_xyz = np.array([target_xs, target_ys, target_zs], dtype=np.float32).T
                
                # 水平面投影轮廓 <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
                hull = get_recthull(pts_xyz[:, 0], pts_xyz[:, 2])
                b_x = hull[:, 0, 0]
                b_z = hull[:, 0, 1]
                effective_pt_num = hull.shape[0]
                
                # 三维轮廓，polygon为底面多边形坐标，height为高度
                min_y = pts_xyz[:, 1].min()
                max_y = pts_xyz[:, 1].max()
                height = max_y - min_y
                polygon = np.zeros((effective_pt_num, 1, 3))
                for pt in range(effective_pt_num):
                    polygon[pt, 0, 0] = hull[pt, 0, 0]
                    polygon[pt, 0, 1] = max_y
                    polygon[pt, 0, 2] = hull[pt, 0, 1]
                r_polygons.append(polygon)
                r_heights.append(height)
                
                # 最近点横纵方向坐标
                position_x = float('inf')
                position_z = float('inf')
                for pt in range(effective_pt_num):
                    if hull[pt, 0, 0] ** 2 + hull[pt, 0, 1] ** 2 < position_x ** 2 + position_z ** 2:
                        position_x = hull[pt, 0, 0]
                        position_z = hull[pt, 0, 1]
                r_positions[i, 0] = position_x
                r_positions[i, 1] = position_z
                    
                # 计算横纵方向尺度
                min_x = pts_xyz[:, 0].min()
                max_x = pts_xyz[:, 0].max()
                min_z = pts_xyz[:, 2].min()
                max_z = pts_xyz[:, 2].max()
                length_x = max_x - min_x
                length_z = max_z - min_z
                
                # 计算多边形轮廓面积
                s_sum = 0
                for b_pt in range(effective_pt_num):
                    s_sum += b_x[b_pt] * b_z[(b_pt + 1) % effective_pt_num] - b_z[b_pt] * b_x[(b_pt + 1) % effective_pt_num]
                target_area = abs(s_sum) / 2
                
                # 计算体积和质量
                min_y = pts_xyz[:, 1].min()
                max_y = pts_xyz[:, 1].max()
                target_height = abs(max_y - min_y)
                target_volume = target_area * target_height
                w_coef = rubbish_weight_coefficient_list[rubbish_items.index(r_classes[i])]
                target_weight = w_coef * target_volume
                
                # 利用轮廓点进行目标定位，更新各区域内最大单体目标（水平面投影面积最大）的面积、体积、位置及尺寸
                for b_pt in range(effective_pt_num):
                    region = locat.findregion(b_x[b_pt], b_z[b_pt])
                    if region > 0 and target_area > region_output[region - 1, 5]:
                        region_output[region - 1, 5] = target_area
                        region_output[region - 1, 6] = target_volume
                        region_output[region - 1, 7] = min_x
                        region_output[region - 1, 8] = min_z
                        region_output[region - 1, 9] = length_x
                        region_output[region - 1, 10] = length_z
                        
                # 利用轮廓点进行目标定位，将质量分配到各区域
                for b_pt in range(effective_pt_num):
                    region = locat.findregion(b_x[b_pt], b_z[b_pt])
                    if region > 0:
                        region_output[region - 1, 4] += target_weight / effective_pt_num
                        
            else:
                r_polygons.append(None)
                r_heights.append(None)
                
        # 界定污染等级
        for region_i in range(8):
            if region_output[region_i, 4] > 0 and region_output[region_i, 4] <= 50:
                region_output[region_i, 0] = 1
            elif region_output[region_i, 4] > 50 and region_output[region_i, 4] <= 100:
                region_output[region_i, 0] = 2
            elif region_output[region_i, 4] > 100 and region_output[region_i, 4] <= 150:
                region_output[region_i, 0] = 3
            elif region_output[region_i, 4] > 150 and region_output[region_i, 4] <= 200:
                region_output[region_i, 0] = 4
            elif region_output[region_i, 4] > 200 and region_output[region_i, 4] <= 250:
                region_output[region_i, 0] = 5
            elif region_output[region_i, 4] > 250 and region_output[region_i, 4] <= 300:
                region_output[region_i, 0] = 6
            elif region_output[region_i, 4] > 300:
                region_output[region_i, 0] = 7
                
        # 限制各区域输出
        for region_i in range(8):
            if region_output[region_i, 4] > 350:
                region_output[region_i, 4] = 350
        region_output = np.around(region_output, decimals=6)
        
    # 针对植被目标的处理
    if vegetation_num > 0:
        # 在CPU上操作掩膜
        v_masks_cpu = v_masks.byte().cpu().numpy()
        
        # 为不同植被分配优先级
        vegetation_priority_list = [0, 1, 2]
        assert len(vegetation_priority_list) == len(vegetation_items)
        
        # 遍历每个目标
        for i in range(vegetation_num):
            target_xs = []
            target_ys = []
            target_zs = []
            
            # 提取掩膜中的特征点
            mask = v_masks_cpu[i]
            for pt in range(camera_xyz.shape[0]):
                if mask[int(camera_uv[pt][1]), int(camera_uv[pt][0])]:
                    target_xs.append(camera_xyz[pt][0])
                    target_ys.append(camera_xyz[pt][1])
                    target_zs.append(camera_xyz[pt][2])
                    
            # 如果掩膜中包含特征点云
            if len(target_xs):
                pts_xyz = np.array([target_xs, target_ys, target_zs], dtype=np.float32).T
                
                # 水平面投影轮廓 <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
                hull = get_convexhull(pts_xyz[:, 0], pts_xyz[:, 2])
                b_x = hull[:, 0, 0]
                b_z = hull[:, 0, 1]
                effective_pt_num = hull.shape[0]
                
                # 三维轮廓，polygon为底面多边形坐标，height为高度
                min_y = pts_xyz[:, 1].min()
                max_y = pts_xyz[:, 1].max()
                height = max_y - min_y
                polygon = np.zeros((effective_pt_num, 1, 3))
                for pt in range(effective_pt_num):
                    polygon[pt, 0, 0] = hull[pt, 0, 0]
                    polygon[pt, 0, 1] = max_y
                    polygon[pt, 0, 2] = hull[pt, 0, 1]
                v_polygons.append(polygon)
                v_heights.append(height)
                
                # 最近点横纵方向坐标
                position_x = float('inf')
                position_z = float('inf')
                for pt in range(effective_pt_num):
                    if hull[pt, 0, 0] ** 2 + hull[pt, 0, 1] ** 2 < position_x ** 2 + position_z ** 2:
                        position_x = hull[pt, 0, 0]
                        position_z = hull[pt, 0, 1]
                v_positions[i, 0] = position_x
                v_positions[i, 1] = position_z
                
                # 计算优先级
                target_priority = vegetation_priority_list[vegetation_items.index(v_classes[i])]
                
                # 利用轮廓点进行目标定位，更新各区域内植被的优先级
                for b_pt in range(effective_pt_num):
                    region = locat.findregion(b_x[b_pt], b_z[b_pt])
                    if region > 0 and target_priority > region_output[region - 1, 1]:
                        region_output[region - 1, 1] = target_priority
                        
            else:
                v_polygons.append(None)
                v_heights.append(None)
                
    # 针对行人目标的处理
    if person_num > 0:
        # 在CPU上操作掩膜
        p_masks_cpu = p_masks.byte().cpu().numpy()
        
        # 遍历每个目标
        for i in range(person_num):
            target_xs = []
            target_ys = []
            target_zs = []
            
            # 提取掩膜中的特征点
            mask = p_masks_cpu[i]
            for pt in range(camera_xyz.shape[0]):
                if mask[int(camera_uv[pt][1]), int(camera_uv[pt][0])]:
                    target_xs.append(camera_xyz[pt][0])
                    target_ys.append(camera_xyz[pt][1])
                    target_zs.append(camera_xyz[pt][2])
                    
            # 如果掩膜中包含特征点云
            if len(target_xs):
                pts_xyz = np.array([target_xs, target_ys, target_zs], dtype=np.float32).T
                
                # 水平面投影轮廓 <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
                hull = get_recthull(pts_xyz[:, 0], pts_xyz[:, 2])
                b_x = hull[:, 0, 0]
                b_z = hull[:, 0, 1]
                effective_pt_num = hull.shape[0]
                
                # 三维轮廓，polygon为底面多边形坐标，height为高度
                min_y = pts_xyz[:, 1].min()
                max_y = pts_xyz[:, 1].max()
                height = max_y - min_y
                polygon = np.zeros((effective_pt_num, 1, 3))
                for pt in range(effective_pt_num):
                    polygon[pt, 0, 0] = hull[pt, 0, 0]
                    polygon[pt, 0, 1] = max_y
                    polygon[pt, 0, 2] = hull[pt, 0, 1]
                p_polygons.append(polygon)
                p_heights.append(height)
                
                # 最近点横纵方向坐标
                position_x = float('inf')
                position_z = float('inf')
                for pt in range(effective_pt_num):
                    if hull[pt, 0, 0] ** 2 + hull[pt, 0, 1] ** 2 < position_x ** 2 + position_z ** 2:
                        position_x = hull[pt, 0, 0]
                        position_z = hull[pt, 0, 1]
                p_positions[i, 0] = position_x
                p_positions[i, 1] = position_z
                
                # 利用轮廓点进行目标定位，更新各区域内行人标志位
                for b_pt in range(effective_pt_num):
                    region = locat.findregion(b_x[b_pt], b_z[b_pt])
                    if region > 0:
                        region_output[region - 1, 2] = 1
                        
            else:
                p_polygons.append(None)
                p_heights.append(None)
                
    return r_masks, r_classes, r_scores, r_boxes, r_polygons, r_heights, r_positions,\
     v_masks, v_classes, v_scores, v_boxes, v_polygons, v_heights, v_positions,\
      p_masks, p_classes, p_scores, p_boxes, p_polygons, p_heights, p_positions
    
def convert(region_output):
    # 输出变量areasinfo_msg
    # 分别添加8个区域的污染等级、植被类型、行人标志、区域ID、最大垃圾体积、区域内总质量
    areasinfo_msg = AreasInfo()
    for region_i in range(8):
        areainfo_msg = AreaInfo()
        areainfo_msg.rubbish_grade = int(region_output[region_i, 0])
        areainfo_msg.vegetation_type = int(region_output[region_i, 1])
        areainfo_msg.has_person = bool(region_output[region_i, 2])
        areainfo_msg.area_id = int(region_output[region_i, 3])
        areainfo_msg.total_weight = float(region_output[region_i, 4])
        areainfo_msg.max_volumn = float(region_output[region_i, 6])
        areasinfo_msg.infos.append(areainfo_msg)
    pub_areasinfo.publish(areasinfo_msg)
    
    # 输出变量objects_msg
    # 若某个区域内存在最大单体目标，且该目标的两个方向的尺寸至少有一个超出极限值，则添加该目标的尺寸和位置信息
    objects_msg = Objects()
    objects_msg_flag = False
    for region_i in range(8):
        if region_output[region_i, 7] >= max_width or region_output[region_i, 8] >= max_width:
            object_msg = Object()
            object_msg.y = - region_output[region_i, 7] + coordinate_offset_y
            object_msg.x = region_output[region_i, 8] + coordinate_offset_x
            object_msg.h = region_output[region_i, 9]
            object_msg.w = region_output[region_i, 10]
            objects_msg.objects.append(object_msg)
            objects_msg_flag = True
    if objects_msg_flag:
        pub_objects.publish(objects_msg)
        
    # 输出变量envinfo_msg
    # 分别添加天气状况、道路类型
    envinfo_msg = EnvInfo()
    envinfo_msg.weather = 0
    envinfo_msg.road_type = 0
    pub_envinfo.publish(envinfo_msg)
    
def image_callback(image):
    global image_stamp_list
    global cv_image_list
    
    image_stamp = image.header.stamp.secs + 0.000000001 * image.header.stamp.nsecs
    cv_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    
    if len(image_stamp_list) < 30:
        image_stamp_list.append(image_stamp)
        cv_image_list.append(cv_image)
    else:
        image_stamp_list.pop(0)
        cv_image_list.pop(0)
        image_stamp_list.append(image_stamp)
        cv_image_list.append(cv_image)
        
def zed_callback(pointcloud):
    time_start_all = time.time()
    
    global region_output
    global image_stamp_list
    global cv_image_list
    
    # 图像与点云消息同步
    lidar_stamp = pointcloud.header.stamp.secs + 0.000000001 * pointcloud.header.stamp.nsecs
    et_m = float('inf')
    id_stamp = 0
    for t in range(len(image_stamp_list)):
        et = abs(image_stamp_list[t] - lidar_stamp)
        if et < et_m:
            et_m = et
            id_stamp = t
    cv_image = cv_image_list[id_stamp]
    frame_height = cv_image.shape[0]
    frame_width = cv_image.shape[1]
    
    time_now = time.time()
    
    global display_switch
    display_switch = rospy.get_param("~display_mode")
    global record_switch
    record_switch = rospy.get_param("~record_mode")
    global record_initialized
    global video_raw
    global video_result
    
    global realtime_control
    realtime_control = rospy.get_param("~realtime_control")
    
    global show_result_r, box3d_mode_r, show_mask_r, show_score_r
    global show_result_v, box3d_mode_v, show_mask_v, show_score_v
    global show_result_p, box3d_mode_p, show_mask_p, show_score_p
    global show_pointcloud, show_output, show_time
    global print_stamp, print_xx
    
    if realtime_control:
        show_result_r = rospy.get_param("~show_result_r")
        box3d_mode_r = rospy.get_param("~box3d_mode_r")
        show_mask_r = rospy.get_param("~show_mask_r")
        show_score_r = rospy.get_param("~show_score_r")
        
        show_result_v = rospy.get_param("~show_result_v")
        box3d_mode_v = rospy.get_param("~box3d_mode_v")
        show_mask_v = rospy.get_param("~show_mask_v")
        show_score_v = rospy.get_param("~show_score_v")
        
        show_result_p = rospy.get_param("~show_result_p")
        box3d_mode_p = rospy.get_param("~box3d_mode_p")
        show_mask_p = rospy.get_param("~show_mask_p")
        show_score_p = rospy.get_param("~show_score_p")
        
        show_pointcloud = rospy.get_param("~show_pointcloud")
        show_output = rospy.get_param("~show_output")
        show_time = rospy.get_param("~show_time")
        print_stamp = rospy.get_param("~print_stamp")
        print_xx = rospy.get_param("~print_xx")
        
    time_rosparam = time.time() - time_now
    
    if record_switch and not record_initialized:
        video_raw = cv2.VideoWriter(path_raw, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height), True)
        video_result = cv2.VideoWriter(path_result, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height), True)
        record_initialized = True
        print("Start recording.")
        
    if not display_switch:
        cv2.destroyAllWindows()
        
    if not record_switch and record_initialized:
        video_raw.release()
        video_result.release()
        record_initialized = False
        print("Save video.")
        
    # 载入图像
    current_image = cv_image.copy()
    
    # 目标检测
    time_now = time.time()
    if modal_custom or modal_coco:
        target_masks, target_classes, target_scores, target_boxes = detection(current_image)
    time_detection = time.time() - time_now
    
    # 载入点云
    time_now = time.time()
    # xyz <class 'numpy.ndarray'> (n, 4) 表示lidar坐标系下点云的齐次坐标[x, y, z, 1]，n为点的数量
    xyz = pointcloud2_to_xyz_array(pointcloud, remove_nans=True)
    
    if limit_mode:
        alpha = 90 - 0.5 * field_of_view
        k = math.tan(alpha * math.pi / 180.0)
        xyz = xyz[np.logical_and((xyz[:, 0] > k * xyz[:, 1]), (xyz[:, 0] > -k * xyz[:, 1]))]
    if clip_mode:
        xyz = xyz[np.logical_and((xyz[:, 0] ** 2 + xyz[:, 1] ** 2 > min_distance ** 2), (xyz[:, 0] ** 2 + xyz[:, 1] ** 2 < max_distance ** 2))]
        xyz = xyz[np.logical_and((xyz[:, 2] > view_lower_limit - sensor_height), (xyz[:, 2] < view_higher_limit - sensor_height))]
        
    # 将lidar的3D点云投影至图像平面
    # camera_xyz <class 'numpy.ndarray'> (n, 4) 表示camera坐标系下点云的齐次坐标[x, y, z, 1]，n为点的数量
    # camera_uv <class 'numpy.ndarray'> (n, 2) 表示像素坐标系下点云的坐标[u, v]，n为点的数量
    camera_xyz, camera_uv = project_pointcloud(xyz, calib.lidar_to_cam, calib.lidar_to_img, frame_height, frame_width)
    time_projection = time.time() - time_now
    
    # 更新region_output，初始化区域ID
    region_output = np.zeros((8, 11))
    for region_i in range(8):
        region_output[region_i, 3] = region_i + 1
        
    # 数据融合与结果后处理
    time_now = time.time()
    if modal_custom or modal_coco:
        r_masks, r_classes, r_scores, r_boxes, r_polygons, r_heights, r_positions,\
         v_masks, v_classes, v_scores, v_boxes, v_polygons, v_heights, v_positions,\
          p_masks, p_classes, p_scores, p_boxes, p_polygons, p_heights, p_positions = fusion(
                    camera_xyz, camera_uv, target_masks, target_classes, target_scores, target_boxes)
    time_fusion = time.time() - time_now
    
    # 发布检测结果话题
    if modal_custom or modal_coco:
        convert(region_output)
        
    # 修改图像
    time_now = time.time()
    if display_switch or record_switch:
        # 添加点云
        if show_pointcloud:
            current_image = pointcloud_display(current_image, camera_xyz, camera_uv)
        # 添加目标检测结果
        if modal_custom or modal_coco:
            # 添加垃圾目标检测结果
            if r_classes.shape[0] > 0 and show_result_r:
                current_image = target_display(current_image, r_masks, r_classes, r_scores, r_boxes, r_polygons, r_heights, r_positions,
                 box3d_mode_r, show_mask_r, show_score_r)
            # 添加植被目标检测结果
            if v_classes.shape[0] > 0 and show_result_v:
                current_image = target_display(current_image, v_masks, v_classes, v_scores, v_boxes, v_polygons, v_heights, v_positions,
                 box3d_mode_v, show_mask_v, show_score_v)
            # 添加行人目标检测结果
            if p_classes.shape[0] > 0 and show_result_p:
                current_image = target_display(current_image, p_masks, p_classes, p_scores, p_boxes, p_polygons, p_heights, p_positions,
                 box3d_mode_p, show_mask_p, show_score_p)
        # 添加输出结果
        if show_output:
            current_image = output_display(current_image)
        # 添加系统时间
        if show_time:
            cv2.putText(current_image, str(time.time()), (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
    if display_switch:
        cv2.namedWindow("raw_image", cv2.WINDOW_NORMAL)
        cv2.imshow("raw_image", cv_image)
        cv2.namedWindow("result_image", cv2.WINDOW_NORMAL)
        cv2.imshow("result_image", current_image)
        if cv2.waitKey(1) == 27:
            if record_switch and record_initialized:
                video_raw.release()
                video_result.release()
                print("Save video.")
            cv2.destroyAllWindows()
            # 按下Esc键停止python程序
            rospy.signal_shutdown("It's over.")
            
    if record_switch and record_initialized:
        video_raw.write(cv_image)
        video_result.write(current_image)
    time_display = time.time() - time_now
    time_all = time.time() - time_start_all
    
    if print_xx:
        print('region_output')
        # format格式化函数
        # {:.0f} 不带小数，{:.2f} 保留两位小数，{:>3} 右对齐且宽度为3，{:<3} 左对齐且宽度为3
        for i in range(8):
            print("ID:{:.0f} R:{:.0f} V:{:.0f} P:{:.0f} w:{:>3} s:{:>6} v:{:>6} x:{:>6} z:{:>6} dx:{:>6} dz:{:>6}".format(
                region_output[i, 3], region_output[i, 0], region_output[i, 1], region_output[i, 2], 
                int(region_output[i, 4]), round(region_output[i, 5], 4), round(region_output[i, 6], 4), 
                round(region_output[i, 7], 2), round(region_output[i, 8], 2), round(region_output[i, 9], 2), round(region_output[i, 10], 2)))
        print()
        
    if print_stamp:
        print("Input pointcloud size:   ", xyz.shape[0])
        print("Process pointcloud size: ", camera_xyz.shape[0])
        print()
        print("time cost of rosparam:   ", round(time_rosparam, 5))
        print("time cost of detection:  ", round(time_detection, 5))
        print("time cost of projection: ", round(time_projection, 5))
        print("time cost of fusion:     ", round(time_fusion, 5))
        print("time cost of display:    ", round(time_display, 5))
        print("time cost of all:        ", round(time_all, 5))
        print("------------------------------------------------------------------------------")
        print()
        
if __name__ == '__main__':
    # 初始化detection节点
    rospy.init_node("detection")
    modal_custom = rospy.get_param("~modal_custom")
    modal_coco = rospy.get_param("~modal_coco")
    
    if modal_custom or modal_coco:
        # CUDA加速模式
        cuda_mode = True
        if cuda_mode:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            
        # YOLACT网络一（用于检测路面垃圾、植被）
        if modal_custom:
            if seucar_switch:
                trained_model = '/home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_20201106_1.pth'
            else:
                trained_model = '/home/lishangjie/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_20201106_1.pth'
            model_path = SavePath.from_str(trained_model)
            yolact_config = model_path.model_name + '_config_custom'
            set_cfg_custom(yolact_config)
            # 加载网络模型
            print('Loading the custom model...')
            net_custom = Yolact_custom()
            net_custom.load_weights(trained_model)
            net_custom.eval()
            if cuda_mode:
                net_custom = net_custom.cuda()
            net_custom.detect.use_fast_nms = True
            net_custom.detect.use_cross_class_nms = False
            cfg_custom.mask_proto_debug = False
            print('  Done.\n')
            
        # YOLACT网络二（用于检测行人）
        if modal_coco:
            if seucar_switch:
                trained_model = '/home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_54_800000.pth'
            else:
                trained_model = '/home/lishangjie/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_54_800000.pth'
            model_path = SavePath.from_str(trained_model)
            yolact_config = model_path.model_name + '_config_coco'
            set_cfg_coco(yolact_config)
            # 加载网络模型
            print('Loading the coco model...')
            net_coco = Yolact_coco()
            net_coco.load_weights(trained_model)
            net_coco.eval()
            if cuda_mode:
                net_coco = net_coco.cuda()
            net_coco.detect.use_fast_nms = True
            net_coco.detect.use_cross_class_nms = False
            cfg_coco.mask_proto_debug = False
            print('  Done.\n')
            
    # 设置帧率
    target_fps_seucar = rospy.get_param("~target_fps_seucar")
    target_fps_test = rospy.get_param("~target_fps_test")
    
    if seucar_switch:
        target_fps = target_fps_seucar
        path_raw = '/home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/video_raw.mp4'
        path_result = '/home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/video_result.mp4'
    else:
        target_fps = target_fps_test
        path_raw = '/home/lishangjie/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/video_raw.mp4'
        path_result = '/home/lishangjie/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/video_result.mp4'
        
    # 设置区域划分参数
    locat = Locat()
    l1 = rospy.get_param("~region_l1")
    l2 = rospy.get_param("~region_l2")
    l3 = rospy.get_param("~region_l3")
    l4 = rospy.get_param("~region_l4")
    l5 = rospy.get_param("~region_l5")
    locat.loadlocat(l1, l2, l3, l4, l5)
    print()
    
    max_width = rospy.get_param("~max_width")
    coordinate_offset_x = rospy.get_param("~coordinate_offset_x")
    coordinate_offset_y = rospy.get_param("~coordinate_offset_y")
    
    # 设置相机参数
    CameraT = RegionDivide()
    if seucar_switch:
        if(not CameraT.loadCameraInfo("/home/seucar/wendao/sweeper/region_divide/right.yaml")):
            print("loadCameraInfo failed!")
    else:
        if(not CameraT.loadCameraInfo("/home/lishangjie/wendao/sweeper/region_divide/right.yaml")):
            print("loadCameraInfo failed!")
    CameraT.setRegionParams(l1, l2, l3, l4, l5)
    print()
    
    # 获取订阅话题名、发布话题名及其他参数
    sub_topic_image = rospy.get_param("~sub_topic_image")
    sub_topic_zed = rospy.get_param("~sub_topic_zed")
    
    pub_topic_areasinfo = rospy.get_param("~pub_topic_areasinfo")
    pub_topic_objects = rospy.get_param("~pub_topic_objects")
    pub_topic_envinfo = rospy.get_param("~pub_topic_envinfo")
    
    calib = Calib()
    file_path = rospy.get_param("~calibration_file_path")
    calib.loadcalib(file_path)
    print()
    
    limit_mode = rospy.get_param("~limit_mode")
    field_of_view = rospy.get_param("~field_of_view")
    
    clip_mode = rospy.get_param("~clip_mode")
    sensor_height = rospy.get_param("~sensor_height")
    view_higher_limit = rospy.get_param("~view_higher_limit")
    view_lower_limit = rospy.get_param("~view_lower_limit")
    min_distance = rospy.get_param("~min_distance")
    max_distance = rospy.get_param("~max_distance")
    
    jet_color = rospy.get_param("~jet_color")
    
    # region_output  <class 'numpy.ndarray'> (8, 11)
    # 第1维度，一个元素对应一个区域的信息
    # 第2维度，第1个元素为污染等级(0, 1, 2, 3, 4, 5, 6, 7)
    # 第2维度，第2个元素为植被类型(0无, 1草, 2灌木, 3花)
    # 第2维度，第3个元素为行人标志(0无, 1有)
    # 第2维度，第4个元素为区域ID(1, 2, 3, 4, 5, 6, 7, 8)
    # 第2维度，第5个元素为区域内垃圾总质量(单位g)
    # 第2维度，第6个元素为区域内最大单体垃圾的面积(单位m2)
    # 第2维度，第7个元素为区域内最大单体垃圾的体积(单位m3)
    # 第2维度，第8个元素为区域内最大单体垃圾的左前点x坐标(单位m)
    # 第2维度，第9个元素为区域内最大单体垃圾的左前点z坐标(单位m)
    # 第2维度，第10个元素为区域内最大单体垃圾的x方向长度(单位m)
    # 第2维度，第11个元素为区域内最大单体垃圾的z方向长度(单位m)
    # 最大单体的定义：水平面投影面积最大
    region_output = np.zeros((8, 11))
    
    # 发布消息队列设为1，订阅消息队列设为1，并保证订阅消息缓冲区足够大
    # 这样可以实现每次订阅最新的节点消息，避免因队列消息拥挤而导致的延迟
    pub_areasinfo = rospy.Publisher(pub_topic_areasinfo, AreasInfo, queue_size=1)
    pub_objects = rospy.Publisher(pub_topic_objects, Objects, queue_size=1)
    pub_envinfo = rospy.Publisher(pub_topic_envinfo, EnvInfo, queue_size=1)
    
    print('Waiting for node...\n')
    image_stamp_list = []
    cv_image_list = []
    rospy.Subscriber(sub_topic_image, Image, image_callback, queue_size=1, buff_size=52428800)
    while len(image_stamp_list) < 30:
        time.sleep(1)
        
    rospy.Subscriber(sub_topic_zed, PointCloud2, zed_callback, queue_size=1, buff_size=52428800)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()

