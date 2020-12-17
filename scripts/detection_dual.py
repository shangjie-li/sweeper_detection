# -*- coding: UTF-8 -*-
#!/usr/bin/env python3

"""
修改注释_v1(20201217)：
    1.同时运行两个YOLACT网络，分别检测行人和其他类别的目标
    2.行人数据集为COCO，路面垃圾、植被数据集为自定义数据集
    3.用于检测行人的权重文件为yolact_resnet50_54_800000.pth
    4.用于检测其他目标的权重文件为yolact_resnet50_20201106_1.pth
    5.在param.yaml中增加部分参数，实现动态设置显示信息
    6.出于代码简洁度考虑，删除parse_args()函数
    7.为统一两个网络的目标类别名称，重新定义classes
    8.在config.py中max_size改为320，RTX2060S帧率16fps，Xavier帧率5fps
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
display_masks = True
display_bboxes = True
display_scores = True

import rospy
from sensor_msgs.msg import Image
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

if seucar_switch:
    sys.path.append('/home/seucar/wendao/sweeper/region_divide')
else:
    sys.path.append('/home/lishangjie/wendao/sweeper/region_divide')
from region_divide import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

color_cache = defaultdict(lambda: {})

def get_color(color_idx, on_gpu=None):
    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    global color_cache
    
    if on_gpu is not None and color_idx in color_cache[on_gpu]:
        return color_cache[on_gpu][color_idx]
    else:
        color = COLORS[color_idx]
        # The image might come in as RGB or BRG, depending
        if on_gpu is not None:
            color = torch.Tensor(color).to(on_gpu).float() / 255.
            color_cache[on_gpu][color_idx] = color
        return color

def result_display(img, masks, classes, scores, boxes, num_target):
    img_gpu = img / 255.0
    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if display_masks and num_target > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_target)], dim=0)
        mask_alpha = 0.45
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_target):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_target > 1:
            inv_alph_cumul = inv_alph_masks[:(num_target-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    # 按置信度从低到高的顺序显示目标信息
    for i in reversed(range(num_target)):
        color = COLORS[i]
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        # 图像左上角坐标x1 y1
        x1, y1, x2, y2 = boxes[i, :]
        
        # 显示检测结果
        if display_bboxes:
            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
        score = scores[i]
        _class = classes[i]
        text_str = '%s: %.2f' % (_class, score) if display_scores else _class
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        # 图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度
        cv2.putText(img_numpy, text_str, (x1, y1 - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img_numpy

def get_boundary(img_numpy, num_target, masks, cpt_num=10):
    # boundary_pts是四维数组，第一维代表目标的数量，第二维代表点的数量(有效点数<=cpt_num)，第三维是1，第四维存储uv(无效点坐标u=0,v=0)
    boundary_pts = np.zeros((num_target, cpt_num, 1, 2))
    for i in range(num_target):
        try:
            binary_image = masks[i, :, :].byte().cpu().numpy()
            binary_image = binary_image.astype(np.uint8)
            
            # contours是list型数据，list中每个元素是一个np.array的数组，存储轮廓点坐标uv，len(contours)即轮廓数量
            # 参数cv2.CHAIN_APPROX_NONE代表存储所有的轮廓点
            try:
                binary_image_back, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            except ValueError:
                contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            # 寻找最大轮廓
            contours_p_num = 0
            p_num_max_index = 0
            for con_i in range(len(contours)):
                p_num = contours[con_i].shape[0]
                if p_num > contours_p_num:
                    contours_p_num = p_num
                    p_num_max_index = con_i
            
            contours_max = contours[p_num_max_index]
            # cv2.drawContours的输入参数包括image, contours, contourIdx, color, thickness
            # 参数contourIdx为-1时绘制contours中的所有轮廓
            cv2.drawContours(img_numpy, contours_max, -1, (255, 255, 255), thickness=1)
            contours_array = contours_max
            
            # contours_array是三维数组，第一维代表点的数量，第二维是1，第三维存储uv
            if contours_array.shape[0] > cpt_num:
                cpt_interval = int(contours_array.shape[0] / cpt_num)
                cpts = []
                for cpt_i in range(cpt_num):
                    cpts.append(cpt_i * cpt_interval)
            else:
                cpts = list(range(contours_array.shape[0]))
            
            contours_cpts = contours_array[cpts, :, :]
            # cv2.drawContours的输入参数包括image, contours, contourIdx, color, thickness
            # 参数contourIdx为-1时绘制contours中的所有轮廓
            cv2.drawContours(img_numpy, contours_cpts, -1, (0, 0, 255), 2)
            for c_cpt in range(contours_cpts.shape[0]):
                boundary_pts[i, c_cpt, 0, :] = contours_cpts[c_cpt, 0, :]
        
        except IndexError:
            pass
    
    boundary_pts = boundary_pts.astype(np.uint16)
    return img_numpy, boundary_pts

def s_display(result_image, region_s, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1):
    offs_x = 0
    offs_y = 22
    text_str = 'S:' + str(round(region_s[0, 0].astype(np.float32), 3))
    text_xy = (max(CameraT.b3[0], 0) + offs_x, CameraT.b3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'S:' + str(round(region_s[1, 0].astype(np.float32), 3))
    text_xy = (CameraT.a3[0] + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'S:' + str(round(region_s[2, 0].astype(np.float32), 3))
    text_xy = (int((CameraT.a3[0] + CameraT.a4[0]) / 2) + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'S:' + str(round(region_s[3, 0].astype(np.float32), 3))
    text_xy = (CameraT.a4[0] + offs_x, CameraT.a4[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'S:' + str(round(region_s[4, 0].astype(np.float32), 3))
    text_xy = (max(CameraT.b5[0], 0) + offs_x, CameraT.b5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'S:' + str(round(region_s[5, 0].astype(np.float32), 3))
    text_xy = (CameraT.a5[0] + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'S:' + str(round(region_s[6, 0].astype(np.float32), 3))
    text_xy = (int((CameraT.a5[0] + CameraT.a6[0]) / 2) + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'S:' + str(round(region_s[7, 0].astype(np.float32), 3))
    text_xy = (CameraT.a6[0] + offs_x, CameraT.a6[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return result_image
    
def p_display(result_image, region_p, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1):
    offs_x = 0
    offs_y = 34
    text_str = str(round(region_p[0, 0], 1)) + ',' + str(round(region_p[0, 1], 1)) + ',' + str(round(region_p[0, 2], 2)) + ',' + str(round(region_p[0, 3], 2))
    text_xy = (max(CameraT.b3[0], 0) + offs_x, CameraT.b3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = str(round(region_p[1, 0], 1)) + ',' + str(round(region_p[1, 1], 1)) + ',' + str(round(region_p[1, 2], 2)) + ',' + str(round(region_p[1, 3], 2))
    text_xy = (CameraT.a3[0] + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = str(round(region_p[2, 0], 1)) + ',' + str(round(region_p[2, 1], 1)) + ',' + str(round(region_p[2, 2], 2)) + ',' + str(round(region_p[2, 3], 2))
    text_xy = (int((CameraT.a3[0] + CameraT.a4[0]) / 2) + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = str(round(region_p[3, 0], 1)) + ',' + str(round(region_p[3, 1], 1)) + ',' + str(round(region_p[3, 2], 2)) + ',' + str(round(region_p[3, 3], 2))
    text_xy = (CameraT.a4[0] + offs_x, CameraT.a4[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = str(round(region_p[4, 0], 1)) + ',' + str(round(region_p[4, 1], 1)) + ',' + str(round(region_p[4, 2], 2)) + ',' + str(round(region_p[4, 3], 2))
    text_xy = (max(CameraT.b5[0], 0) + offs_x, CameraT.b5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = str(round(region_p[5, 0], 1)) + ',' + str(round(region_p[5, 1], 1)) + ',' + str(round(region_p[5, 2], 2)) + ',' + str(round(region_p[5, 3], 2))
    text_xy = (CameraT.a5[0] + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = str(round(region_p[6, 0], 1)) + ',' + str(round(region_p[6, 1], 1)) + ',' + str(round(region_p[6, 2], 2)) + ',' + str(round(region_p[6, 3], 2))
    text_xy = (int((CameraT.a5[0] + CameraT.a6[0]) / 2) + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = str(round(region_p[7, 0], 1)) + ',' + str(round(region_p[7, 1], 1)) + ',' + str(round(region_p[7, 2], 2)) + ',' + str(round(region_p[7, 3], 2))
    text_xy = (CameraT.a6[0] + offs_x, CameraT.a6[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return result_image

def w_display(result_image, region_w, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1):
    offs_x = 52
    offs_y = 22
    text_str = 'W:' + str(region_w[0, 0].astype(np.int16))
    text_xy = (max(CameraT.b3[0], 0) + offs_x, CameraT.b3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'W:' + str(region_w[1, 0].astype(np.int16))
    text_xy = (CameraT.a3[0] + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'W:' + str(region_w[2, 0].astype(np.int16))
    text_xy = (int((CameraT.a3[0] + CameraT.a4[0]) / 2) + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'W:' + str(region_w[3, 0].astype(np.int16))
    text_xy = (CameraT.a4[0] + offs_x, CameraT.a4[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'W:' + str(region_w[4, 0].astype(np.int16))
    text_xy = (max(CameraT.b5[0], 0) + offs_x, CameraT.b5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'W:' + str(region_w[5, 0].astype(np.int16))
    text_xy = (CameraT.a5[0] + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'W:' + str(region_w[6, 0].astype(np.int16))
    text_xy = (int((CameraT.a5[0] + CameraT.a6[0]) / 2) + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'W:' + str(region_w[7, 0].astype(np.int16))
    text_xy = (CameraT.a6[0] + offs_x, CameraT.a6[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return result_image

def output_display(result_image, region_output, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1):
    offs_x = 0
    offs_y = 10
    text_str = 'R:' + str(region_output[0, 0].astype(np.uint8)) + ' P:' + str(region_output[0, 2].astype(np.uint8)) + ' V:' + str(region_output[0, 1].astype(np.uint8))
    text_xy = (max(CameraT.b3[0], 0) + offs_x, CameraT.b3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[1, 0].astype(np.uint8)) + ' P:' + str(region_output[1, 2].astype(np.uint8)) + ' V:' + str(region_output[1, 1].astype(np.uint8))
    text_xy = (CameraT.a3[0] + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[2, 0].astype(np.uint8)) + ' P:' + str(region_output[2, 2].astype(np.uint8)) + ' V:' + str(region_output[2, 1].astype(np.uint8))
    text_xy = (int((CameraT.a3[0] + CameraT.a4[0]) / 2) + offs_x, CameraT.a3[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[3, 0].astype(np.uint8)) + ' P:' + str(region_output[3, 2].astype(np.uint8)) + ' V:' + str(region_output[3, 1].astype(np.uint8))
    text_xy = (CameraT.a4[0] + offs_x, CameraT.a4[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[4, 0].astype(np.uint8)) + ' P:' + str(region_output[4, 2].astype(np.uint8)) + ' V:' + str(region_output[4, 1].astype(np.uint8))
    text_xy = (max(CameraT.b5[0], 0) + offs_x, CameraT.b5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[5, 0].astype(np.uint8)) + ' P:' + str(region_output[5, 2].astype(np.uint8)) + ' V:' + str(region_output[5, 1].astype(np.uint8))
    text_xy = (CameraT.a5[0] + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[6, 0].astype(np.uint8)) + ' P:' + str(region_output[6, 2].astype(np.uint8)) + ' V:' + str(region_output[6, 1].astype(np.uint8))
    text_xy = (int((CameraT.a5[0] + CameraT.a6[0]) / 2) + offs_x, CameraT.a5[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[7, 0].astype(np.uint8)) + ' P:' + str(region_output[7, 2].astype(np.uint8)) + ' V:' + str(region_output[7, 1].astype(np.uint8))
    text_xy = (CameraT.a6[0] + offs_x, CameraT.a6[1] + offs_y)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return result_image

def image_callback(image_data):
    time_start = time.time()
    global cv_image
    cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
    
    global display_switch
    display_switch = rospy.get_param("~display_mode")
    global record_switch
    record_switch = rospy.get_param("~record_mode")
    global record_initialized
    global video_raw
    global video_result
    
    global display_masks, display_bboxes, display_scores
    display_masks = rospy.get_param("~display_masks")
    display_bboxes = rospy.get_param("~display_bboxes")
    display_scores = rospy.get_param("~display_scores")
    
    if record_switch and not record_initialized:
        path_raw = 'video_raw.mp4'
        path_result = 'video_result.mp4'
        if seucar_switch:
            target_fps = 5
        else:
            target_fps = 16
        frame_height = cv_image.shape[0]
        frame_width = cv_image.shape[1]
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
    
    # region_output是8行6列数组，第i行存储第i个区域的信息
    # 第1列为污染等级(0,1,2,3,4,5,6,7)
    # 第2列为植被类型(0无,1草,2灌木,3花)
    # 第3列为行人标志(0无,1有)
    # 第4列为区域ID(1,2,3,4,5,6,7,8)
    # 第5列为区域内最大垃圾体积
    # 第6列为区域内总质量
    region_output = np.zeros((8, 6))
    
    # 区域ID初始化
    for region_i in range(8):
        region_output[region_i, 3] = region_i + 1
    
    # 初始化变量
    rubbsih_num = 0
    vegetation_num = 0
    person_num = 0
    
    with torch.no_grad():
        # 目标检测（网络一）
        frame = torch.from_numpy(cv_image).cuda().float()
        batch_custom = FastBaseTransform_custom()(frame.unsqueeze(0))
        preds_custom = net_custom(batch_custom)
        
        # 目标检测（网络二）
        frame = torch.from_numpy(cv_image).cuda().float()
        batch_coco = FastBaseTransform_coco()(frame.unsqueeze(0))
        preds_coco = net_coco(batch_coco)
        
        # 针对不同目标类别设置不同top_k
        rubbish_top_k = 20
        vegetation_top_k = 4
        person_top_k = 4
        other_top_k = 2
        
        total_top_k_custom = rubbish_top_k + vegetation_top_k + other_top_k
        total_top_k_coco = person_top_k
        
        # 针对不同目标类别设置不同score_threshold
        rubbish_score_threshold_1 = 0.05
        rubbish_score_threshold_2 = 0.15
        rubbish_score_threshold_3 = 0.25
        vegetation_score_threshold = 0.40
        person_score_threshold = 0.20
        other_score_threshold = 0.30
        
        min_score_threshold_custom = min([rubbish_score_threshold_1, rubbish_score_threshold_2, rubbish_score_threshold_3,
                                            vegetation_score_threshold, other_score_threshold])
        min_score_threshold_coco = person_score_threshold
        
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
        
        # 合并来自两个网络的检测结果
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
        
        # 保留的检测类别
        remain_list = []
        rubbish_items_1 = ['branch', 'leaves', 'metal_bottle', 'paper_box', 'peel']
        rubbish_items_2 = ['plastic_bag', 'plastic_bottle', 'solid_clod', 'solid_crumb']
        rubbish_items_3 = ['paper_scraps']
        vegetation_items = ['grass', 'shrub', 'flower']
        person_items = ['person']
        other_items = []
        
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
        num_dets_to_consider = len(remain_list)
        
        # 确保处理对象非空
        if num_dets_to_consider > 0:
            target_masks = masks[remain_list]
            target_classes = classes[remain_list]
            target_scores = scores[remain_list]
            target_boxes = boxes[remain_list]
            
            # 显示检测结果
            if display_switch or record_switch:
                result_image = result_display(frame, target_masks, target_classes, target_scores, target_boxes, num_dets_to_consider)
            else:
                result_image = frame.byte().cpu().numpy()
            
            # 分别存储垃圾目标、植被目标、行人目标，并进行结果后处理
            check_k = 0
            rubbish_remain_list = []
            vegetation_remain_list = []
            person_remain_list = []
            rubbish_items = ['branch', 'ads', 'cigarette_butt', 'firecracker', 'glass bottle',
                'leaves', 'metal_bottle', 'paper_box', 'paper_scraps', 'peel', 'plastic_bag',
                'plastic_bottle', 'solid_clod', 'solid_crumb']
            vegetation_items = ['grass', 'shrub', 'flower']
            person_items = ['person']
            while check_k < target_classes.shape[0]:
                if target_classes[check_k] in rubbish_items:
                    rubbish_remain_list.append(check_k)
                if target_classes[check_k] in vegetation_items:
                    vegetation_remain_list.append(check_k)
                if target_classes[check_k] in person_items:
                    person_remain_list.append(check_k)
                check_k += 1
            
            rubbish_masks = target_masks[rubbish_remain_list, :, :]
            rubbish_classes = target_classes[rubbish_remain_list]
            rubbish_scores = target_scores[rubbish_remain_list]
            rubbish_boxes = target_boxes[rubbish_remain_list, :]
            
            vegetation_masks = target_masks[vegetation_remain_list, :, :]
            vegetation_classes = target_classes[vegetation_remain_list]
            vegetation_scores = target_scores[vegetation_remain_list]
            vegetation_boxes = target_boxes[vegetation_remain_list, :]
            
            person_masks = target_masks[person_remain_list, :, :]
            person_classes = target_classes[person_remain_list]
            person_scores = target_scores[person_remain_list]
            person_boxes = target_boxes[person_remain_list, :]
            
            rubbsih_num = len(rubbish_remain_list)
            vegetation_num = len(vegetation_remain_list)
            person_num = len(person_remain_list)
            
            # 针对垃圾目标的处理
            if rubbsih_num > 0:
                # 在本算法中，目标的定位和相关属性估计均依赖于掩膜，若检测结果不含掩膜，则无法计算
                # 对跨区域目标的处理：
                # 设某目标面积为S、体积为V、质量为M，且该目标为超大目标，若该目标有一半的定位点在a区域，另一半的定位点在b区域
                # 则Ma=Mb=M/2，Sa=Sb=S，Va=Vb=V
                
                # 掩膜边界取点
                # rubbish_boundary_pts是四维数组，第一维代表目标的数量，第二维代表点的数量(有效点数<=cpt_num)，第三维是1，第四维存储uv(无效点坐标u=0,v=0)
                result_image, rubbish_boundary_pts = get_boundary(result_image, rubbsih_num, rubbish_masks, cpt_num=10)
                # s_polygon存储每个垃圾目标在世界坐标系中投影于地面的面积
                s_polygon = np.zeros((rubbsih_num, 1))
                
                rubbish_list = ['branch', 'ads', 'cigarette_butt', 'firecracker', 'glass bottle',
                    'leaves', 'metal_bottle', 'paper_box', 'paper_scraps', 'peel', 'plastic_bag',
                    'plastic_bottle', 'solid_clod', 'solid_crumb']
                # 为不同垃圾分配体积系数(单位m)
                rubbish_volume_coefficient_list = [0.01, 0.01, 0.01, 0.01, 0.01,
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                    0.01, 0.01, 0.01]
                # 为不同垃圾分配质量系数(单位g/m2)，当前，质量的计算不取决于体积，只取决于面积
                rubbish_weight_coefficient_list = [160, 80, 200, 200, 8000,
                    80, 1050, 80, 80, 6000, 80,
                    775, 15750, 4000]
                
                # region_s存储各区域内垃圾目标的最大单体面积(单位m2)
                region_s = np.zeros((8, 1))
                # region_v存储各区域内垃圾目标的最大单体体积(单位m3)
                region_v = np.zeros((8, 1))
                # region_p存储各区域内最大单体目标的位置及尺寸(单位m)，记录目标左前点x坐标、左前点z坐标、x方向长度、z方向长度
                region_p = np.zeros((8, 4))
                
                # region_w存储各区域内垃圾目标的质量的总和(单位g)
                region_w = np.zeros((8, 1))
                
                # 遍历每个目标
                for i in range(rubbish_boundary_pts.shape[0]):
                    effective_pt_num = 0
                    b_x, b_z = [], []
                    b_area_id = []
                    # 统计有效点数量，计算每个有效点的世界坐标和所在区域
                    for b_pt in range(rubbish_boundary_pts.shape[1]):
                        b_pt_u = rubbish_boundary_pts[i, b_pt, 0, 0]
                        b_pt_v = rubbish_boundary_pts[i, b_pt, 0, 1]
                        # 排除像素坐标无效点(u=0,v=0)
                        if b_pt_u or b_pt_v:
                            loc_b_pt = p2d_table[b_pt_u, b_pt_v]
                            # 排除世界坐标无效点(x=0,z=0)
                            if loc_b_pt[0] or loc_b_pt[1]:
                                effective_pt_num += 1
                                b_x.append(loc_b_pt[0])
                                b_z.append(loc_b_pt[1])
                                b_area_id.append(CameraT.whatArea(loc_b_pt[0], loc_b_pt[1]))
                    
                    # 如果有效点数不小于3，计算面积和质量
                    if effective_pt_num >= 3:
                        # 计算面积
                        s_sum = 0
                        for b_pt in range(effective_pt_num):
                            s_sum += b_x[b_pt]*b_z[(b_pt + 1)%effective_pt_num] - b_z[b_pt]*b_x[(b_pt + 1)%effective_pt_num]
                        s_polygon[i, 0] = abs(s_sum) / 2
                        # 计算正交矩形包络轮廓
                        min_x = min(b_x)
                        min_z = min(b_z)
                        length_x = max(b_x) - min(b_x)
                        length_z = max(b_z) - min(b_z)
                        # 更新各区域内最大单体目标的面积、体积、位置及尺寸
                        for b_pt in range(effective_pt_num):
                            # 排除区域ID无效点(ID=0)
                            if b_area_id[b_pt]:
                                if s_polygon[i, 0] > region_s[b_area_id[b_pt] - 1, 0]:
                                    # 最大单体目标的面积
                                    region_s[b_area_id[b_pt] - 1, 0] = s_polygon[i, 0]
                                    # 最大单体目标的体积
                                    v_coef = rubbish_volume_coefficient_list[rubbish_list.index(rubbish_classes[i])]
                                    region_v[b_area_id[b_pt] - 1, 0] = s_polygon[i, 0] * v_coef
                                    # 最大单体目标的位置及尺寸
                                    region_p[b_area_id[b_pt] - 1, 0] = min_x
                                    region_p[b_area_id[b_pt] - 1, 1] = min_z
                                    region_p[b_area_id[b_pt] - 1, 2] = length_x
                                    region_p[b_area_id[b_pt] - 1, 3] = length_z
                                        
                        # 计算目标质量并分配给各区域
                        for b_pt in range(effective_pt_num):
                            # 排除区域ID无效点(ID=0)
                            if b_area_id[b_pt]:
                                w_coef = rubbish_weight_coefficient_list[rubbish_list.index(rubbish_classes[i])]
                                rubbish_weight = s_polygon[i, 0] * w_coef
                                region_w[b_area_id[b_pt] - 1, 0] += rubbish_weight / effective_pt_num
                
                # 界定污染等级
                for region_i in range(8):
                    if region_w[region_i, 0] > 0 and region_w[region_i, 0] <= 50:
                        region_output[region_i, 0] = 1
                    elif region_w[region_i, 0] > 50 and region_w[region_i, 0] <= 100:
                        region_output[region_i, 0] = 2
                    elif region_w[region_i, 0] > 100 and region_w[region_i, 0] <= 150:
                        region_output[region_i, 0] = 3
                    elif region_w[region_i, 0] > 150 and region_w[region_i, 0] <= 200:
                        region_output[region_i, 0] = 4
                    elif region_w[region_i, 0] > 200 and region_w[region_i, 0] <= 250:
                        region_output[region_i, 0] = 5
                    elif region_w[region_i, 0] > 250 and region_w[region_i, 0] <= 300:
                        region_output[region_i, 0] = 6
                    elif region_w[region_i, 0] > 300:
                        region_output[region_i, 0] = 7
                        # 限制各区域输出的最大质量
                        if region_w[region_i, 0] > 350:
                            region_w[region_i, 0] = 350
                
                # 区域内最大垃圾体积
                for region_i in range(8):
                    region_output[region_i, 4] = region_v[region_i, 0]
                
                # 区域内总质量
                for region_i in range(8):
                    region_output[region_i, 5] = region_w[region_i, 0]
                
                # 显示各区域针对垃圾目标的处理结果
                if display_switch or record_switch:
                    result_image = s_display(result_image, region_s, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.4, font_thickness = 1)
                    result_image = p_display(result_image, region_p, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.4, font_thickness = 1)
                    result_image = w_display(result_image, region_w, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.4, font_thickness = 1)
            
            # 针对植被目标的处理
            if vegetation_num > 0:
                # 掩膜边界取点
                result_image, vegetation_boundary_pts = get_boundary(result_image, vegetation_num, vegetation_masks, cpt_num=20)
                # region_vegetation_type存储各区域内植被类型
                region_vegetation_type = np.zeros((8, 1))
                
                # 遍历每个目标
                for i in range(vegetation_boundary_pts.shape[0]):
                    effective_pt_num = 0
                    b_area_id = []
                    # 统计有效点数量，计算每个有效点的世界坐标和所在区域
                    for b_pt in range(vegetation_boundary_pts.shape[1]):
                        b_pt_u = vegetation_boundary_pts[i, b_pt, 0, 0]
                        b_pt_v = vegetation_boundary_pts[i, b_pt, 0, 1]
                        # 排除像素坐标无效点(u=0,v=0)
                        if b_pt_u or b_pt_v:
                            loc_b_pt = p2d_table[b_pt_u, b_pt_v]
                            # 排除世界坐标无效点(x=0,z=0)
                            if loc_b_pt[0] or loc_b_pt[1]:
                                effective_pt_num += 1
                                b_area_id.append(CameraT.whatArea(loc_b_pt[0], loc_b_pt[1]))
                    
                    # 计算植被类型优先级
                    vegetation_list = ['grass', 'shrub', 'flower']
                    v_type = vegetation_list.index(vegetation_classes[i]) + 1
                    # 更新各区域内植被类型
                    for b_pt in range(effective_pt_num):
                        # 排除区域ID无效点(ID=0)
                        if b_area_id[b_pt]:
                            if v_type > region_vegetation_type[b_area_id[b_pt] - 1, 0]:
                                region_vegetation_type[b_area_id[b_pt] - 1, 0] = v_type
                
                # 界定植被类型
                for region_i in range(8):
                    region_output[region_i, 1] = region_vegetation_type[region_i, 0]
            
            # 针对行人目标的处理
            if person_num > 0:
                # 掩膜边界取点
                result_image, person_boundary_pts = get_boundary(result_image, person_num, person_masks, cpt_num=20)
                # region_person_type存储各区域内有无行人
                region_person_type = np.zeros((8, 1))
                
                # 遍历每个目标
                for i in range(person_boundary_pts.shape[0]):
                    effective_pt = False
                    effective_area_id = 0
                    effective_pt_u = 0
                    effective_pt_v = 0
                    # 统计有效点数量，计算每个有效点的世界坐标和所在区域
                    for b_pt in range(person_boundary_pts.shape[1]):
                        b_pt_u = person_boundary_pts[i, b_pt, 0, 0]
                        b_pt_v = person_boundary_pts[i, b_pt, 0, 1]
                        # 排除像素坐标无效点(u=0,v=0)
                        if b_pt_u or b_pt_v:
                            loc_b_pt = p2d_table[b_pt_u, b_pt_v]
                            # 排除世界坐标无效点(x=0,z=0)
                            if loc_b_pt[0] or loc_b_pt[1]:
                                loc_area_id = CameraT.whatArea(loc_b_pt[0], loc_b_pt[1])
                                # 排除区域ID无效点(ID=0)，且只考虑边界点中处于图像最下方的点，即行人所在位置
                                if loc_area_id and b_pt_v > effective_pt_v:
                                    effective_pt = True
                                    effective_area_id = loc_area_id
                    
                    # 更新行人所在区域的标志位
                    if effective_pt:
                        region_person_type[effective_area_id - 1, 0] = 1
                
                # 界定有无行人
                for region_i in range(8):
                    region_output[region_i, 2] = region_person_type[region_i, 0]
            
        else:
            result_image = frame.byte().cpu().numpy()
    
    # 输出变量areasinfo_msg
    # 分别添加8个区域的污染等级、植被类型、行人标志、区域ID、最大垃圾体积、区域内总质量
    areasinfo_msg = AreasInfo()
    for region_i in range(8):
        areainfo_msg = AreaInfo()
        areainfo_msg.rubbish_grade = int(region_output[region_i, 0])
        areainfo_msg.vegetation_type = int(region_output[region_i, 1])
        areainfo_msg.has_person = bool(region_output[region_i, 2])
        areainfo_msg.area_id = int(region_output[region_i, 3])
        areainfo_msg.max_volumn = float(region_output[region_i, 4])
        areainfo_msg.total_weight = float(region_output[region_i, 5])
        areasinfo_msg.infos.append(areainfo_msg)
    pub_areasinfo.publish(areasinfo_msg)
    
    # 输出变量objects_msg
    # 若某个区域内存在最大单体目标，且该目标的两个方向的尺寸至少有一个超出极限值，则添加该目标的尺寸和位置信息
    objects_msg = Objects()
    objects_msg_flag = False
    # 确保region_p已经初始化
    if rubbsih_num > 0:
        for region_i in range(8):
            if region_p[region_i, 2] >= max_width or region_p[region_i, 3] >= max_width:
                object_msg = Object()
                object_msg.y = - region_p[region_i, 0] + coordinate_offset_y
                object_msg.x = region_p[region_i, 1] + coordinate_offset_x
                object_msg.h = region_p[region_i, 2]
                object_msg.w = region_p[region_i, 3]
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
    
    if display_switch or record_switch:
        # 显示区域划分边界线
        result_image = CameraT.drawLine(result_image, w=1)
        result_image = output_display(result_image, region_output, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.4, font_thickness = 1)
        # 显示系统时间
        cv2.putText(result_image, str(time.time()), (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    if display_switch:
        print('region_output')
        print(region_output)
        cv2.namedWindow("raw_image", cv2.WINDOW_NORMAL)
        cv2.imshow("raw_image", cv_image)
        cv2.namedWindow("result_image", cv2.WINDOW_NORMAL)
        cv2.imshow("result_image", result_image)
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
        video_result.write(result_image)
    
    time_end_all = time.time()
    print("totally time cost:", time_end_all - time_start)

if __name__ == '__main__':
    # CUDA加速模式
    cuda_mode = True
    if cuda_mode:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    # YOLACT网络一（用于检测路面垃圾、植被）
    if seucar_switch:
        trained_model = '/home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_20201106_1.pth'
    else:
        trained_model = '/home/lishangjie/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_20201106_1.pth'
    model_path = SavePath.from_str(trained_model)
    yolact_config = model_path.model_name + '_config_custom'
    print('Config not specified. Parsed %s from the file name.\n' % yolact_config)
    set_cfg_custom(yolact_config)
    # 加载网络模型
    print('Loading the first model...')
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
    if seucar_switch:
        trained_model = '/home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_54_800000.pth'
    else:
        trained_model = '/home/lishangjie/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_54_800000.pth'
    model_path = SavePath.from_str(trained_model)
    yolact_config = model_path.model_name + '_config_coco'
    print('Config not specified. Parsed %s from the file name.\n' % yolact_config)
    set_cfg_coco(yolact_config)
    # 加载网络模型
    print('Loading the second model...')
    net_coco = Yolact_coco()
    net_coco.load_weights(trained_model)
    net_coco.eval()
    if cuda_mode:
        net_coco = net_coco.cuda()
    net_coco.detect.use_fast_nms = True
    net_coco.detect.use_cross_class_nms = False
    cfg_coco.mask_proto_debug = False
    print('  Done.\n')
    
    # 设置相机参数
    CameraT = RegionDivide()
    if seucar_switch:
        if(not CameraT.loadCameraInfo("/home/seucar/wendao/sweeper/region_divide/right.yaml")):
            print("loadCameraInfo failed!")
    else:
        if(not CameraT.loadCameraInfo("/home/lishangjie/wendao/sweeper/region_divide/right.yaml")):
            print("loadCameraInfo failed!")
    
    # 初始化detection节点
    print()
    print('Waiting for node...')
    rospy.init_node("detection")
    
    # 设置区域划分参数
    l1 = rospy.get_param("~region_l1")
    l2 = rospy.get_param("~region_l2")
    l3 = rospy.get_param("~region_l3")
    l4 = rospy.get_param("~region_l4")
    l5 = rospy.get_param("~region_l5")
    CameraT.setRegionParams(l1, l2, l3, l4, l5)
    
    # p2d_table是三维数组，第一维是u第二维是v，第三维存储世界坐标系xz
    p2d_table = pixel2disTable(CameraT.size[0], CameraT.size[1], CameraT.height, CameraT.theta, CameraT.fx, CameraT.fy, CameraT.cx, CameraT.cy)
    cv_image = np.array([])
    
    # 获取订阅话题名、发布话题名及其他参数
    sub_topic_image = rospy.get_param("~sub_topic_image")
    pub_topic_areasinfo = rospy.get_param("~pub_topic_areasinfo")
    pub_topic_objects = rospy.get_param("~pub_topic_objects")
    pub_topic_envinfo = rospy.get_param("~pub_topic_envinfo")
    
    max_width = rospy.get_param("~max_width")
    coordinate_offset_x = rospy.get_param("~coordinate_offset_x")
    coordinate_offset_y = rospy.get_param("~coordinate_offset_y")
    
    # 发布消息队列设为1，订阅消息队列设为1，并保证订阅消息缓冲区足够大
    # 这样可以实现每次订阅最新的节点消息，避免因队列消息拥挤而导致的延迟
    pub_areasinfo = rospy.Publisher(pub_topic_areasinfo, AreasInfo, queue_size=1)
    pub_objects = rospy.Publisher(pub_topic_objects, Objects, queue_size=1)
    pub_envinfo = rospy.Publisher(pub_topic_envinfo, EnvInfo, queue_size=1)
    
    rospy.Subscriber(sub_topic_image, Image, image_callback, queue_size=1, buff_size=52428800)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
