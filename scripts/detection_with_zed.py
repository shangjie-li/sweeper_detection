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
    
TODO：
    1.分别存储垃圾、植被、行人目标的掩膜、类别、置信度、边界框
    2.垃圾目标显示选项：2D/3D边界框、掩膜、类别、置信度
    3.植被目标显示选项：2D/3D边界框、掩膜、类别、置信度
    4.行人目标显示选项：2D/3D边界框、掩膜、类别、置信度
    5.统计回调函数中获取rosparam变量的时间
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
display_contours = True

show_pointcloud = True
show_s = True
show_p = True
show_w = True
show_output = True
show_region = True
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

def pointcloud_display(img, camera_xyz, camera_uv):
    jc = Jet_Color()
    depth = np.sqrt(np.square(camera_xyz[:, 0]) + np.square(camera_xyz[:, 1]) + np.square(camera_xyz[:, 2]))
    for pt in range(0, camera_uv.shape[0]):
        cv_color = jc.get_jet_color(depth[pt] * jet_color)
        cv2.circle(img, (int(camera_uv[pt][0]), int(camera_uv[pt][1])), 1, cv_color, thickness=-1)
    return img
    
def get_color(color_idx, on_gpu=None):
    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    color_cache = defaultdict(lambda: {})
    
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
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0
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
    
def output_display(img, region_output):
    r = region_output
    # format格式化函数
    # {:.0f} 不带小数，{:.2f} 保留两位小数，{:>3} 右对齐且宽度为3，{:<3} 左对齐且宽度为3
    for i in range(8):
        out_str = "ID:{:.0f} R:{:.0f} V:{:.0f} P:{:.0f} w:{:>3} s:{:>6} v:{:>6} x:{:>6} z:{:>6} dx:{:>6} dz:{:>6}".format(
                    r[i, 3], r[i, 0], r[i, 1], r[i, 2], 
                    int(r[i, 4]), round(r[i, 5], 4), round(r[i, 6], 4), 
                    round(r[i, 7], 2), round(r[i, 8], 2), round(r[i, 9], 2), round(r[i, 10], 2))
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
        num_dets_to_consider = len(remain_list)
        
        if num_dets_to_consider > 0:
            return masks[remain_list], classes[remain_list], scores[remain_list], boxes[remain_list], num_dets_to_consider
        else:
            return None, None, None, None, 0
            
def get_boundingbox(xs, zs):
    # 功能：以垂直包络矩形作为投影轮廓
    #
    # 输入：xs <class 'numpy.ndarray'> 为(n,)维矩阵，代表横坐标
    #      zs <class 'numpy.ndarray'> 为(n,)维矩阵，代表纵坐标
    #
    # 输出：hull <class 'numpy.ndarray'> 为(n,1,2)维矩阵，n为轮廓点数
    
    min_x = xs.min()
    max_x = xs.max()
    min_z = zs.min()
    max_z = zs.max()
    
    p1 = np.array([[min_x, min_z]])
    p2 = np.array([[max_x, min_z]])
    p3 = np.array([[max_x, max_z]])
    p4 = np.array([[min_x, max_z]])
    
    # 水平面投影轮廓，(n,1,2)维矩阵，n为轮廓点数
    hull = np.array([p1, p2, p3, p4])
    return hull
    
def get_convexhull(xs, zs):
    # 功能：以凸包作为投影轮廓
    #
    # 输入：xs <class 'numpy.ndarray'> 为(n,)维矩阵，代表横坐标
    #      zs <class 'numpy.ndarray'> 为(n,)维矩阵，代表纵坐标
    #
    # 输出：hull <class 'numpy.ndarray'> 为(n,1,2)维矩阵，n为轮廓点数
    
    xs = xs * 100
    zs = zs * 100
    xs = xs.astype(np.int)
    zs = zs.astype(np.int)
    
    pts = np.array((xs, zs)).T
    hull = cv2.convexHull(pts)
    
    # 水平面投影轮廓，(n,1,2)维矩阵，n为轮廓点数
    hull = hull / 100.0
    return hull
    
def fusion(camera_xyz, camera_uv, target_masks, target_classes, target_scores, target_boxes, num_dets_to_consider):
    # region_output是8行11列数组，每一行存储一个区域的信息
    # 第1列为污染等级(0,1,2,3,4,5,6,7)
    # 第2列为植被类型(0无,1草,2灌木,3花)
    # 第3列为行人标志(0无,1有)
    # 第4列为区域ID(1,2,3,4,5,6,7,8)
    # 第5列为区域内垃圾总质量(单位g)
    # 第6列为区域内最大单体垃圾的面积(单位m2)
    # 第7列为区域内最大单体垃圾的体积(单位m3)
    # 第8列为区域内最大单体垃圾的左前点x坐标(单位m)
    # 第9列为区域内最大单体垃圾的左前点z坐标(单位m)
    # 第10列为区域内最大单体垃圾的x方向长度(单位m)
    # 第11列为区域内最大单体垃圾的z方向长度(单位m)
    # 最大单体：水平面投影面积最大
    region_output = np.zeros((8, 11))
    
    # 区域ID初始化
    for region_i in range(8):
        region_output[region_i, 3] = region_i + 1
        
    # 确保处理对象非空
    if num_dets_to_consider > 0:
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
            
        rubbish_masks = target_masks[rubbish_remain_list, :, :]
        rubbish_classes = target_classes[rubbish_remain_list]
        
        vegetation_masks = target_masks[vegetation_remain_list, :, :]
        vegetation_classes = target_classes[vegetation_remain_list]
        
        person_masks = target_masks[person_remain_list, :, :]
        person_classes = target_classes[person_remain_list]
        
        rubbsih_num = len(rubbish_remain_list)
        vegetation_num = len(vegetation_remain_list)
        person_num = len(person_remain_list)
        
        # 针对垃圾目标的处理
        if rubbsih_num > 0:
            # 在CPU上操作掩膜
            rubbish_masks = rubbish_masks.byte().cpu().numpy()
            
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
                for pt in range(camera_xyz.shape[0]):
                    if rubbish_masks[i, int(camera_uv[pt][1]), int(camera_uv[pt][0])]:
                        target_xs.append(camera_xyz[pt][0])
                        target_ys.append(camera_xyz[pt][1])
                        target_zs.append(camera_xyz[pt][2])
                        
                # 如果掩膜中包含特征点云
                if len(target_xs):
                    pts_xyz = np.array([target_xs, target_ys, target_zs], dtype=np.float32).T
                    
                    # 水平面投影轮廓，hull为n×1×2维矩阵，n为轮廓点数
                    hull = get_boundingbox(pts_xyz[:, 0], pts_xyz[:, 2])
                    b_x = hull[:, 0, 0]
                    b_z = hull[:, 0, 1]
                    effective_pt_num = hull.shape[0]
                    
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
                        s_sum += b_x[b_pt]*b_z[(b_pt + 1)%effective_pt_num] - b_z[b_pt]*b_x[(b_pt + 1)%effective_pt_num]
                    target_area = abs(s_sum) / 2
                    
                    # 计算体积和质量
                    min_y = pts_xyz[:, 1].min()
                    max_y = pts_xyz[:, 1].max()
                    target_height = abs(max_y - min_y)
                    target_volume = target_area * target_height
                    w_coef = rubbish_weight_coefficient_list[rubbish_items.index(rubbish_classes[i])]
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
            vegetation_masks = vegetation_masks.byte().cpu().numpy()
            
            # 为不同植被分配优先级
            vegetation_priority_list = [0, 1, 2]
            assert len(vegetation_priority_list) == len(vegetation_items)
            
            # 遍历每个目标
            for i in range(vegetation_num):
                target_xs = []
                target_ys = []
                target_zs = []
                
                # 提取掩膜中的特征点
                for pt in range(camera_xyz.shape[0]):
                    if vegetation_masks[i, int(camera_uv[pt][1]), int(camera_uv[pt][0])]:
                        target_xs.append(camera_xyz[pt][0])
                        target_ys.append(camera_xyz[pt][1])
                        target_zs.append(camera_xyz[pt][2])
                        
                # 如果掩膜中包含特征点云
                if len(target_xs):
                    pts_xyz = np.array([target_xs, target_ys, target_zs], dtype=np.float32).T
                    
                    # 水平面投影轮廓，hull为n×1×2维矩阵，n为轮廓点数
                    hull = get_convexhull(pts_xyz[:, 0], pts_xyz[:, 2])
                    b_x = hull[:, 0, 0]
                    b_z = hull[:, 0, 1]
                    effective_pt_num = hull.shape[0]
                    
                    # 计算优先级
                    target_priority = vegetation_priority_list[vegetation_items.index(vegetation_classes[i])]
                    
                    # 利用轮廓点进行目标定位，更新各区域内植被的优先级
                    for b_pt in range(effective_pt_num):
                        region = locat.findregion(b_x[b_pt], b_z[b_pt])
                        if region > 0 and target_priority > region_output[region - 1, 1]:
                            region_output[region - 1, 1] = target_priority
                            
        # 针对行人目标的处理
        if person_num > 0:
            # 在CPU上操作掩膜
            person_masks = person_masks.byte().cpu().numpy()
            
            # 遍历每个目标
            for i in range(person_num):
                target_xs = []
                target_ys = []
                target_zs = []
                
                # 提取掩膜中的特征点
                for pt in range(camera_xyz.shape[0]):
                    if person_masks[i, int(camera_uv[pt][1]), int(camera_uv[pt][0])]:
                        target_xs.append(camera_xyz[pt][0])
                        target_ys.append(camera_xyz[pt][1])
                        target_zs.append(camera_xyz[pt][2])
                        
                # 如果掩膜中包含特征点云
                if len(target_xs):
                    pts_xyz = np.array([target_xs, target_ys, target_zs], dtype=np.float32).T
                    
                    # 水平面投影轮廓，hull为n×1×2维矩阵，n为轮廓点数
                    hull = get_boundingbox(pts_xyz[:, 0], pts_xyz[:, 2])
                    b_x = hull[:, 0, 0]
                    b_z = hull[:, 0, 1]
                    effective_pt_num = hull.shape[0]
                    
                    # 利用轮廓点进行目标定位，更新各区域内行人标志位
                    for b_pt in range(effective_pt_num):
                        region = locat.findregion(b_x[b_pt], b_z[b_pt])
                        if region > 0:
                            region_output[region - 1, 2] = 1
                        
    return region_output
        
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
    
    global display_switch
    display_switch = rospy.get_param("~display_mode")
    global record_switch
    record_switch = rospy.get_param("~record_mode")
    global record_initialized
    global video_raw
    global video_result
    
    global display_masks, display_bboxes, display_scores, display_contours
    display_masks = rospy.get_param("~display_masks")
    display_bboxes = rospy.get_param("~display_bboxes")
    display_scores = rospy.get_param("~display_scores")
    display_contours = rospy.get_param("~display_contours")
    
    global show_pointcloud, show_s, show_p, show_w, show_output, show_region, show_time
    show_pointcloud = rospy.get_param("~show_pointcloud")
    show_s = rospy.get_param("~show_s")
    show_p = rospy.get_param("~show_p")
    show_w = rospy.get_param("~show_w")
    show_output = rospy.get_param("~show_output")
    show_region = rospy.get_param("~show_region")
    show_time = rospy.get_param("~show_time")
    
    global print_stamp, print_xx
    print_stamp = rospy.get_param("~print_stamp")
    print_xx = rospy.get_param("~print_xx")
    
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
    time_start = time.time()
    if modal_custom or modal_coco:
        target_masks, target_classes, target_scores, target_boxes, num_target = detection(current_image)
    else:
        target_masks, target_classes, target_scores, target_boxes = None, None, None, None
        num_target = 0
    time_detection = round(time.time() - time_start, 3)
    
    # 载入点云
    pointXYZ = pointcloud2_to_xyz_array(pointcloud, remove_nans=True)
    if print_xx:
        print("Input pointcloud of", pointXYZ.shape[0])
    if limit_mode:
        alpha = 90 - 0.5 * field_of_view
        k = math.tan(alpha * math.pi / 180.0)
        pointXYZ = pointXYZ[np.logical_and((pointXYZ[:, 0] > k * pointXYZ[:, 1]), (pointXYZ[:, 0] > -k * pointXYZ[:, 1]))]
    if clip_mode:
        pointXYZ = pointXYZ[np.logical_and((pointXYZ[:, 0] ** 2 + pointXYZ[:, 1] ** 2 > min_distance ** 2), (pointXYZ[:, 0] ** 2 + pointXYZ[:, 1] ** 2 < max_distance ** 2))]
        pointXYZ = pointXYZ[np.logical_and((pointXYZ[:, 2] > view_lower_limit - sensor_height), (pointXYZ[:, 2] < view_higher_limit - sensor_height))]
        
    cloud_xyz = calib.lidar_to_cam.dot(pointXYZ.T).T
    cloud_uv = calib.lidar_to_img.dot(pointXYZ.T).T
    cloud_uv = np.true_divide(cloud_uv[:, :2], cloud_uv[:, [-1]])
    camera_xyz = cloud_xyz[(cloud_uv[:, 0] >= 0) & (cloud_uv[:, 0] < frame_width) & (cloud_uv[:, 1] >= 0) & (cloud_uv[:, 1] < frame_height)]
    camera_uv = cloud_uv[(cloud_uv[:, 0] >= 0) & (cloud_uv[:, 0] < frame_width) & (cloud_uv[:, 1] >= 0) & (cloud_uv[:, 1] < frame_height)]
    if print_xx:
        print("Process pointcloud of", camera_xyz.shape[0])
        print()
    
    # 数据融合及结果统计
    time_start = time.time()
    if modal_custom or modal_coco:
        region_output = fusion(camera_xyz, camera_uv, target_masks, target_classes, target_scores, target_boxes, num_target)
    else:
        region_output = None
    time_fusion = round(time.time() - time_start, 3)
    
    # 发布检测结果话题
    if modal_custom or modal_coco:
        convert(region_output)
    else:
        rospy.logerr("Nothing to be converted!")
        
    # 修改图像
    if display_switch or record_switch:
        # 添加点云
        if show_pointcloud:
            current_image = pointcloud_display(current_image, camera_xyz, camera_uv)
        # 添加检测结果
        if num_target > 0:
            current_image = result_display(current_image, target_masks, target_classes, target_scores, target_boxes, num_target)
        if num_target > 0:
            current_image = output_display(current_image, region_output)
        # 添加系统时间
        if show_time:
            cv2.putText(current_image, str(time.time()), (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
    if display_switch:
        if print_xx and region_output is not None:
            # region_output是8行11列数组，每一行存储一个区域的信息
            # 第1列为污染等级(0,1,2,3,4,5,6,7)
            # 第2列为植被类型(0无,1草,2灌木,3花)
            # 第3列为行人标志(0无,1有)
            # 第4列为区域ID(1,2,3,4,5,6,7,8)
            # 第5列为区域内垃圾总质量(单位g)
            # 第6列为区域内最大单体垃圾的面积(单位m2)
            # 第7列为区域内最大单体垃圾的体积(单位m3)
            # 第8列为区域内最大单体垃圾的左前点x坐标(单位m)
            # 第9列为区域内最大单体垃圾的左前点z坐标(单位m)
            # 第10列为区域内最大单体垃圾的x方向长度(单位m)
            # 第11列为区域内最大单体垃圾的z方向长度(单位m)
            # 最大单体：水平面投影面积最大
            print('region_output')
            r = region_output
            # format格式化函数
            # {:.0f} 不带小数，{:.2f} 保留两位小数，{:>3} 右对齐且宽度为3，{:<3} 左对齐且宽度为3
            for i in range(8):
                print("ID:{:.0f} R:{:.0f} V:{:.0f} P:{:.0f} w:{:>3} s:{:>6} v:{:>6} x:{:>6} z:{:>6} dx:{:>6} dz:{:>6}".format(
                    r[i, 3], r[i, 0], r[i, 1], r[i, 2], 
                    int(r[i, 4]), round(r[i, 5], 4), round(r[i, 6], 4), 
                    round(r[i, 7], 2), round(r[i, 8], 2), round(r[i, 9], 2), round(r[i, 10], 2)))
            print()
            
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
        
    time_end_all = time.time()
    if print_stamp:
        print("totally time cost:", time_end_all - time_start_all)
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

