# -*- coding: UTF-8 -*-
#!/usr/bin/env python3

"""
修改注释(20200712)：
    1.在parse_args中预留参数接口
    2.增加显示时间戳功能
    3.优化地面目标面积估算方法
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from transmiter.msg import AreaInfo
from transmiter.msg import AreasInfo
import message_filters
import numpy as np
np.set_printoptions(suppress=True)
import time
import math

import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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

from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess
from data import cfg, set_cfg

sys.path.append('/home/seucar/wendao/sweeper/region_divide')
from region_divide import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='/home/seucar/wendao/sweeper/ros_ws/src/sweeper_detection/scripts/weights/yolact_resnet50_155_30000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.15, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    # Added by shangjie 20200710.
    parser.add_argument('--display_mode', default=True, type=str2bool,
                        help='Display detection results in real time.')
    parser.add_argument('--record_mode', default=False, type=str2bool,
                        help='Record detection results in real time.')
    parser.add_argument('--l1', default=5, type=int,
                        help='Parameter L1 of region divide')
    parser.add_argument('--l2', default=3, type=int,
                        help='Parameter L2 of region divide')
    parser.add_argument('--l3', default=3, type=int,
                        help='Parameter L3 of region divide')
    parser.add_argument('--l4', default=3, type=int,
                        help='Parameter L4 of region divide')
    parser.add_argument('--l5', default=2, type=int,
                        help='Parameter L5 of region divide')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

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
    if args.display_masks and cfg.eval_mask_branch and num_target > 0:
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
        score = scores[i]
        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
        _class = cfg.dataset.class_names[classes[i]]
        text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
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
            binary_image_back, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
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
            if display_switch:
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
            if display_switch:
                cv2.drawContours(img_numpy, contours_cpts, -1, (0, 0, 255), 3)
            for c_cpt in range(contours_cpts.shape[0]):
                boundary_pts[i, c_cpt, 0, :] = contours_cpts[c_cpt, 0, :]
        
        except IndexError:
            pass
    
    boundary_pts = boundary_pts.astype(np.uint16)
    return img_numpy, boundary_pts

def w_display(result_image, region_w, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1):
    text_str = 'weight:' + str(region_w[0, 0].astype(np.uint8))
    text_xy = (max(CameraT.b3[0], 0), CameraT.b3[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'weight:' + str(region_w[1, 0].astype(np.uint8))
    text_xy = (CameraT.a3[0], CameraT.a3[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'weight:' + str(region_w[2, 0].astype(np.uint8))
    text_xy = (int((CameraT.a3[0] + CameraT.a4[0]) / 2), CameraT.a3[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'weight:' + str(region_w[3, 0].astype(np.uint8))
    text_xy = (CameraT.a4[0], CameraT.a4[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'weight:' + str(region_w[4, 0].astype(np.uint8))
    text_xy = (max(CameraT.b5[0], 0), CameraT.b5[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'weight:' + str(region_w[5, 0].astype(np.uint8))
    text_xy = (CameraT.a5[0], CameraT.a5[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'weight:' + str(region_w[6, 0].astype(np.uint8))
    text_xy = (int((CameraT.a5[0] + CameraT.a6[0]) / 2), CameraT.a5[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'weight:' + str(region_w[7, 0].astype(np.uint8))
    text_xy = (CameraT.a6[0], CameraT.a6[1] + 28)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return result_image

def output_display(result_image, region_output, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1):
    text_str = 'R:' + str(region_output[0, 0].astype(np.uint8)) + ' P:' + str(region_output[0, 2].astype(np.uint8)) + ' V:' + str(region_output[0, 1].astype(np.uint8))
    text_xy = (max(CameraT.b3[0], 0), CameraT.b3[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[1, 0].astype(np.uint8)) + ' P:' + str(region_output[1, 2].astype(np.uint8)) + ' V:' + str(region_output[1, 1].astype(np.uint8))
    text_xy = (CameraT.a3[0], CameraT.a3[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[2, 0].astype(np.uint8)) + ' P:' + str(region_output[2, 2].astype(np.uint8)) + ' V:' + str(region_output[2, 1].astype(np.uint8))
    text_xy = (int((CameraT.a3[0] + CameraT.a4[0]) / 2), CameraT.a3[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[3, 0].astype(np.uint8)) + ' P:' + str(region_output[3, 2].astype(np.uint8)) + ' V:' + str(region_output[3, 1].astype(np.uint8))
    text_xy = (CameraT.a4[0], CameraT.a4[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[4, 0].astype(np.uint8)) + ' P:' + str(region_output[4, 2].astype(np.uint8)) + ' V:' + str(region_output[4, 1].astype(np.uint8))
    text_xy = (max(CameraT.b5[0], 0), CameraT.b5[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[5, 0].astype(np.uint8)) + ' P:' + str(region_output[5, 2].astype(np.uint8)) + ' V:' + str(region_output[5, 1].astype(np.uint8))
    text_xy = (CameraT.a5[0], CameraT.a5[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[6, 0].astype(np.uint8)) + ' P:' + str(region_output[6, 2].astype(np.uint8)) + ' V:' + str(region_output[6, 1].astype(np.uint8))
    text_xy = (int((CameraT.a5[0] + CameraT.a6[0]) / 2), CameraT.a5[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    text_str = 'R:' + str(region_output[7, 0].astype(np.uint8)) + ' P:' + str(region_output[7, 2].astype(np.uint8)) + ' V:' + str(region_output[7, 1].astype(np.uint8))
    text_xy = (CameraT.a6[0], CameraT.a6[1] + 12)
    cv2.putText(result_image, text_str, text_xy, font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return result_image

def image_callback(image_data):
    time_start = time.time()
    global cv_image
    cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
    
    # region_output是8行4列数组，第i行存储第i个区域的信息
    # 每行的第1列为污染等级(0,1,2,3,4)、第2列为植被类型(0无,1草,2灌木,3花)、第3列为行人标志(0无,1有)、第4列为区域ID(1,2,3,4,5,6,7,8)
    region_output = np.zeros((8, 4))
    for region_i in range(8):
        region_output[region_i, 3] = region_i + 1
    
    with torch.no_grad():
        # 目标检测
        frame = torch.from_numpy(cv_image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        
        # 建立每个目标的蒙版target_masks、类别target_classes、置信度target_scores、边界框target_boxes的一一对应关系
        h, w, _ = frame.shape
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            # 检测结果
            t = postprocess(preds, w, h, visualize_lincomb = args.display_lincomb,
                                         crop_masks        = args.crop,
                                         score_threshold   = args.score_threshold)
            cfg.rescore_bbox = save
        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:args.top_k]
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        
        num_dets_to_consider = min(args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < args.score_threshold:
                num_dets_to_consider = j
                break
        
        if num_dets_to_consider > 0:
            target_masks = masks[:num_dets_to_consider, :, :]
            target_classes = classes[:num_dets_to_consider]
            target_scores = scores[:num_dets_to_consider]
            target_boxes = boxes[:num_dets_to_consider, :]
            
            # 显示检测结果
            if display_switch:
                result_image = result_display(frame, target_masks, target_classes, target_scores, target_boxes, num_dets_to_consider)
            else:
                result_image = frame.byte().cpu().numpy()
            
            # 分别存储垃圾目标和植被目标
            check_k = 0
            rubbish_remain_list = []
            vegetation_remain_list = []
            rubbish_items = ['ads', 'cigarette', 'firecracker', 'glass bottle', 'leaves', 'metal', 'paper', 'peel', 'plastic', 'solid clod', 'solid crumb']
            vegetation_items = ['grass', 'shrub', 'flower']
            while check_k < target_classes.shape[0]:
                if cfg.dataset.class_names[target_classes[check_k]] in rubbish_items:
                    rubbish_remain_list.append(check_k)
                if cfg.dataset.class_names[target_classes[check_k]] in vegetation_items:
                    vegetation_remain_list.append(check_k)
                check_k += 1
            
            rubbish_masks = target_masks[rubbish_remain_list, :, :]
            rubbish_classes = target_classes[rubbish_remain_list]
            rubbish_scores = target_scores[rubbish_remain_list]
            rubbish_boxes = target_boxes[rubbish_remain_list, :]
            
            vegetation_masks = target_masks[vegetation_remain_list, :, :]
            vegetation_classes = target_classes[vegetation_remain_list]
            vegetation_scores = target_scores[vegetation_remain_list]
            vegetation_boxes = target_boxes[vegetation_remain_list, :]
            
            rubbsih_num = len(rubbish_remain_list)
            vegetation_num = len(vegetation_remain_list)
            
            # 针对垃圾目标的处理
            if rubbsih_num > 0:
                # 掩膜边界取点
                result_image, rubbish_boundary_pts = get_boundary(result_image, rubbsih_num, rubbish_masks, cpt_num=10)
                # s_polygon存储每个垃圾目标在世界坐标系中投影于地面的面积
                s_polygon = np.zeros((rubbsih_num, 1))
                rubbish_list = ['ads', 'cigarette', 'firecracker', 'glass bottle', 'leaves', 'metal', 'paper', 'peel', 'plastic', 'solid clod', 'solid crumb']
                rubbish_weight_coefficient_list = [80, 200, 200, 8000, 80, 1050, 80, 6000, 775, 15750, 4000]
                # region_w存储各区域内垃圾目标的质量的总和
                region_w = np.zeros((8, 1))
                for i in range(rubbish_boundary_pts.shape[0]):
                    effective_pt_num = 0
                    b_x, b_z = [], []
                    b_area_id = []
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
                    if effective_pt_num >= 3:
                        s_sum = 0
                        for b_pt in range(effective_pt_num):
                            s_sum += b_x[b_pt]*b_z[(b_pt + 1)%effective_pt_num] - b_z[b_pt]*b_x[(b_pt + 1)%effective_pt_num]
                        s_polygon[i, 0] = abs(s_sum) / 2
                        for b_pt in range(effective_pt_num):
                            # 排除区域ID无效点(ID=0)
                            if b_area_id[b_pt]:
                                rubbish_weight = s_polygon[i, 0] * rubbish_weight_coefficient_list[rubbish_list.index(cfg.dataset.class_names[rubbish_classes[i]])]
                                region_w[b_area_id[b_pt] - 1, 0] += rubbish_weight / effective_pt_num
                
                # 界定污染等级
                for region_i in range(8):
                    if region_w[region_i, 0] > 0 and region_w[region_i, 0] <= 50:
                        region_output[region_i, 0] = 1
                    elif region_w[region_i, 0] > 50 and region_w[region_i, 0] <= 100:
                        region_output[region_i, 0] = 2
                    elif region_w[region_i, 0] > 100 and region_w[region_i, 0] <= 150:
                        region_output[region_i, 0] = 3
                    elif region_w[region_i, 0] > 150:
                        region_output[region_i, 0] = 4
                if display_switch:
                    print('region_w')
                    print(region_w)
                    result_image = w_display(result_image, region_w, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1)
            
            # 针对植被目标的处理
            if vegetation_num > 0:
                # 掩膜边界取点
                result_image, vegetation_boundary_pts = get_boundary(result_image, vegetation_num, vegetation_masks, cpt_num=20)
                # region_vegetation_type存储各区域内植被类型
                region_vegetation_type = np.zeros((8, 1))
                for i in range(vegetation_boundary_pts.shape[0]):
                    effective_pt_num = 0
                    b_area_id = []
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
                    for b_pt in range(effective_pt_num):
                        # 排除区域ID无效点(ID=0)
                        if b_area_id[b_pt]:
                            # 优先级顺序
                            vegetation_list = ['grass', 'shrub', 'flower']
                            v_type = vegetation_list.index(cfg.dataset.class_names[vegetation_classes[i]]) + 1
                            current_v_type = region_vegetation_type[b_area_id[b_pt] - 1, 0]
                            if v_type > current_v_type:
                                region_vegetation_type[b_area_id[b_pt] - 1, 0] = v_type
                
                for region_i in range(8):
                    region_output[region_i, 1] = region_vegetation_type[region_i, 0]
            
        else:
            result_image = frame.byte().cpu().numpy()
    
    areasinfo_msg = AreasInfo()
    for region_i in range(8):
        region_output_msg = AreaInfo()
        region_output_msg.rubbish_grade = int(region_output[region_i, 0])
        region_output_msg.has_person = bool(region_output[region_i, 2])
        region_output_msg.vegetation_type = int(region_output[region_i, 1])
        region_output_msg.area_id = int(region_output[region_i, 3])
        areasinfo_msg.infos.append(region_output_msg)
    pub.publish(areasinfo_msg)
    
    if display_switch:
        print('region_output')
        print(region_output)
        result_image = CameraT.drawLine(result_image, w=1)
        result_image = output_display(result_image, region_output, font_face = cv2.FONT_HERSHEY_DUPLEX, font_scale = 0.5, font_thickness = 1)
        cv2.putText(result_image, str(time.time()), (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("result_image", result_image)
    if record_switch:
        video_out.write(result_image)
    
    if cv2.waitKey(1) == 27:
        if record_switch:
            video_out.release()
        cv2.destroyAllWindows()
        rospy.signal_shutdown("It's over.")
    
    time_end_all = time.time()
    print("totally time cost:", time_end_all - time_start)

if __name__ == '__main__':
    # YOLACT
    parse_args()
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    # 加载网络模型
    print('Loading model...')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    
    if args.cuda:
        net = net.cuda()
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    print('  Done.')
    
    display_switch = args.display_mode
    record_switch = args.record_mode
    
    # 设置保存结果文件
    if record_switch:
        out_path = 'video_out.mp4'
        target_fps = 10
        frame_height = 480
        frame_width = 640
        video_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height), True)
    
    # 设置相机参数
    CameraT = RegionDivide()
    if(not CameraT.loadCameraInfo("/home/seucar/wendao/sweeper/region_divide/right.yaml")):
        print("loadCameraInfo failed!")
    
    # 设置区域划分参数
    CameraT.setRegionParams(args.l1, args.l2, args.l3, args.l4, args.l5)
    
    # p2d_table是三维数组，第一维是u第二维是v，第三维存储世界坐标系xz
    p2d_table = pixel2disTable(CameraT.size[0], CameraT.size[1], CameraT.height, CameraT.theta, CameraT.fx, CameraT.fy, CameraT.cx, CameraT.cy)
    cv_image = np.array([])
    
    # 初始化fusion节点
    # 发布消息队列设为1，订阅消息队列设为1，并保证订阅消息缓冲区足够大
    # 这样可以实现每次订阅最新的节点消息，避免因队列消息拥挤而导致的延迟
    print('Waiting for node...')
    rospy.init_node("detection")
    rospy.Subscriber('/image_rectified', Image, image_callback, queue_size=1, buff_size=52428800)
    pub = rospy.Publisher('/sweeper/area_info', AreasInfo, queue_size=1)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
