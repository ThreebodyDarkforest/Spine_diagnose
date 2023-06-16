#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp
import gradio

import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules.components.logger import LOGGER
from modules.yolov6.core.inferer import Inferer
from modules.yolov6.utils.events import save_yaml, load_yaml
from modules.adapter import get_disease_str
from modules.classifier import get_resnet_model, classify
from modules.detector import detect, get_yolo_model
from modules.util import filter_box, plot_boxes, crop_img, get_center, IMG_EXT

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Spine diagnose PyTorch Inference.', add_help=add_help)
    parser.add_argument('--detect-model', type=str, default='weights/detect.pt', help='detect model path(s) for inference.')
    parser.add_argument('--classify-model', type=str, default='weights/classify.pt', help='classify model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=str, default='0', help='the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str, default='data/spine.yaml', help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--save-img', action='store_true', help='save inference results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default='runs/inference', help='save inference results to project/name.')
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    args = parser.parse_args()
    LOGGER.info(args)
    return args

def main(image, args):
    try:
        device = torch.device(args.device)
    except:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    detect_model, classify_model = None, None
    cfgs = load_yaml(args.yaml)
    if 'yolov6' in cfgs['detect']:
        detect_model, stride, dclass_names = get_yolo_model(args.detect_model, args.yaml, device=device)
    if 'resnet' in cfgs['classify']:
        classify_model, cclass_names = get_resnet_model(args.classify_model, args.yaml, device=device)
    img_src, boxes = detect(detect_model, image, dclass_names, stride=stride, device=device)
    boxes = filter_box(img_src, boxes)
    imgs = crop_img(img_src, boxes, 4)

    ret = [classify(classify_model, img, cclass_names, device=device) for img in imgs]
    for data, box in zip(ret, boxes):
            disease_type, conf, logits = data[0], data[1], data[2]
            box['label'] += ' ' + get_disease_str(disease_type)
            box['dlabel'] = get_disease_str(disease_type)
            box['logits'] = [x * box['confidence'] for x in logits]
            box['confidence'] *= conf[0] * conf[1]
    plot_boxes(img_src, 1, boxes)
    return img_src

if __name__ == "__main__":
    args = get_args_parser()
    webui_predict = lambda image : main(image, args)
    interface = gradio.Interface(fn=webui_predict, inputs="image", outputs="image")
    interface.launch()