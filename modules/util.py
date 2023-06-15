import json, os
import PIL
import cv2
import numpy as np
from skimage import morphology
from copy import deepcopy
import os.path as osp
import shutil
import sys
import tempfile
from importlib import import_module
from addict import Dict

bone_asc = {
    'Anchor': 0,
    'L5-S1': 1,
    'L5': 2,
    'L4-L5': 3,
    'L4': 4,
    'L3-L4': 5,
    'L3': 6,
    'L2-L3': 7,
    'L2': 8,
    'L1-L2': 9,
    'L1': 10,
    'T12-L1': 11,
    'All': 12,
}

IMG_EXT = ["jpg", "jpeg", "png", "bmp"]

def get_skeleton(img, roi_box, scale=0.5):
    img = img[roi_box[0][1]:roi_box[1][1], roi_box[0][0]:roi_box[1][0]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist_trans = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    bw_img = np.uint8(dist_trans > 0.3 * dist_trans.max()) * 255

    bw_img = cv2.resize(bw_img, np.int16(np.float32(bw_img.shape) * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    opened = cv2.bitwise_not(opened)
    
    skeleton = morphology.skeletonize(opened)
    skeleton = morphology.remove_small_holes(skeleton)
    skeleton = morphology.remove_small_objects(skeleton, min_size=10)
    result = cv2.resize(np.uint8(skeleton), bw_img.shape)
    return result

def get_center(box):
    return (box['xyxy'][0][0] + box['xyxy'][1][0]) / 2, (box['xyxy'][0][1] + box['xyxy'][1][1]) / 2

def distance(p1, p2):
    return int(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

def get_size(box):
    return (box['xyxy'][1][0] - box['xyxy'][0][0], box['xyxy'][1][1] - box['xyxy'][0][1])

def crop_img(img, boxes, padding: int = 2):
    crop_imgs = []
    for box in boxes:
        p1, p2 = box['xyxy'][0], box['xyxy'][1]
        crop_imgs.append(img[int(p1[0]) - padding : int(p2[0]) + padding, \
                             int(p1[1]) - padding : int(p2[1]) + padding])
    return crop_imgs

def filter_box(img, boxes):
    boxes = sorted(boxes, key=lambda x : bone_asc[x['label']])
    anchor_box, all_box = boxes[0], boxes[-1]
    assert anchor_box['label'] == 'Anchor' and all_box['label'] == 'All', 'Invalid image.'
    
    # filter those boxes that out of the roi
    boxes = list(filter(lambda x : x['xyxy'][0][1] + x['xyxy'][1][1] <=  
                        anchor_box['xyxy'][0][1] + anchor_box['xyxy'][1][1] and
                        get_center(x)[0] > all_box['xyxy'][0][0] and
                        get_center(x)[0] < all_box['xyxy'][1][0] and
                        get_center(x)[1] > all_box['xyxy'][0][1] and
                        get_center(x)[1] < all_box['xyxy'][1][1], boxes))
    bucket = [[] for _ in range(13)]
    [bucket[bone_asc[x['label']]].append(x) for x in boxes]
    all_box_size = get_size(all_box)
    mesh_size = all_box_size[1] / 11
    bone_str = list(bone_asc.keys())

    # Fix unscanned boxes
    for i in range(1, 12):
        if len(bucket[i]) <= 0:
            for box in boxes:
                # TODO: use some other precise method to find the most matched box
                if (i - 1) * mesh_size <= get_center(box)[1] <= i * mesh_size:
                    nw_box = deepcopy(box)
                    nw_box['label'] = bone_str[i]
                    bucket[i].append(box)
        if len(bucket[i]) <= 0:
            # TODO: use prior knowledge of angles and positions.
            nw_box = deepcopy(bucket[i - 1][0])
            nw_box['xyxy'] = ((nw_box['xyxy'][0][0], nw_box['xyxy'][0][1] - mesh_size), 
                              (nw_box['xyxy'][1][0], nw_box['xyxy'][1][1] - mesh_size))
            nw_box['label'] = bone_str[i]
            bucket[i].append(nw_box)
        if len(bucket[i]) > 1:
            bucket[i] = sorted(bucket[i], key=lambda x : x['confidence'], reverse=True)
    
    result = [x[0] for x in bucket[1:12]]
    return result

def plot_boxes(img, lw, boxes, font=cv2.FONT_HERSHEY_COMPLEX, txt_color=(255, 255, 255), color=(255, 0, 0), use_dot=True):
    for box in boxes:
        p1, p2 = box['xyxy'][0], box['xyxy'][1]
        center = get_center(box)
        center = int(center[0]), int(center[1])
        if use_dot:
            cv2.circle(img, center, 2, color, lw)
        else:
            cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(box['label'], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        #cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        text = box['label'] + ' ' + str(int(box['confidence'] * 100))
        if use_dot:
            cv2.putText(img, text, (center[0] + 4, center[1] + 4), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
        else:
            cv2.putText(img, text, (p1[0] + 4, p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

# The code is based on
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
# Copyright (c) OpenMMLab.


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config(object):

    @staticmethod
    def _file2dict(filename):
        filename = str(filename)
        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                shutil.copyfile(filename,
                                osp.join(temp_config_dir, '_tempconfig.py'))
                sys.path.insert(0, temp_config_dir)
                mod = import_module('_tempconfig')
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules['_tempconfig']
        else:
            raise IOError('Only .py type are supported now!')
        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)