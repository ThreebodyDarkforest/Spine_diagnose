import os
import torch
import torch.nn as nn
from util import get_center, distance
from typing import Union, List, Dict
from sklearn.metrics import average_precision_score
import numpy as np

# It's stupid here.
num_class = {
    "L1" : 0,
    "L2" : 1,
    "L3" : 2,
    "L4" : 3, 
    "L5" : 4,
    "T12-L1" : 5,
    "L1-L2" : 6,
    "L2-L3" : 7,
    "L3-L4" : 8,
    "L4-L5" : 9,
    "L5-S1" : 10,
    "T11-T12": 5,
    "Anchor": 11,
    "All": 12,
}

disease_class = {
    "v1": 0,
    "v2": 1,
    "v3": 2,
    "v4": 3,
    "v5": 4,
    "None": 5,
}

def estimate_single_depracated(result: List[Dict], target: List[Dict], num_class: dict):
    fp, fn, tp, tn = 0, 0, 0, 0
    assert len(result) == 11 or len(target) == 11, 'Invalid label input.'
    result = sorted(result, key=lambda x : num_class[x['bone_type']])
    target = sorted(target, key=lambda x : num_class[x['bone_type']])
    for predict, label in zip(result, target):
        if predict['bone_type'] == label['bone_type'] and \
           predict['disease_type'] == label['disease_type'] \
           and distance(predict['coord'], label['coord']) <= 8:
            tp += 1
        else:
            fp += 1
            fn += 1
    return fp, fn, tp, tn

def estimate_single(result: List[Dict], target: List[Dict], num_class: dict, disease_class: dict):
    fp, fn, tp, tn = 0, 0, 0, 0
    vis = [int(1e9) for _ in range(11)]
    res = [None for _ in range(11)]

    bone_one_hot = np.zeros((0, len(num_class.keys())), dtype=np.int8)
    disea_one_hot = np.zeros((0, len(disease_class.keys())), dtype=np.int8)
    diseb_one_hot = np.zeros((0, len(disease_class.keys())), dtype=np.int8)

    bone_pred = np.array([None for _ in range(len(num_class.keys()))], dtype=np.float32)
    disea_pred = np.array([None for _ in range(len(disease_class.keys()))], dtype=np.float32)
    diseb_pred = np.array([None for _ in range(len(disease_class.keys()))], dtype=np.float32)

    for i, label in enumerate(target):
        for predict in result:
            if predict['bone_type'] == label['bone_type'] and \
               predict['disease_type'] == label['disease_type'] \
               and distance(predict['coord'], label['coord']) <= 8:
                if vis[num_class[predict['bone_type']]] < distance(predict['coord'], label['coord']): continue
                if vis[num_class[predict['bone_type']]] == int(1e9): tp += 1
                vis[num_class[predict['bone_type']]] = distance(predict['coord'], label['coord'])
                res[i] = predict
            
            if distance(predict['coord'], label['coord']) <= 8:
                bone_one_hot = np.vstack((bone_one_hot,  np.eye(len(num_class.keys()))[num_class[label['bone_type']]]))
                dise_label = [_.strip() for _ in label['disease_type'].split(',')]
                [dise_label.append('None') for _ in range(2 - len(dise_label))]
                disea_one_hot = np.vstack((disea_one_hot,  np.eye(len(disease_class.keys()))[disease_class[dise_label[0]]]))
                diseb_one_hot = np.vstack((diseb_one_hot,  np.eye(len(disease_class.keys()))[disease_class[dise_label[1]]]))

                bone_pred = np.vstack((bone_pred, np.array(predict['detect_logits'], dtype=np.float32)))
                disea_pred = np.vstack((disea_pred, np.array(predict['classify_logits'][0], dtype=np.float32)))
                diseb_pred = np.vstack((diseb_pred, np.array(predict['classify_logits'][1], dtype=np.float32)))

    for predict in result:
        flag = False
        for label in target:
            if predict['bone_type'] == label['bone_type'] and \
               predict['disease_type'] == label['disease_type'] \
               and distance(predict['coord'], label['coord']) <= 8:
                flag = True
        if not flag: fp += 1
    for _ in res:
        fn += 1 if _ is None else 0
    
    return fp, fn, tp, tn, {
        'bone': (bone_one_hot, bone_pred[1:]),
        'disea': (disea_one_hot, disea_pred[1:]),
        'diseb': (diseb_one_hot, diseb_pred[1:]),
    }

# TODO: It seems not so easy to caculate ap
def evalutate(results, labels, num_class: dict = num_class, disease_class: dict = disease_class, \
              save_dir: str = None, plot_res: bool = False, visualize: bool = True):
    fp, fn, tp, tn = 0, 0, 0, 0

    bone_one_hot = np.zeros((0, len(num_class.keys())), dtype=np.int8)
    disea_one_hot = np.zeros((0, len(disease_class.keys())), dtype=np.int8)
    diseb_one_hot = np.zeros((0, len(disease_class.keys())), dtype=np.int8)

    bone_pred = np.zeros((0, len(num_class.keys())), dtype=np.float32)
    disea_pred = np.zeros((0, len(disease_class.keys())), dtype=np.float32)
    diseb_pred = np.zeros((0, len(disease_class.keys())), dtype=np.float32)

    for i, (result, target) in enumerate(zip(results, labels)):
        a, b, c, d, estim = estimate_single(result['result'], target, num_class, disease_class)
        fp += a
        fn += b
        tp += c
        tn += d

        bone_one_hot = np.vstack((bone_one_hot, estim['bone'][0]))
        disea_one_hot = np.vstack((disea_one_hot, estim['disea'][0]))
        diseb_one_hot = np.vstack((diseb_one_hot, estim['diseb'][0]))

        bone_pred = np.vstack((bone_pred, estim['bone'][1]))
        disea_pred = np.vstack((disea_pred, estim['disea'][1]))
        diseb_pred = np.vstack((diseb_pred, estim['diseb'][1]))
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    #print(bone_one_hot)
    #print(bone_pred)

    # calculate ap
    bone_ap = average_precision_score(bone_one_hot, bone_pred, average='micro')
    disea_ap = average_precision_score(disea_one_hot, disea_pred, average='micro')
    diseb_ap = average_precision_score(diseb_one_hot, diseb_pred, average='micro')

    mAP = (bone_ap + disea_ap + diseb_ap) / 3

    return { 
        "precision": precision, 
        "recall": recall, 
        "f1_score": f1_score, 
        'bone_AP': bone_ap,
        'disease_AP': (disea_ap, diseb_ap),
        'mAP': mAP,
    }