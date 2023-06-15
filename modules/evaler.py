import os
import torch
import torch.nn as nn
from util import get_center, distance
from typing import Union, List, Dict

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
}

def estimate_single(result: List[Dict], target: List[Dict], num_class: dict):
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

# TODO: It seems not so easy to caculate ap
def evalutate(results, labels, num_class: dict = num_class, \
              save_dir: str = None, plot_res: bool = False, visualize: bool = True):
    fp, fn, tp, tn = 0, 0, 0, 0
    for result, target in zip(results, labels):
        a, b, c, d = estimate_single(result['result'], target, num_class)
        fp += a
        fn += b
        tp += c
        tn += d
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return { "precision": precision, "recall": recall, "f1_score": f1_score }