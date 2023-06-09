import torch
from yolov6.utils.events import LOGGER, load_yaml
from resnet.core.inferer import Inferer
from resnet.data.datasets import data_transform
import torch.nn as nn
from typing import Union, List, Optional
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F

transform = data_transform['val']

def get_resnet_model(weights, config_path, half: bool = False, device = 'cpu'):
    assert '.pt' in weights, 'Invalid model path.'
    _dict = load_yaml(config_path)
    model_type = _dict['classify']
    class_names, class_num = _dict['dnames'], _dict['dnc']
    model = Inferer.load_model(weights, class_num, model_type, device).to(device)
    return model, class_names

def classify(model: nn.Module, img: Union[str, np.ndarray], class_names: List[str], \
             model_type: str = 'resnet', plot: bool = False, device = 'cpu'):
    model_type = model_type.lower()
    assert model_type in ['resnet', 'resnest', 'vit', 'swin']

    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = transform(img).to(device)
    if len(img.shape) == 3:
        img = img[None]
        
    model.eval()

    if model_type == 'resnet':
        logits = model(img)
        predict_y = (torch.max(logits[:, :6], dim=1)[1].item(), torch.max(logits[:, 6:], dim=1)[1].item())
        label = class_names[predict_y[0]], class_names[predict_y[1]]
        return label, (torch.max(F.softmax(logits[:, :6]), dim=1)[0].item(), torch.max(F.softmax(logits[:, 6:]), dim=1)[0].item())
    else:
        pass

if __name__ == '__main__':
    pass