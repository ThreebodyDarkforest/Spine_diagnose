import torch
from components.logger import LOGGER
from yolov6.utils.events import load_yaml
from resnet.core.inferer import Inferer
from resnet.data.datasets import data_transform
import torch.nn as nn
from typing import Union, List, Optional
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from resnet.core import util as resnet_util
from vit.core import util as vit_util
from swin_trans.core import util as swin_util

transform = data_transform['val']

def get_resnet_model(weightsA, weightsB, config_path, half: bool = False, \
                     pretrained: bool = False, device = 'cpu'):
    assert '.pt' in weightsA and '.pt' in weightsB, 'Invalid model path.'
    _dict = load_yaml(config_path)
    model_type = _dict['classify']
    class_namesA, class_numA = _dict['dnames1'], _dict['dnc1']
    class_namesB, class_numB = _dict['dnames2'], _dict['dnc2']
    if not pretrained:
        modelA = resnet_util.load_model(weightsA, class_numA, model_type, device)
        modelB = resnet_util.load_model(weightsB, class_numB, model_type, device)
    else:
        modelA = resnet_util.load_pretrained(weightsA, class_numA, model_type, device)
        modelB = resnet_util.load_pretrained(weightsB, class_numB, model_type, device)
    return modelA, class_namesA, modelB, class_namesB

def get_vit_model(weights, config_path, half: bool = False, \
                     pretrained: bool = False, device = 'cpu'):
    assert '.pt' in weights, 'Invalid model path.'
    _dict = load_yaml(config_path)
    model_type = _dict['classify']
    class_names, class_num = _dict['dnames'], _dict['dnc']
    if not pretrained:
        model = vit_util.load_model(weights, class_num, model_type, device)
    else:
        model = vit_util.load_pretrained(weights, class_num, model_type, device)
    return model, class_names

def get_swin_model(weights, config_path, half: bool = False, \
                     pretrained: bool = False, device = 'cpu'):
    assert '.pt' in weights, 'Invalid model path.'
    _dict = load_yaml(config_path)
    model_type = _dict['classify']
    class_names, class_num = _dict['dnames'], _dict['dnc']
    if not pretrained:
        model = swin_util.load_model(weights, class_num, model_type, device)
    else:
        model = swin_util.load_pretrained(weights, class_num, model_type, device)
    return model, class_names

def classify(model: nn.Module, img: Union[str, np.ndarray], class_names: List[str], \
             model_type: str = 'resnet', plot: bool = False, device = 'cpu'):
    '''Classify the object to particular classes.

    Please ensure that your input image is 
    appropriately preprocessed to resemble a vertebra.

    Args:
        model: (nn.Module), the model to use, note that only supports resnet type now.
        img: (Union[str, np.ndarray]), the input image to classify
        class_names: (List[str]), the list of class names
        model_type: (str), the type of the model, only supports 'resnet' now
        plot: (bool), whether to plot the image or not
        device: (str), the device to use, default is 'cpu'
    
    Returns:
        label: (Tuple[str, str]), the predicted label of the input image
        softmax: (Tuple[float, float]), the softmax probability of the predicted label
    '''
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
        predict_y = (torch.max(logits[:, :len(class_names)], dim=1)[1].item(), torch.max(logits[:, len(class_names):], dim=1)[1].item())
        label = class_names[predict_y[0]], class_names[predict_y[1]]
        conf = torch.concat((F.softmax(logits[:, :len(class_names)], dim=1), F.softmax(logits[:, len(class_names):], dim=1)), dim=1).tolist()[0]
        precision = (torch.max(F.softmax(logits[:, :len(class_names)], dim=1), dim=1)[0].item(), torch.max(F.softmax(logits[:, len(class_names):], dim=1), dim=1)[0].item())
        return label, precision, conf
    else:
        pass

if __name__ == '__main__':
    pass