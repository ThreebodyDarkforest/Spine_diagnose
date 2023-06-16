import torch
import torchvision
import torch.nn as nn

def load_pretrained(weights, class_num, model_type: str = 'swin_b', device='cpu'):
    ckpt = torch.load(weights, map_location=device)
    model = get_model(model_type, device)
    model.load_state_dict(ckpt)
    inchannel = model.head.in_features
    model.head = nn.Linear(inchannel, class_num * 2)
    return model.to(device)

def load_model(weights, class_num, model_type: str = 'swin_b', device='cpu'):
    ckpt = torch.load(weights, map_location=device)
    model = get_model(model_type, device)
    inchannel = model.head.in_features
    model.head = nn.Linear(inchannel, class_num * 2)
    model.load_state_dict(ckpt)
    return model.to(device)

def get_model(model_type: str = 'swin_b', device='cpu'):
    if model_type == 'swin_b':
        model = torchvision.models.swin_b()
    elif model_type == 'swin_t':
        model = torchvision.models.swin_t()
    else:
        model = torchvision.models.swin_s()
    return model.to(device)

def get_raw_model(model_type: str = 'swin_b', class_num: int = 6, device='cpu'):
    model = get_model(model_type)
    inchannel = model.head.in_features
    model.head = nn.Linear(inchannel, class_num * 2)
    return model.to(device)