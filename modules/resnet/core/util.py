import torch
import torchvision
import torch.nn as nn

def load_pretrained(weights, class_num, model_type: str = 'resnet50', device='cpu'):
    ckpt = torch.load(weights, map_location=device)
    model = get_model(model_type, device)
    model.load_state_dict(ckpt)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, class_num * 2)
    return model.to(device)

def load_model(weights, class_num, model_type: str = 'resnet50', device='cpu'):
    ckpt = torch.load(weights, map_location=device)
    model = get_model(model_type, device)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, class_num * 2)
    model.load_state_dict(ckpt)
    return model.to(device)

def get_model(model_type: str = 'resnet50', device='cpu'):
    if model_type == 'resnet50':
        model = torchvision.models.resnet50()
    elif model_type == 'resnet34':
        model = torchvision.models.resnet34()
    else:
        model = torchvision.models.resnet18()
    return model.to(device)