import torch
import torchvision
import torch.nn as nn

def load_pretrained(weights, class_num, model_type: str = 'ViT-B-32', device='cpu'):
    ckpt = torch.load(weights, map_location=device)
    model = get_model(model_type, device)
    model.load_state_dict(ckpt)
    inchannel = model.hidden_dim
    model.heads = nn.Linear(inchannel, class_num * 2)
    return model.to(device)

def load_model(weights, class_num, model_type: str = 'ViT-B-32', device='cpu'):
    ckpt = torch.load(weights, map_location=device)
    model = get_model(model_type, device)
    inchannel = model.hidden_dim
    model.heads = nn.Linear(inchannel, class_num * 2)
    model.load_state_dict(ckpt)
    return model.to(device)

def get_model(model_type: str = 'ViT-B-32', device='cpu'):
    if model_type == 'ViT-B-32':
        model = torchvision.models.vit_b_32()
    elif model_type == 'ViT-L-32':
        model = torchvision.models.vit_l_32()
    else:
        model = torchvision.models.vit_b_16()
    return model.to(device)

def get_raw_model(model_type: str = 'ViT-B-32', class_num: int = 6, device='cpu'):
    model = get_model(model_type)
    inchannel = model.hidden_dim
    model.heads = nn.Linear(inchannel, class_num * 2)
    return model.to(device)