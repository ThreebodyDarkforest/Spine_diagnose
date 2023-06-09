import torchvision
import torch
import torch.nn as nn

class Inferer:
    def __init__(self, source, weights, device, cfg) -> None:
        # TODO: wanna add more modality but it's too hard, T_T
        self.device = device
        self.cfg = cfg
        self.model = self.load_model(weights, self.cfg.model_type, device)

    @staticmethod
    def load_model(weights, class_num, model_type: str = 'resnet50', device='cpu'):
        ckpt = torch.load(weights, map_location=device)
        if model_type == 'resnet50':
            model = torchvision.models.resnet50()
        elif model_type == 'resnet34':
            model = torchvision.models.resnet34()
        else:
            model = torchvision.models.resnet18()
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, class_num * 2)
        model.load_state_dict(ckpt)
        return model
    
    def infer(self, classes, save_dir):
        pass