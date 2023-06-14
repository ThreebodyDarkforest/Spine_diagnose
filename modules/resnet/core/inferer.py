import torchvision
import torch
import torch.nn as nn
from .util import load_model

class Inferer:
    def __init__(self, source, weights, device, cfg) -> None:
        # TODO: wanna add more modality but it's too hard, T_T
        self.device = device
        self.cfg = cfg
        self.model = load_model(weights, self.cfg.model_type, device)
    
    def infer(self, classes, save_dir):
        pass