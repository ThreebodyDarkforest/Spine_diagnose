import torchvision.transforms as transforms
from PIL import Image
import numpy as np

data_transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation((-30, 30)),
        transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0, std=0.1)),
        #transforms.RandomVerticalFlip(),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def add_gaussian_noise(img, mean=0, std=1):
    noise = np.random.randn(img.size[0], img.size[1], 3) * std + mean
    noisy_img = np.array(img, dtype=np.float64) + noise
    return Image.fromarray(np.uint8(noisy_img))