import glob, os, ast
import torch
from torch.utils.data import Dataset, DataLoader
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

class SpineDataSet(Dataset):
    def __init__(self, path, class_cnt, transforms=None):
        data_paths = sorted(glob.glob(os.path.join(path, '**.jpg')))
        anno_paths = sorted(glob.glob(os.path.join(path, '**.txt')))
        self.transforms = transforms
        self.data_info = []
        for pair in zip(data_paths, anno_paths):
            img_path = pair[0]
            anno_path = pair[1]
            with open(anno_path, 'r') as f:
                ret = f.readline()
                disease_type = ast.literal_eval(ret[2:])
                if len(disease_type) == 0: disease_type.extend([class_cnt - 1, class_cnt - 1])
                if len(disease_type) <= 1: disease_type.append(class_cnt - 1)
                assert max(disease_type) < class_cnt, 'Invalid dataset.'
                self.data_info.append([img_path, disease_type])

    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path, target = self.data_info[idx]
        data = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            data = self.transforms(data)
        return data, target


def get_dataloader(img_dir, class_cnt, batch_size: int = 32, shuffle = False, dtype=None, num_workers: int = 0):
    transform = data_transform['val'] if dtype == 'val' else data_transform['train']
    dataset = SpineDataSet(img_dir, class_cnt, transform)
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

