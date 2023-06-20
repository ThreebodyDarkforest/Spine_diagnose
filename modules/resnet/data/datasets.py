import glob, os, ast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from PIL import Image
from resnet.data.util import data_transform

class SpineDataSet(Dataset):
    def __init__(self, path, class_cnt, transforms=None):
        data_paths = sorted(glob.glob(os.path.join(path, '**.jpg')))
        anno_paths = sorted(glob.glob(os.path.join(path, '**.txt')))
        self.transforms = transforms
        self.data_info = []
        self.class_cnt = class_cnt
        for pair in zip(data_paths, anno_paths):
            img_path = pair[0]
            anno_path = pair[1]
            with open(anno_path, 'r') as f:
                ret = f.readline()
                disease_type = ast.literal_eval(ret[2:])
                if len(disease_type) == 0: disease_type.extend([5, 5])
                if len(disease_type) <= 1: disease_type.append(5)
                self.data_info.append([img_path, int(ret[0]), disease_type])
        self.sample_weights, self.class_weights = self._make_weights()
    
    def _make_weights(self):
        count = [0] * self.class_cnt
        for img, type_id, dise_type in self.data_info:
            count[dise_type[0]] += 1
            count[dise_type[1]] += 1
        weight_per_class = [0.] * self.class_cnt
        for i in range(self.class_cnt):
            weight_per_class[i] = float(sum(count)) / float(count[i])
        weight = [0] * len(self.data_info)
        for idx, (img, type_id, dise_type) in enumerate(self.data_info):
            weight[idx] = weight_per_class[dise_type[0]]
        return weight, weight_per_class

    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path, type_id, label = self.data_info[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, type_id, label
    
def get_dataloader(img_dir, class_cnt, batch_size: int = 32, dtype=None, num_workers: int = 0):
    transform = data_transform['val'] if dtype == 'val' else data_transform['train']
    dataset = SpineDataSet(img_dir, class_cnt, transform)
    sampler = WeightedRandomSampler(torch.DoubleTensor(dataset.sample_weights), len(dataset.sample_weights))
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)