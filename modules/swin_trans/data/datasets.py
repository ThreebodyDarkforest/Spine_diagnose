import glob, os, ast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from resnet.data.util import data_transform

class SpineDataSet(Dataset):
    def __init__(self, path, transforms=None):
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
                if len(disease_type) == 0: disease_type.extend([5, 5])
                if len(disease_type) <= 1: disease_type.append(5)
                self.data_info.append([img_path, int(ret[0]), disease_type])
    
    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path, type_id, label = self.data_info[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, type_id, label
    
def get_dataloader(img_dir, batch_size: int = 32, dtype=None, num_workers: int = 0):
    transform = data_transform['val'] if dtype == 'val' else data_transform['train']
    dataset = SpineDataSet(img_dir, transform)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)