import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import glob, os, ast
from PIL import Image
import numpy as np
import random

PATH = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32

disease_type = ['v1', 'v2', 'v3', 'v4', 'v5', 'None']

data_transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation((-45, 45)),
        transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0, std=0.1)),
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

class SpineLoader(Dataset):
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

def add_gaussian_noise(img, mean=0, std=1):
    noise = np.random.randn(img.size[0], img.size[1], 3) * std + mean
    noisy_img = np.array(img, dtype=np.float64) + noise
    return Image.fromarray(np.uint8(noisy_img))

def train(model: nn.Module, dataloader: DataLoader, eval_dataloader: DataLoader, max_epochs=10, lr=1e-4, device='cpu'):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0

    for epoch in range(max_epochs):
        model.train()
        print(f'epoch {epoch + 1}:')
        train_acc = 0.0
        for step, data in enumerate(dataloader):
            imgs, type_ids, labels = data
            optimizer.zero_grad()
            logits = model(imgs.to(device))
            predict_y = torch.concat((torch.max(logits[:, :6], dim=1)[1], torch.max(logits[:, 6:], dim=1)[1]))
            ret = (predict_y == torch.concat((labels[0].to(device), labels[1].to(device)))).view(2, -1)
            acc = (ret.sum(dim=0) >= 2).sum().item() / type_ids.shape[0]
            train_acc += acc
            loss = loss_fn(logits[:, :6], labels[0].to(device)) + loss_fn(logits[:, 6:], labels[1].to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(dataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain progress: {:^3.0f}%[{}->{}] loss: {:.4f} acc: {:.3f}".format(int(rate * 100), a, b, loss, acc), end="")
        train_acc = train_acc / len(dataloader)
        eval(epoch, loss, train_acc, model, eval_dataloader, device)

def eval(epoch, loss, train_acc, model: nn.Module, dataloader: DataLoader, device='cpu'):
    model.eval()
    acc = 0.0
    with torch.no_grad():
        for data in dataloader:
            imgs, type_ids, labels = data
            logits = model(imgs.to(device))
            predict_y = torch.concat((torch.max(logits[:, :6], dim=1)[1], torch.max(logits[:, 6:], dim=1)[1]))
            #print('predict: ', disease_type[predict_y[0].item()], disease_type[predict_y[1].item()], \
            #      'ground truth: ', disease_type[labels[0].item()], disease_type[labels[1].item()])
            ret = (predict_y == torch.concat((labels[0].to(device), labels[1].to(device)))).view(2, -1)
            acc += (ret.sum(dim=0) >= 2).sum().item() / type_ids.shape[0]
        val_acc = acc / len(dataloader)
        print('\t[epoch %d] train_loss: %.3f train_acc: %.3f test_acc: %.3f' %
          (epoch, loss, train_acc, val_acc))

if __name__ == '__main__':
    model = torchvision.models.swin_v2_s(pretrained=False) # torchvision.models.vit_b_32(pretrained=False)
    weight_path = os.path.join(PATH, 'models/swin_v2_s.pth')
    model.load_state_dict(torch.load(weight_path))
    dataset = SpineLoader(os.path.join(PATH, 'dataset/train'), data_transform['train'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = SpineLoader(os.path.join(PATH, 'dataset/val'), data_transform['val'])
    val_dataloader = DataLoader(val_dataset)

    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    inchannel = model.head.in_features
    model.head = nn.Linear(inchannel, 12)
    model.to(device)
    train(model, dataloader, val_dataloader, 30, 2e-5, device=device)
    #eval(1, 1, model, val_dataloader, device)