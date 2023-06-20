from resnet.data import datasets
from resnet.data.util import data_transform
from .util import get_raw_model, load_model
import torch
import torch.nn as nn
import torch.optim as optim
from resnet.data.datasets import get_dataloader
import os
from components.logger import LOGGER
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, train_dir: str, val_dir: str, class_num: int = 6, model_name: str = 'resnet50', num_workers: int = 0, \
                 optimizer: str = 'Adam', batch_size: int = 32, model: nn.Module = None, device='cpu') -> None:
        assert optimizer in ['SGD', 'Adam'], 'Invalid optimizer.'
        self.train_data = get_dataloader(train_dir, class_num, batch_size, num_workers=num_workers)
        self.eval_data = get_dataloader(val_dir, class_num, batch_size, num_workers=num_workers)
        if model is None:
            self.model = get_raw_model(model_name, class_num, device)
        else:
            self.model = model.to(device)
        self.device = device
        self.optim = optimizer
        self.model_name = model_name
        self.tblogger = None
    
    def train(self, max_epochs=10, lr=1e-4, save=False, \
              save_every=5, save_best=False, save_path: str = None):
        running_loss, max_acc = 0.0, 0.0
        loss_fn = nn.CrossEntropyLoss()
        self.tblogger = SummaryWriter(save_path)
        optimizer = optim.Adam(self.model.parameters(), lr=lr) \
                    if self.optim == 'Adam' else optim.SGD(self.model.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 3, eta_min=1e-5, last_epoch=-1)
        
        for epoch in range(max_epochs):
            self.model.train()
            print(f'epoch {epoch + 1}/{max_epochs}:')
            train_acc = 0.0
            for step, data in enumerate(self.train_data):
                imgs, type_ids, labels = data
                optimizer.zero_grad()
                logits = self.model(imgs.to(self.device))
                predict_y = torch.concat((torch.max(logits[:, :6], dim=1)[1], torch.max(logits[:, 6:], dim=1)[1]))
                ret = (predict_y == torch.concat((labels[0].to(self.device), labels[1].to(self.device)))).view(2, -1)
                acc = (ret.sum(dim=0) >= 2).sum().item() / type_ids.shape[0]
                train_acc += acc
                loss = loss_fn(logits[:, :6], labels[0].to(self.device)) + loss_fn(logits[:, 6:], labels[1].to(self.device))
                loss.backward()
                optimizer.step()
                #lr_scheduler.step()

                running_loss += loss.item()
                rate = (step + 1) / len(self.train_data)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtrain progress: {:^3.0f}%[{}->{}] loss: {:.4f} acc: {:.3f}".format(int(rate * 100), a, b, loss, acc), end="")
            train_acc = train_acc / len(self.train_data)

            val_acc = self.eval()

            # tensorboard summary
            self.tblogger.add_scalar('train_acc', train_acc, epoch)
            self.tblogger.add_scalar('val_acc', val_acc, epoch)
            self.tblogger.add_scalar('loss', running_loss / len(self.train_data), epoch)

            LOGGER.info(f'\t[epoch {epoch}] val_acc: {val_acc} train_loss: {running_loss / len(self.train_data)}')
            running_loss = 0.0
            if max_acc < val_acc and save_best:
                LOGGER.info(f'Saving best checkpoint to {save_path}')
                max_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.model_name}_best.pt'))
            if (epoch + 1) % save_every == 0 and save:
                LOGGER.info(f'Saving checkpoint to {save_path}')
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.model_name}_{epoch + 1}.pt'))

    def eval(self):
        self.model.eval()
        acc = 0.0
        with torch.no_grad():
            for data in self.eval_data:
                imgs, type_ids, labels = data
                logits = self.model(imgs.to(self.device))
                predict_y = torch.concat((torch.max(logits[:, :6], dim=1)[1], torch.max(logits[:, 6:], dim=1)[1]))
                #print('predict: ', disease_type[predict_y[0].item()], disease_type[predict_y[1].item()], \
                #      'ground truth: ', disease_type[labels[0].item()], disease_type[labels[1].item()])
                ret = (predict_y == torch.concat((labels[0].to(self.device), labels[1].to(self.device)))).view(2, -1)
                acc += (ret.sum(dim=0) >= 2).sum().item() / type_ids.shape[0]
            val_acc = acc / len(self.eval_data)
        return val_acc