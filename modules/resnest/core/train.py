##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import os
import time
import json
import logging
import argparse
from resnest.data.datasets import get_dataloader, SpineDataSet

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torch.optim as optim

from resnest.torch.config import get_cfg
from resnest.torch.models.build import get_model
from resnest.torch.datasets import get_dataset
from resnest.data.datasets import data_transform

from resnest.torch.loss import get_criterion
from resnest.torch.utils import (save_checkpoint, accuracy,
        AverageMeter, LR_Scheduler, torch_dist_sum, mkdir,
        cached_log_stream, PathManager)

from components.logger import LOGGER

class Trainer():
    def __init__(self, train_dir: str, val_dir: str, class_num: int = 6, model_name: str = 'ViT-B-32', num_workers: int = 0, \
                 optimizer: str = 'Adam', final_drop=0.0, last_gamma=False, batch_size: int = 32, model: nn.Module = None, device='cpu') -> None:
        assert optimizer in ['SGD', 'Adam'], 'Invalid optimizer.'
        self.train_data = get_dataloader(train_dir, class_num, batch_size, shuffle=True, num_workers=num_workers)
        self.eval_data = get_dataloader(val_dir, class_num, batch_size, num_workers=num_workers)
        
        model_kwargs = {}
        if final_drop > 0.0:
            model_kwargs['final_drop'] = final_drop

        if last_gamma:
            model_kwargs['last_gamma'] = True
        model_kwargs['num_classes'] = class_num * 2
        
        if model is None:
            self.model = get_model(model_name)(**model_kwargs).to(device)
        else:
            self.model = model.to(device)
        self.device = device
        self.optim = optimizer
        self.model_name = model_name
        self.class_num = class_num
        self.best_pred = 0.
        self.acclist_train = []
        self.acclist_val = []
    
    def train(self, max_epochs=10, lr=1e-4, save=False, \
              weight_decay=0, momentum=0, lr_scheduler='cos', warmup_epochs=0, \
              save_every=5, save_best=False, save_path: str = None):
        running_loss, max_acc = 0., 0.
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) \
                    if self.optim == 'Adam' else optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = LR_Scheduler(lr_scheduler, 
                                 base_lr=lr, 
                                 num_epochs=max_epochs, 
                                 iters_per_epoch=len(self.train_data),
                                 warmup_epochs=warmup_epochs)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for epoch in range(max_epochs):
            #self.train_sampler.set_epoch(epoch)
            self.model.train()
            print(f'epoch {epoch + 1}/{max_epochs}:')
            train_acc = 0.0
            for step, data in enumerate(self.train_data):
                imgs, labels = data
                scheduler(optimizer, step, epoch, self.best_pred)
                optimizer.zero_grad()
                logits = self.model(imgs.to(self.device))
                loss = loss_fn(logits[:, :self.class_num], labels[0].to(self.device)) + loss_fn(logits[:, self.class_num:], labels[1].to(self.device))
                loss.backward()
                optimizer.step()

                predict_y = torch.concat((torch.max(logits[:, :self.class_num], dim=1)[1], torch.max(logits[:, self.class_num:], dim=1)[1]))
                ret = (predict_y == torch.concat((labels[0].to(self.device), labels[1].to(self.device)))).view(2, -1)
                acc = (ret.sum(dim=0) >= 2).sum().item() / imgs.size(0)
                train_acc += acc
                running_loss += loss.item()

                rate = (step + 1) / len(self.train_data)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtrain progress: {:^3.0f}%[{}->{}] loss: {:.4f} acc: {:.3f}".format(int(rate * 100), a, b, loss, acc), end="")
            
            self.acclist_train += [train_acc / len(self.train_data)]

            val_acc = self.eval()
            LOGGER.info(f'\t[epoch {epoch}] val_acc: {val_acc} train_loss: {running_loss / len(self.train_data)}')
            running_loss = 0.0
            if max_acc < val_acc and save_best:
                LOGGER.info(f'Saving best checkpoint to {save_path}')
                max_acc = val_acc
                '''save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': self.best_pred,
                    'acclist_train': self.acclist_train,
                    'acclist_val': self.acclist_val,
                },
                directory=save_path,
                is_best=False,
                filename=os.path.join(save_path, f'{self.model_name}_best.pt'))'''
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.model_name}_best.pt'))
            if (epoch + 1) % save_every == 0 and save:
                LOGGER.info(f'Saving checkpoint to {save_path}')
                '''save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': self.best_pred,
                    'acclist_train': self.acclist_train,
                    'acclist_val': self.acclist_val,
                },
                directory=save_path,
                is_best=False,
                filename=os.path.join(save_path, f'{self.model_name}_{epoch + 1}.pt'))'''
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.model_name}_{epoch + 1}.pt'))
    
    def eval(self):
        self.model.eval()
        acc = 0.0
        with torch.no_grad():
            for step, data in enumerate(self.eval_data):
                imgs, labels = data
                logits = self.model(imgs.to(self.device))
                predict_y = torch.concat((torch.max(logits[:, :self.class_num], dim=1)[1], torch.max(logits[:, self.class_num:], dim=1)[1]))
                #print('predict: ', disease_type[predict_y[0].item()], disease_type[predict_y[1].item()], \
                #      'ground truth: ', disease_type[labels[0].item()], disease_type[labels[1].item()])
                ret = (predict_y == torch.concat((labels[0].to(self.device), labels[1].to(self.device)))).view(2, -1)
                acc += (ret.sum(dim=0) >= 2).sum().item() / imgs.size(0)
            val_acc = acc / len(self.eval_data)
        self.acclist_val += [val_acc]
        return val_acc