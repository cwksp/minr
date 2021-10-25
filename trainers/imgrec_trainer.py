import os
import os.path as osp
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from torchvision import transforms

from trainers import register
from .base_trainer import BaseTrainer
from utils.siren import make_coord


@register('imgrec_trainer')
class ImgrecTrainer(BaseTrainer):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        if self.is_master:
            os.mkdir(osp.join(self.cfg['save_dir'], 'vis'))
        self.evaluate_seed = self.visualize_seed = 9

    def get_visbatch(self, dataset, n_vis):
        n_vis = min(n_vis, len(dataset))
        ret = dict()
        for i in range(n_vis):
            x = dataset[i]
            for k, v in x.items():
                if ret.get(k) is None:
                    ret[k] = []
                ret[k].append(v)
        for k, v in ret.items():
            ret[k] = torch.stack(v)
        return ret

    def make_datasets(self):
        super().make_datasets()
        np.random.seed(self.visualize_seed)
        n_vis = 8
        if hasattr(self, 'train_loader'):
            self.visbatch_train = self.get_visbatch(self.train_loader.dataset, n_vis)
        if hasattr(self, 'test_loader'):
            self.visbatch_test = self.get_visbatch(self.test_loader.dataset, n_vis)

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        if self.epoch <= round(self.cfg['max_epoch'] * 0.8):
            lr = base_lr
        else:
            lr = base_lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_step(self, data):
        """
            data:
                support_imgs
                query_coords
                gt
        }
        """
        data = {k: v.cuda() for k, v in data.items()}
        gt = data.pop('gt')

        pred = self.model_ddp(data)
        loss = F.mse_loss(pred, gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def evaluate_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        gt = data.pop('gt')

        B = gt.shape[0]
        with torch.no_grad():
            pred = self.model_ddp(data)
            mse = (pred - gt).pow(2).view(B, -1).mean(dim=-1)
            loss = mse.mean()
            psnr = (-10 * torch.log10(mse)).mean()

        return {'loss': loss.item(), 'psnr': psnr.item()}

    def evaluate_epoch(self):
        np.random.seed(self.evaluate_seed)
        super().evaluate_epoch()

    def get_visgrid(self, data):
        data = copy.deepcopy(data)
        data = {k: v.cuda() for k, v in data.items()}

        pred = self.model_ddp(data)
        pred = pred.clamp(0, 1).cpu().permute(0, 3, 1, 2)
        gt = data['gt'].cpu().permute(0, 3, 1, 2)
        visgrid = torch.cat([pred, gt], dim=0)
        visgrid = torchvision.utils.make_grid(visgrid, nrow=len(pred))
        return transforms.ToPILImage()(visgrid)

    def visualize_epoch(self):
        self.model_ddp.eval()

        if hasattr(self, 'visbatch_train'):
            visgrid_train = self.get_visgrid(self.visbatch_train)
            visgrid_train.save(osp.join(self.cfg['save_dir'], 'vis', f'train-epoch{self.epoch}.png'))
            if self.enable_wandb:
                wandb.log({'vis_train_dataset': wandb.Image(visgrid_train)}, step=self.epoch)

        if hasattr(self, 'visbatch_test'):
            visgrid_test = self.get_visgrid(self.visbatch_test)
            visgrid_test.save(osp.join(self.cfg['save_dir'], 'vis', f'test-epoch{self.epoch}.png'))
            if self.enable_wandb:
                wandb.log({'vis_test_dataset': wandb.Image(visgrid_test)}, step=self.epoch)
