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


@register('nvs_trainer')
class NVSTrainer(BaseTrainer):
    """
        Novel View Synthesis trainer.
        A data point contains support set (images with poses) and query set (rays).
    """

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
            if k in ['query_rays_o', 'query_rays_d', 'query_imgs']:
                ret[k] = ret[k][:, :min(ret[k].shape[1], 4), ...] ## visualize limit of query
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

    def sample_query_rays_(self, data, n_sample, smart_sample):
        B, N, H, W, _ = data['query_rays_o'].shape
        data['query_rays_o'] = data['query_rays_o'].view(B, N * H * W, 3)
        data['query_rays_d'] = data['query_rays_d'].view(B, N * H * W, 3)
        # query_imgs: (B, N, 3, H, W)
        data['query_gts'] = data['query_imgs'].permute(0, 1, 3, 4, 2).contiguous().view(B, N * H * W, 3)

        if not smart_sample:
            inds = np.random.choice(N * H * W, n_sample, replace=False)
            # inds = [np.random.choice(N * H * W, n_sample, replace=False) for _ in range(B)]
        else:
            # Smart sample: half of GTs are foreground
            inds = []
            fg_n_sample = n_sample // 2
            for i in range(B):
                fg = ((data['query_gts'][i].min(dim=-1).values < 1).nonzero().view(-1)).cpu().numpy()
                if fg_n_sample <= len(fg):
                    fg = np.random.choice(fg, fg_n_sample, replace=False)
                else:
                    fg = np.concatenate([fg, np.random.choice(fg, fg_n_sample - len(fg), replace=True)], axis=0)
                rd = np.random.choice(N * H * W, n_sample - fg_n_sample, replace=False)
                inds.append(np.concatenate([fg, rd], axis=0))

        for k in ['query_rays_o', 'query_rays_d', 'query_gts']:
            data[k] = data[k][:, inds, :]
            # t = torch.empty(B, n_sample, 3, device=data[k].device)
            # for i in range(B):
            #     t[i] = data[k][i][inds[i], :]
            # data[k] = t

    def train_step(self, data):
        """
            data:
                support_rays_o, support_rays_d, support_imgs
                query_rays_o, query_rays_d, query_imgs
                near, far
        }
        """
        data = {k: v.cuda() for k, v in data.items()}
        self.sample_query_rays_(data, n_sample=self.cfg['n_sample_query_rays'], smart_sample=self.cfg.get('smart_sample', False))

        data.pop('query_imgs')
        gts = data.pop('query_gts')

        pred = self.model_ddp(data) # (B, ..., 3)
        loss = F.mse_loss(pred, gts)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def evaluate_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        self.sample_query_rays_(data, n_sample=self.cfg['n_sample_query_rays'], smart_sample=False)

        data.pop('query_imgs')
        gts = data.pop('query_gts')

        B = gts.shape[0]
        with torch.no_grad():
            pred = self.model_ddp(data) # (B, ..., 3)
            mse = (pred - gts).pow(2).view(B, -1).mean(dim=-1)
            loss = mse.mean()
            psnr = (-10 * torch.log10(mse)).mean()

        return {'loss': loss.item(), 'psnr': psnr.item()}

    def evaluate_epoch(self):
        np.random.seed(self.evaluate_seed)
        super().evaluate_epoch()

    def get_visgrid(self, data):
        data = copy.deepcopy(data)
        data = {k: v.cuda() for k, v in data.items()}

        B, Ns, H, W, _ = data['support_rays_o'].shape
        n_support_rays = Ns * H * W
        Nq = data['query_rays_o'].shape[1]
        n_query_rays = Nq * H * W

        all_rays_o = torch.cat([
            data['support_rays_o'].view(B, n_support_rays, 3),
            data['query_rays_o'].view(B, n_query_rays, 3)], dim=1)
        all_rays_d = torch.cat([
            data['support_rays_d'].view(B, n_support_rays, 3),
            data['query_rays_d'].view(B, n_query_rays, 3)], dim=1)

        data['query_rays_o'] = all_rays_o
        data['query_rays_d'] = all_rays_d

        query_imgs = data.pop('query_imgs')
        with torch.no_grad():
            pred = self.model_ddp(data, mode='batched_rendering', batch_size=self.cfg['batched_rendering_size'])
            pred_support = pred[:, :n_support_rays, :].view(B, Ns, H, W, 3).permute(0, 1, 4, 2, 3)
            pred_query = pred[:, n_support_rays:, :].view(B, Nq, H, W, 3).permute(0, 1, 4, 2, 3)

        err_support = (pred_support - data['support_imgs']).pow(2).mean(dim=2, keepdim=True).expand(-1, -1, 3, -1, -1)
        err_support = (err_support * 50).clamp(0, 1)
        err_query = (pred_query - query_imgs).pow(2).mean(dim=2, keepdim=True).expand(-1, -1, 3, -1, -1)
        err_query = (err_query * 50).clamp(0, 1)

        visgrid = torch.cat([
            data['support_imgs'], query_imgs,
            pred_support.clamp(0, 1), pred_query.clamp(0, 1),
            err_support, err_query], dim=1) # (B, (Ns + Nq) * 3, 3, H, W)

        sep_line_rgb = [0.5, 0, 0]
        sep_line_width = 2
        for i in range(3):
            visgrid[:, Ns, i, :, :sep_line_width] = sep_line_rgb[i]

        visgrid = visgrid.view(-1, 3, H, W).detach().cpu()
        visgrid = torchvision.utils.make_grid(visgrid, nrow=Ns + Nq)
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


@register('nvs_ae_trainer')
class NVSAETrainer(NVSTrainer):

    def train_step(self, data):
        data['query_rays_o'] = data['support_rays_o'].clone().detach()
        data['query_rays_d'] = data['support_rays_d'].clone().detach()
        data['query_imgs'] = data['support_imgs'].clone().detach()
        return super().train_step(data)

    def evaluate_step(self, data):
        data['query_rays_o'] = data['support_rays_o'].clone().detach()
        data['query_rays_d'] = data['support_rays_d'].clone().detach()
        data['query_imgs'] = data['support_imgs'].clone().detach()
        return super().evaluate_step(data)
