import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import utils
from models.nerf import HypoNeRF
from models.pe_mlp import HypoPeMlp


def finetune_on_support(init_params, imgs, rays_o, rays_d, n_iter, near, far, use_viewdirs, batch_size=1024):
    """
        imgs: shape (B, N, 3, H, W)
        rays_o, rays_d: shape (B, N, H, W, 3)
    """
    B, N, H, W, _ = rays_o.shape
    gts = imgs.permute(0, 1, 3, 4, 2).view(B, N * H * W, 3)
    rays_o = rays_o.view(B, N * H * W, 3)
    rays_d = rays_d.view(B, N * H * W, 3)

    hyponerf = HypoNeRF(use_viewdirs=use_viewdirs)
    params = dict()
    params_lst = []
    for k, v in init_params.items():
        p = nn.Parameter(v.detach().clone())
        params_lst.append(p)
        params[k] = p
    optimizer = torch.optim.Adam(params_lst, lr=1e-4)

    with torch.enable_grad():
        for i_iter in range(n_iter):
            inds = np.random.choice(N * H * W, batch_size, replace=False)
            rays_o_, rays_d_, gts_ = rays_o[:, inds, :], rays_d[:, inds, :], gts[:, inds, :]
            pred = utils.render_rays(hyponerf, rays_o_, rays_d_, params=params,
                                     near=near, far=far, use_viewdirs=use_viewdirs)
            loss = F.mse_loss(pred, gts_)
            if i_iter % (n_iter // 10) == 0:
                print(i_iter, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = loss = None

    for k, v in params.items():
        params[k] = v.data
    return params


class BaseNvsHypernet(nn.Module):

    def __init__(self, hyponet_spec):
        super().__init__()
        self.hyponet_name = hyponet_spec['name']
        self.hyponet = {
            'nerf': HypoNeRF,
            'pe_mlp': HypoPeMlp,
        }[self.hyponet_name](**hyponet_spec['args'])

    def generate_params(self, rays_o, rays_d, imgs):
        raise NotImplementedError

    def forward(self, data, mode='default', **kwargs):
        s_rays_o, s_rays_d, s_imgs = data['support_rays_o'], data['support_rays_d'], data['support_imgs']
        params = self.generate_params(s_rays_o, s_rays_d, s_imgs)
        # params = finetune_on_support(params, s_imgs, s_rays_o, s_rays_d, n_iter=200, near=float(data['near'][0]), far=float(data['far'][0]), use_viewdirs=self.use_viewdirs)
        q_rays_o, q_rays_d = data['query_rays_o'], data['query_rays_d']

        if mode == 'default':
            if self.hyponet_name == 'nerf':
                pred = utils.render_rays(self.hyponet, q_rays_o, q_rays_d, params=params,
                                        near=float(data['near'][0]), far=float(data['far'][0]), use_viewdirs=self.hyponet.use_viewdirs)
            elif self.hyponet_name == 'pe_mlp':
                pred = self.hyponet(torch.cat([q_rays_o, q_rays_d], dim=-1), params)

        elif mode == 'batched_rendering':
            B = q_rays_o.shape[0]
            query_shape = q_rays_o.shape[1: -1]
            q_rays_o = q_rays_o.view(B, -1, 3)
            q_rays_d = q_rays_d.view(B, -1, 3)
            tot_rays = q_rays_o.shape[1]
            bs_rays = kwargs['batch_size']
            pred = []
            ql = 0
            while ql < tot_rays:
                qr = min(ql + bs_rays, tot_rays)
                rays_o = q_rays_o[:, ql: qr, :].contiguous()
                rays_d = q_rays_d[:, ql: qr, :].contiguous()
                if self.hyponet_name == 'nerf':
                    cur = utils.render_rays(self.hyponet, rays_o, rays_d, params=params,
                                            near=float(data['near'][0]), far=float(data['far'][0]), use_viewdirs=self.hyponet.use_viewdirs)
                elif self.hyponet_name == 'pe_mlp':
                    cur = self.hyponet(torch.cat([rays_o, rays_d], dim=-1), params)
                pred.append(cur)
                ql = qr
            pred = torch.cat(pred, dim=1).view(B, *query_shape, 3)

        return pred
