import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import register


class HypoNeRF(nn.Module):

    def __init__(self, depth=6, hidden_dim=256, use_viewdirs=False):
        super().__init__()
        self.params_shape = dict()
        self.depth = depth
        self.use_viewdirs = use_viewdirs

        last = 3
        last *= 40 # posenc
        for i in range(depth - 1):
            self.params_shape[f'wb{i}'] = (last + 1, hidden_dim) # + 1 for bias
            last = hidden_dim

        if use_viewdirs:
            self.params_shape['wb_sigma'] = (last + 1, 1)

            self.params_shape['wb_mapview'] = (3 + 1, hidden_dim // 2)
            self.params_shape['wb_catview'] = (last + hidden_dim // 2 + 1, hidden_dim)
            self.params_shape['wb_rgb'] = (hidden_dim + 1, 3)
        else:
            self.params_shape['wb_rgbsigma'] = (last + 1, 4) # 4: rgb + sigma

    def convert_posenc(self, x):
        x = torch.cat([
            torch.cat([torch.sin(x * 2**i), torch.cos(x * 2**i)], dim=-1)
            for i in np.linspace(0, 8, 20)
        ], dim=-1)
        return x

    def forward(self, coords, params, viewdirs=None):
        """
        Args:
            coords: shape (B, ..., 3).
            params: dict, value shape (B, NROWS_i, NCOLS_i).
            viewdirs: optional.
        Returns:
            shape (B, ..., 3 + 1): rgb (3) + sigma (1).
        """
        device = coords.device
        B, query_shape = coords.shape[0], coords.shape[1: -1]

        coords = coords.view(B, -1, 3)
        coords = self.convert_posenc(coords)
        if viewdirs is not None:
            viewdirs = viewdirs.view(B, -1, 3)

        x = coords
        for i in range(self.depth - 1):
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = F.relu(torch.bmm(torch.cat([x, one], dim=-1), params[f'wb{i}']))

        if self.use_viewdirs:
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            sigma = F.relu(torch.bmm(torch.cat([x, one], dim=-1), params['wb_sigma']))

            v = F.normalize(viewdirs, dim=-1)
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, v.shape[1], 1)
            v = F.relu(torch.bmm(torch.cat([v, one], dim=-1), params['wb_mapview']))
            x = torch.cat([x, v], dim=-1)
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = F.relu(torch.bmm(torch.cat([x, one], dim=-1), params['wb_catview']))

            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            rgb = torch.sigmoid(torch.bmm(torch.cat([x, one], dim=-1), params['wb_rgb']))
        else:
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = torch.bmm(torch.cat([x, one], dim=-1), params['wb_rgbsigma'])
            rgb, sigma = torch.sigmoid(x[..., :3]), F.relu(x[..., 3:])

        x = torch.cat([rgb, sigma], dim=-1)
        x = x.view(B, *query_shape, -1)
        return x


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


class BaseNerfHypernet(nn.Module):

    def __init__(self, use_viewdirs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.hyponet = HypoNeRF(use_viewdirs=use_viewdirs)

    def generate_params(self, rays_o, rays_d, imgs):
        raise NotImplementedError

    def forward(self, data, mode='default', **kwargs):
        s_rays_o, s_rays_d, s_imgs = data['support_rays_o'], data['support_rays_d'], data['support_imgs']
        params = self.generate_params(s_rays_o, s_rays_d, s_imgs)
        # params = finetune_on_support(params, s_imgs, s_rays_o, s_rays_d, n_iter=200,
        #                              near=float(data['near'][0]), far=float(data['far'][0]), use_viewdirs=self.use_viewdirs)
        q_rays_o, q_rays_d = data['query_rays_o'], data['query_rays_d']

        if mode == 'default':
            pred = utils.render_rays(self.hyponet, q_rays_o, q_rays_d, params=params,
                                     near=float(data['near'][0]), far=float(data['far'][0]), use_viewdirs=self.use_viewdirs)

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
                cur = utils.render_rays(self.hyponet, rays_o, rays_d, params=params,
                                        near=float(data['near'][0]), far=float(data['far'][0]), use_viewdirs=self.use_viewdirs)
                pred.append(cur)
                ql = qr
            pred = torch.cat(pred, dim=1).view(B, *query_shape, 3)

        return pred
