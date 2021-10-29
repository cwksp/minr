import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from .base_nvs_hypernet import BaseNvsHypernet


@register('nvsh-ttp')
class NvshTtp(BaseNvsHypernet):

    def __init__(self, input_size, patch_size, dtoken_dim, hyponet_spec, ttp_net_spec, use_pos_emb=False):
        super().__init__(hyponet_spec)
        self.patch_size = patch_size
        self.prefc = nn.Linear(patch_size**2 * 9, dtoken_dim)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, (input_size // patch_size)**2, dtoken_dim))
        self.ttp_net = models.make(ttp_net_spec, args={'dtoken_dim': dtoken_dim, 'hyponet': self.hyponet})

    def generate_params(self, rays_o, rays_d, imgs):
        B, N, _, H, W = imgs.shape
        x = torch.cat([imgs,
                       rays_o.permute(0, 1, 4, 2, 3),
                       rays_d.permute(0, 1, 4, 2, 3)], dim=2).view(B * N, 9, H, W)
        P = self.patch_size
        x = F.unfold(x, P, stride=P) # (B * N, P * P * 9, (H // P) * (W // P))
        x = x.permute(0, 2, 1).contiguous().view(B, N * (H // P) * (W // P), -1)
        x = self.prefc(x)
        if self.use_pos_emb:
            x = x + self.pos_emb.repeat(1, N, 1)
        params = self.ttp_net(x)
        return params


@register('nvsh-ttp_camtoken')
class NvshTtpCamtoken(BaseNvsHypernet):

    def __init__(self, input_size, patch_size, dtoken_dim, n_camtokens, hyponet_spec, ttp_net_spec):
        super().__init__(hyponet_spec)
        self.dtoken_dim = dtoken_dim
        self.n_camtokens = n_camtokens
        self.patch_size = patch_size
        self.prefc = nn.Linear(patch_size**2 * 3, dtoken_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, (input_size // patch_size)**2, dtoken_dim))
        self.cam_fc = nn.Linear(6, dtoken_dim * n_camtokens)
        self.ttp_net = models.make(ttp_net_spec, args={'dtoken_dim': dtoken_dim, 'hyponet': self.hyponet})

    def generate_params(self, rays_o, rays_d, imgs):
        B, N, _, H, W = imgs.shape
        assert N == 1
        P = self.patch_size
        x = F.unfold(imgs.view(B, 3, H, W), P, stride=P).permute(0, 2, 1).contiguous() # (B, (H // P) * (W // P), P * P * 3)
        x = self.prefc(x) + self.pos_emb

        ch, cw = H // 2, W // 2
        cam = torch.cat([rays_o[:, 0, ch, cw, :], rays_d[:, 0, ch, cw, :]], dim=-1)
        camtokens = self.cam_fc(cam).view(B, self.n_camtokens, self.dtoken_dim) # (B, n_camtokens, dtoken_dim)
        x = torch.cat([x, camtokens], dim=1)

        params = self.ttp_net(x)
        return params
