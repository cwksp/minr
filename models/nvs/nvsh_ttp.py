import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from .base_nvs_hypernet import BaseNvsHypernet


@register('nvsh-ttp')
class NvshTtp(BaseNvsHypernet):

    def __init__(self, input_size, patch_size, dtoken_dim, use_viewdirs, ttp_net_spec):
        super().__init__(use_viewdirs)
        self.patch_size = patch_size
        self.prefc = nn.Linear(patch_size**2 * 9, dtoken_dim)
        # self.pos_emb = nn.Parameter(torch.randn(1, (input_size // patch_size)**2, dtoken_dim))
        self.params_generator = models.make(ttp_net_spec, args={'dtoken_dim': dtoken_dim, 'hyponet': self.hyponet})

    def generate_params(self, rays_o, rays_d, imgs):
        B, N, _, H, W = imgs.shape
        x = torch.cat([imgs,
                       rays_o.permute(0, 1, 4, 2, 3),
                       rays_d.permute(0, 1, 4, 2, 3)], dim=2).view(B * N, 9, H, W)
        P = self.patch_size
        x = F.unfold(x, P, stride=P) # (B * N, P * P * 9, (H // P) * (W // P))
        x = x.permute(0, 2, 1).contiguous().view(B, N * (H // P) * (W // P), -1)
        x = self.prefc(x)
        # x = x + self.pos_emb.repeat(1, N, 1)
        params = self.params_generator(x)
        return params
