import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .base_nerf_hypernet import BaseNerfHypernet


@register('nh-transformer_modfc')
class NHTransformerModfc(BaseNerfHypernet):

    def __init__(self, patch_size, d_model, nhead, dim_feedforward, dropout, num_layers, n_groups, use_viewdirs, num_modfc):
        super().__init__(use_viewdirs=use_viewdirs)
        self.patch_size = patch_size
        img_channels = 3

        # Pre FC
        self.patch_prefc = nn.Linear(self.patch_size * self.patch_size * (img_channels + 6), d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Base params
        self.base_params = dict()
        for name, shape in self.hyponet.params_shape.items():
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

            bias = torch.empty(shape[1], 1)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

            p = nn.Parameter(torch.cat([weight, bias], dim=1).t().detach())
            self.base_params[name] = p
            self.register_parameter(f'base_params_{name}', p)

        # Define modulate vectors
        self.num_modfc = num_modfc
        self.ptoken_rng = dict()
        self.ptoken_postfc = nn.ModuleDict()
        n_ptokens = 0
        for name, shape in self.hyponet.params_shape.items():
            if name == 'wb_rgbsigma' or int(name[-1]) >= self.num_modfc:
                continue
            g = min(n_groups, shape[1]) # group or all
            assert shape[1] % g == 0
            self.ptoken_postfc[name] = nn.Linear(d_model, shape[0] - 1)
            self.ptoken_rng[name] = (n_ptokens, n_ptokens + g)
            n_ptokens += g
        self.ptokens = nn.Parameter(torch.randn(n_ptokens, d_model))

    def generate_params(self, rays_o, rays_d, imgs):
        """
            imgs: shape (B, N, 3, H, W)
            rays_o, rays_d: shape (B, N, H, W, 3)
        """
        B, N, _, H, W = imgs.shape

        # Pre FC to tokens
        x = torch.cat([imgs,
                       rays_o.permute(0, 1, 4, 2, 3),
                       rays_d.permute(0, 1, 4, 2, 3)], dim=2).view(B * N, -1, H, W)
        P = self.patch_size
        x = F.unfold(x, P, stride=P) # (B * N, P * P * 9, (H // P) * (W // P))
        x = x.permute(0, 2, 1).contiguous().view(B, N * (H // P) * (W // P), -1)
        x = x.permute(1, 0, 2) # (n_patches, B, P * P * 9)
        x = self.patch_prefc(x)

        # Feed transformer with param tokens
        n_ptokens = len(self.ptokens)
        ptokens = self.ptokens.unsqueeze(1).expand(-1, B, -1)
        x = torch.cat([ptokens, x], dim=0) # (n_ptokens + n_patches, B, D)
        outp = self.transformer(x)[:n_ptokens] # (n_ptokens, B, D)

        # Translate to params
        params = dict()
        for name, shape in self.hyponet.params_shape.items():
            wb = self.base_params[name].unsqueeze(0).expand(B, -1, -1)

            if name == 'wb_rgbsigma' or int(name[-1]) >= self.num_modfc:
                pass
            else:
                ql, qr = self.ptoken_rng[name]
                x = self.ptoken_postfc[name](outp[ql: qr]) # (g, B, (shape[0] - 1)); g = min(shape[1], n_groups)
                x = x.permute(1, 2, 0) # (B, (shape[0] - 1), g)
                w, b = wb[:, :-1, :], wb[:, -1:, :]
                w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)
                wb = torch.cat([w, b], dim=1)

            params[name] = wb

        return params
