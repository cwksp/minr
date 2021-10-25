import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.transformer import TransformerEncoder


@register('ttp-modfc')
class TtpModfc(nn.Module):

    def __init__(self, dtoken_dim, hyponet, n_layers, n_heads, head_dim, ff_hidden_dim, dropout=0.,
                 n_groups=1, num_modfc=-1):
        super().__init__()
        self.hyponet = hyponet

        self.transformer = TransformerEncoder(dtoken_dim, n_layers, n_heads, head_dim, ff_hidden_dim, dropout=dropout)

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
        if num_modfc < 0:
            self.tomod_names = set(self.hyponet.params_shape.keys())
        else:
            self.tomod_names = set(f'wb{i}' for i in range(num_modfc))
        self.ptoken_rng = dict()
        self.ptoken_postfc = nn.ModuleDict()
        n_ptokens = 0
        for name, shape in self.hyponet.params_shape.items():
            if name in self.tomod_names:
                g = min(n_groups, shape[1]) # divide groups, or groups of ones
                assert shape[1] % g == 0
                self.ptoken_postfc[name] = nn.Linear(dtoken_dim, shape[0] - 1)
                self.ptoken_rng[name] = (n_ptokens, n_ptokens + g)
                n_ptokens += g
        self.ptokens = nn.Parameter(torch.randn(n_ptokens, dtoken_dim))

    def forward(self, x):
        """
            x: shape (B, L, dtoken_dim)
        """
        B = x.shape[0]

        # Feed transformer with param tokens
        n_ptokens = len(self.ptokens)
        ptokens = self.ptokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([x, ptokens], dim=1) # (B, n_dtokens + n_ptokens, D)
        outp = self.transformer(x)[:, -n_ptokens:, :] # (B, n_ptokens, D)

        # Translate to params
        params = dict()
        for name, shape in self.hyponet.params_shape.items():
            wb = self.base_params[name].unsqueeze(0).expand(B, -1, -1)

            if name in self.tomod_names:
                ql, qr = self.ptoken_rng[name]
                x = outp[:, ql: qr, :]
                x = self.ptoken_postfc[name](x) # (B, g, (shape[0] - 1))
                x = x.transpose(-1, -2) # (B, (shape[0] - 1), g)
                w, b = wb[:, :-1, :], wb[:, -1:, :]
                w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)
                wb = torch.cat([w, b], dim=1)

            params[name] = wb

        return params
