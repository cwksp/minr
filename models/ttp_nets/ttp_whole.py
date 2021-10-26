import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.transformer import TransformerEncoder, TransformerDecoder


@register('ttp-whole')
class TtpWhole(nn.Module):

    def __init__(self, dtoken_dim, hyponet, n_htokens, tokens_transformer_args, params_transformer_args):
        super().__init__()
        self.hyponet = hyponet

        self.htokens = nn.Parameter(torch.randn(n_htokens, dtoken_dim))
        self.tokens_transformer = TransformerEncoder(dim=dtoken_dim, **tokens_transformer_args)

        self.params_init = nn.ParameterDict()
        self.params_transformer = nn.ModuleDict()
        self.params_convertfc = nn.ModuleDict()

        self.base_params = nn.ParameterDict()
        self.k_base = nn.ParameterDict()
        self.k_gen = nn.ParameterDict()

        for name, shape in hyponet.params_shape.items():
            dim = shape[0]
            self.params_init[name] = nn.Parameter(torch.randn(shape[1], dim))
            self.params_transformer[name] = TransformerDecoder(dim=dim, **params_transformer_args)
            self.params_convertfc[name] = nn.Linear(dtoken_dim, dim)

            l = nn.Linear(shape[0] - 1, shape[1])
            weight = l.weight.data.clone()
            bias = l.bias.data.unsqueeze(-1).clone()
            p = nn.Parameter(torch.cat([weight, bias], dim=1).t().detach())
            self.base_params[name] = p
            self.k_base[name] = nn.Parameter(torch.tensor(1.))
            self.k_gen[name] = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        """
            x: (B, L, dtoken_dim)
        """
        B = x.shape[0]

        # Get htokens
        n_htokens = len(self.htokens)
        htokens = self.htokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([x, htokens], dim=1) # (B, n_dtokens + n_htokens, dtoken_dim)
        htokens = self.tokens_transformer(x)[:, -n_htokens:, :] # (B, n_ptokens, dtoken_dim)

        # Apply htokens to params in each layer
        params = dict()
        for name, shape in self.hyponet.params_shape.items():
            init = self.params_init[name].unsqueeze(0).expand(B, -1, -1)
            cond = self.params_convertfc[name](htokens)
            wb = self.params_transformer[name](init, cond) # (B, shape[1], dim)
            wb = F.normalize(init, dim=-1)
            wb = wb.transpose(-1, -2)

            base_param = self.base_params[name].unsqueeze(0).expand(B, -1, -1)
            params[name] = base_param * self.k_base[name] + wb * self.k_gen[name]

        return params
