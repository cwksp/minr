import math

import torch
import torch.nn as nn

from models import register
from .base_nvs_hypernet import BaseNvsHypernet


@register('nvsh-base_params')
class NvshBaseParams(BaseNvsHypernet):

    def __init__(self, hyponet_spec):
        super().__init__(hyponet_spec)
        self.base_params = dict()
        for name, shape in self.hyponet.params_shape.items():
            l = nn.Linear(shape[0] - 1, shape[1])
            weight = l.weight.data.clone()
            bias = l.bias.data.unsqueeze(-1).clone()

            p = nn.Parameter(torch.cat([weight, bias], dim=1).t().detach())
            self.base_params[name] = p
            self.register_parameter(f'base_params_{name}', p)

    def generate_params(self, rays_o, rays_d, imgs):
        B = rays_o.shape[0]
        ret = dict()
        for k, v in self.base_params.items():
            ret[k] = v.unsqueeze(0).expand(B, -1, -1)
        return ret
