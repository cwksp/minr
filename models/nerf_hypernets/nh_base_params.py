import math

import torch
import torch.nn as nn

from models import register
from .base_nerf_hypernet import BaseNerfHypernet


@register('nh-base_params')
class NHBaseParams(BaseNerfHypernet):

    def __init__(self, use_viewdirs):
        super().__init__(use_viewdirs=use_viewdirs)
        self.base_params = dict()
        for name, shape in self.hypo_nerf.params_shape.items():
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

            bias = torch.empty(shape[1], 1)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

            p = nn.Parameter(torch.cat([weight, bias], dim=1).t().detach())
            self.base_params[name] = p
            self.register_parameter(f'base_params_{name}', p)

    def generate_params(self, rays_o, rays_d, imgs):
        B = rays_o.shape[0]
        ret = dict()
        for k, v in self.base_params.items():
            ret[k] = v.unsqueeze(0).expand(B, -1, -1)
        return ret
