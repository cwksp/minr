import numpy as np
import torch
import torch.nn as nn

from models import register
from .base_siren_hypernet import BaseSirenHypernet


@register('sh-base_params')
class SHBaseParams(BaseSirenHypernet):

    def __init__(self):
        super().__init__()
        self.base_params = dict()
        for name, shape in self.hyponet.params_shape.items():
            in_f, out_f = shape[0] - 1, shape[1]
            weight = torch.empty(out_f, in_f)
            b = 1 / in_f if name[-1] == '0' else np.sqrt(6 / in_f) / self.hyponet.w0
            weight.uniform_(-b, b)
            bias = torch.zeros(shape[1], 1)

            p = nn.Parameter(torch.cat([weight, bias], dim=1).t().detach())
            self.base_params[name] = p
            self.register_parameter(f'base_params_{name}', p)

    def generate_params(self, imgs):
        B = imgs.shape[0]
        ret = dict()
        for k, v in self.base_params.items():
            ret[k] = v.unsqueeze(0).expand(B, -1, -1)
        return ret
