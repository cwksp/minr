import numpy as np
import torch
import torch.nn as nn

from models import register
from .base_imgrec_hypernet import BaseImgrecHypernet


@register('imgrech-base_params')
class ImgrechBaseParams(BaseImgrecHypernet):

    def __init__(self, hyponet_name):
        super().__init__(hyponet_name)
        self.base_params = dict()
        for name, shape in self.hyponet.params_shape.items():
            if hyponet_name == 'siren':
                in_f, out_f = shape[0] - 1, shape[1]
                weight = torch.empty(out_f, in_f)
                b = 1 / in_f if name[-1] == '0' else np.sqrt(6 / in_f) / self.hyponet.w0
                weight.uniform_(-b, b)
                bias = torch.zeros(shape[1], 1)

            elif hyponet_name == 'pe_mlp':
                l = nn.Linear(shape[0] - 1, shape[1])
                weight = l.weight.data.clone()
                bias = l.bias.data.unsqueeze(-1).clone()

            p = nn.Parameter(torch.cat([weight, bias], dim=1).t().detach())
            self.base_params[name] = p
            self.register_parameter(f'base_params_{name}', p)

    def generate_params(self, imgs):
        B = imgs.shape[0]
        ret = dict()
        for k, v in self.base_params.items():
            ret[k] = v.unsqueeze(0).expand(B, -1, -1)
        return ret
