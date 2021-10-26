import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from .base_imgrec_hypernet import BaseImgrecHypernet


@register('imgrech-ttp')
class ImgrechNetTtp(BaseImgrecHypernet):

    def __init__(self, input_size, patch_size, dtoken_dim, hyponet_name, ttp_net_spec):
        super().__init__(hyponet_name)
        self.patch_size = patch_size
        self.prefc = nn.Linear(patch_size**2 * 3, dtoken_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, (input_size // patch_size)**2, dtoken_dim))
        self.ttp_net = models.make(ttp_net_spec, args={'dtoken_dim': dtoken_dim, 'hyponet': self.hyponet})
        # self.adjust_k = nn.ParameterDict()
        # for name, shape in self.hyponet.params_shape.items():
        #     self.adjust_k[name] = nn.Parameter(torch.tensor(1e-2))

    def generate_params(self, imgs):
        P = self.patch_size
        x = F.unfold(imgs, P, stride=P).permute(0, 2, 1).contiguous() # (B, (H // P) * (W // P), P * P * 3)
        x = self.prefc(x) + self.pos_emb
        params = self.ttp_net(x)
        # for name, p in params.items():
        #     params[name] = p * self.adjust_k[name]
        return params
