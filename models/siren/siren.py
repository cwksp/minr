import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SirenLayer(nn.Module):

    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last

        b = 1 / in_f if self.is_first else np.sqrt(6 / in_f) / w0
        weight = torch.empty(out_f, in_f)
        weight.uniform_(-b, b)
        bias = torch.zeros(out_f)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        return x + .5 if self.is_last else self.w0 * x


class SirenModel(nn.Module):

    def __init__(self, w0=200, width=256, depth=5):
        super().__init__()
        self.w0 = w0
        self.width = width
        self.depth = depth
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                l = SirenLayer(2, self.width, w0=self.w0, is_first=True)
            elif i < self.depth - 1:
                l = SirenLayer(self.width, self.width, w0=self.w0)
            else:
                l = SirenLayer(self.width, 3, w0=self.w0, is_last=True)
            self.layers.append(l)

    def forward(self, coords):
        in_shp = coords.shape[:-1]
        x = coords.view(-1, 2)
        for i in range(self.depth - 1):
            x = self.layers[i](x)
            x = torch.sin(x)
        x = self.layers[-1](x)
        out = x.view(*in_shp, 3)
        return out
