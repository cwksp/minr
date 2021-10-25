import numpy as np
import torch
import torch.nn as nn


class HypoPeMlp(nn.Module):

    def __init__(self, depth=5, in_dim=2, pe_dim=256, hidden_dim=256):
        super().__init__()
        self.params_shape = dict()
        self.depth = depth
        self.in_dim = in_dim
        self.pe_dim = pe_dim
        last = in_dim
        assert pe_dim % in_dim == 0
        last = pe_dim # posenc
        for i in range(depth):
            cur = hidden_dim if i < depth - 1 else 3
            self.params_shape[f'wb{i}'] = (last + 1, cur) # + 1 for bias
            last = cur
        self.relu = nn.ReLU()

    def convert_posenc(self, x):
        x = torch.cat([
            torch.cat([torch.cos(np.pi * x * 2**i), torch.sin(np.pi * x * 2**i)], dim=-1)
            for i in np.linspace(0, 10, self.pe_dim // self.in_dim // 2)
        ], dim=-1)
        return x

    def forward(self, coords, params):
        device = coords.device
        B, query_shape = coords.shape[0], coords.shape[1: -1]

        x = coords.view(B, -1, 2)
        x = self.convert_posenc(x)
        for i in range(self.depth):
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = torch.bmm(torch.cat([x, one], dim=-1), params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + .5

        x = x.view(B, *query_shape, 3)
        return x
