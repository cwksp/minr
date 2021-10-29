import numpy as np
import torch
import torch.nn as nn


class HypoPeMlp(nn.Module):

    def __init__(self, depth=5, in_dim=2, pe_dim=128, hidden_dim=256):
        super().__init__()
        self.params_shape = dict()
        self.depth = depth
        self.pe_dim = pe_dim
        last = in_dim * pe_dim
        for i in range(depth):
            cur = hidden_dim if i < depth - 1 else 3
            self.params_shape[f'wb{i}'] = (last + 1, cur) # + 1 for bias
            last = cur
        self.relu = nn.ReLU()

    def convert_posenc(self, x):
        w = 2**torch.linspace(0, 10, self.pe_dim // 2, device=x.device)
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, coords, params):
        B, query_shape = coords.shape[0], coords.shape[1: -1]
        x = coords.view(B, -1, coords.shape[-1])
        x = self.convert_posenc(x)
        for i in range(self.depth):
            one = torch.tensor([1.], dtype=torch.float32, device=x.device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = torch.bmm(torch.cat([x, one], dim=-1), params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + .5
        x = x.view(B, *query_shape, 3)
        return x
