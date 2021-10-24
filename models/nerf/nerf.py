import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


@register('nerf')
class NeRF(nn.Module):

    def __init__(self, depth=6, hidden_dim=256, use_viewdir=False):
        super().__init__()
        layers = []
        last = 3
        last *= 40 # posenc
        for i in range(depth - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            last = hidden_dim
        self.layers = nn.Sequential(*layers)

        self.use_viewdir = use_viewdir
        if use_viewdir:
            self.viewdir_prefc = nn.Linear(3, hidden_dim // 2)
            self.sigma_head = nn.Linear(last, 1)
            self.rgb_head = nn.Sequential(
                nn.Linear(last + hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
        else:
            self.head = nn.Linear(last, 4)

    def convert_posenc(self, x):
        x = torch.cat([
            torch.cat([torch.sin(x * 2**i), torch.cos(x * 2**i)], dim=-1)
            for i in np.linspace(0, 8, 20)
        ], dim=-1)
        return x

    def forward(self, coords, viewdirs=None):
        """
        Args:
            coords, viewdirs: shape (..., 3).
        Returns:
            shape (..., 3 + 1): rgb (3) + sigma (1)
        """
        query_shape = coords.shape[:-1]
        coords = coords.view(-1, 3)
        coords = self.convert_posenc(coords)
        x = self.layers(coords)

        if self.use_viewdir:
            sigma = self.sigma_head(x)
            viewdirs = viewdirs.view(-1, 3)
            viewdirs = F.normalize(viewdirs, dim=-1)
            viewdirs = F.relu(self.viewdir_prefc(viewdirs))
            rgb = self.rgb_head(torch.cat([x, viewdirs], dim=-1))
            x = torch.cat([rgb, sigma], dim=-1)
        else:
            x = self.head(x)

        x = torch.cat([torch.sigmoid(x[:, :3]), F.relu(x[:, 3:])], dim=-1)
        x = x.view(*query_shape, -1)
        return x
