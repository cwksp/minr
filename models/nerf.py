import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class HypoNeRF(nn.Module):

    def __init__(self, depth=6, hidden_dim=256, use_viewdirs=False):
        super().__init__()
        self.params_shape = dict()
        self.depth = depth
        self.use_viewdirs = use_viewdirs

        last = 3
        last *= 40 # posenc
        for i in range(depth - 1):
            self.params_shape[f'wb{i}'] = (last + 1, hidden_dim) # + 1 for bias
            last = hidden_dim

        if use_viewdirs:
            self.params_shape['wb_sigma'] = (last + 1, 1)

            self.params_shape['wb_mapview'] = (3 + 1, hidden_dim // 2)
            self.params_shape['wb_catview'] = (last + hidden_dim // 2 + 1, hidden_dim)
            self.params_shape['wb_rgb'] = (hidden_dim + 1, 3)
        else:
            self.params_shape['wb_rgbsigma'] = (last + 1, 4) # 4: rgb + sigma

    def convert_posenc(self, x):
        x = torch.cat([
            torch.cat([torch.sin(x * 2**i), torch.cos(x * 2**i)], dim=-1)
            for i in np.linspace(0, 8, 20)
        ], dim=-1)
        return x

    def forward(self, coords, params, viewdirs=None):
        """
        Args:
            coords: shape (B, ..., 3).
            params: dict, value shape (B, NROWS_i, NCOLS_i).
            viewdirs: optional.
        Returns:
            shape (B, ..., 3 + 1): rgb (3) + sigma (1).
        """
        device = coords.device
        B, query_shape = coords.shape[0], coords.shape[1: -1]

        coords = coords.view(B, -1, 3)
        coords = self.convert_posenc(coords)
        if viewdirs is not None:
            viewdirs = viewdirs.view(B, -1, 3)

        x = coords
        for i in range(self.depth - 1):
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = F.relu(torch.bmm(torch.cat([x, one], dim=-1), params[f'wb{i}']))

        if self.use_viewdirs:
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            sigma = F.relu(torch.bmm(torch.cat([x, one], dim=-1), params['wb_sigma']))

            v = F.normalize(viewdirs, dim=-1)
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, v.shape[1], 1)
            v = F.relu(torch.bmm(torch.cat([v, one], dim=-1), params['wb_mapview']))
            x = torch.cat([x, v], dim=-1)
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = F.relu(torch.bmm(torch.cat([x, one], dim=-1), params['wb_catview']))

            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            rgb = torch.sigmoid(torch.bmm(torch.cat([x, one], dim=-1), params['wb_rgb']))
        else:
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = torch.bmm(torch.cat([x, one], dim=-1), params['wb_rgbsigma'])
            rgb, sigma = torch.sigmoid(x[..., :3]), F.relu(x[..., 3:])

        x = torch.cat([rgb, sigma], dim=-1)
        x = x.view(B, *query_shape, -1)
        return x
