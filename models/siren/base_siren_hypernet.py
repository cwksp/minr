import torch
import torch.nn as nn
import torch.nn.functional as F


class HypoSiren(nn.Module):

    def __init__(self, depth=5, hidden_dim=256, w0=200):
        super().__init__()
        self.params_shape = dict()
        self.depth = depth
        self.w0 = w0
        last = 2
        for i in range(depth):
            cur = hidden_dim if i < depth - 1 else 3
            self.params_shape[f'wb{i}'] = (last + 1, cur) # + 1 for bias
            last = cur

    def forward(self, coords, params):
        device = coords.device
        B, query_shape = coords.shape[0], coords.shape[1: -1]

        x = coords.view(B, -1, 2)
        for i in range(self.depth):
            one = torch.tensor([1.], dtype=torch.float32, device=device).view(1, 1, 1).expand(B, x.shape[1], 1)
            x = torch.bmm(torch.cat([x, one], dim=-1), params[f'wb{i}'])
            if i < self.depth - 1:
                x = torch.sin(self.w0 * x)
            else:
                x = x + .5

        x = x.view(B, *query_shape, 3)
        return x


class BaseSirenHypernet(nn.Module):

    def __init__(self):
        super().__init__()
        self.hyponet = HypoSiren()

    def generate_params(self, imgs):
        raise NotImplementedError

    def forward(self, data, mode='default', **kwargs):
        s_imgs = data['support_imgs']
        params = self.generate_params(s_imgs)
        q_coords = data['query_coords']

        if mode == 'default':
            pred = self.hyponet(q_coords, params)

        elif mode == 'batched_rendering':
            B = q_coords.shape[0]
            query_shape = q_coords.shape[1: -1]
            q_coords = q_coords.view(B, -1, 2)
            tot_rays = q_coords.shape[1]
            bs_rays = kwargs['batch_size']
            pred = []
            ql = 0
            while ql < tot_rays:
                qr = min(ql + bs_rays, tot_rays)
                coords = q_coords[:, ql: qr, :].contiguous()
                cur = self.hyponet(coords, params)
                pred.append(cur)
                ql = qr
            pred = torch.cat(pred, dim=1).view(B, *query_shape, 3)

        return pred
