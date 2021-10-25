import torch
import torch.nn as nn

from models.siren import HypoSiren
from models.pe_mlp import HypoPeMlp


class BaseImgrecHypernet(nn.Module):

    def __init__(self, hyponet_name):
        super().__init__()
        if hyponet_name == 'siren':
            self.hyponet = HypoSiren()
        elif hyponet_name == 'pe_mlp':
            self.hyponet = HypoPeMlp()

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
