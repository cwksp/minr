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
        s_img = data['support_img']
        params = self.generate_params(s_img)
        q_coord = data['query_coord']

        if mode == 'default':
            pred = self.hyponet(q_coord, params)

        elif mode == 'batched_rendering':
            B = q_coord.shape[0]
            query_shape = q_coord.shape[1: -1]
            q_coord = q_coord.view(B, -1, 2)
            tot_rays = q_coord.shape[1]
            bs_rays = kwargs['batch_size']
            pred = []
            ql = 0
            while ql < tot_rays:
                qr = min(ql + bs_rays, tot_rays)
                coords = q_coord[:, ql: qr, :].contiguous()
                cur = self.hyponet(coords, params)
                pred.append(cur)
                ql = qr
            pred = torch.cat(pred, dim=1).view(B, *query_shape, 3)

        return pred
