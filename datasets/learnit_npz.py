import numpy as np
import torch

from datasets import register
from utils import make_coord


@register('learnit_npz')
class LearnitNpz(torch.utils.data.Dataset):

    def __init__(self, root_path, split):
        with np.load(root_path) as npzfile:
            self.data = npzfile[split + '_data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).permute(2, 0, 1).float() / 255.
        return {
            'support_img': x,
            'query_coord': make_coord((x.shape[1], x.shape[2]), flatten=False),
            'gt': x.permute(1, 2, 0),
        }
