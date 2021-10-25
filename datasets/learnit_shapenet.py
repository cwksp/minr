"""
    https://github.com/tancik/learnit
"""
import json
import os
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from datasets import register
from utils import get_rays_batch


@register('learnit_shapenet')
class LearnitShapenet(Dataset):

    def __init__(self, root_path, split_file, split, n_support_views, n_query_views,
                 truncate=None, repeat=1, views_rng=None):
        obj_ids = json.load(open(split_file, 'r'))[split]
        self.obj_paths = [os.path.join(root_path, _) for _ in obj_ids]
        self.n_support_views = n_support_views
        self.n_query_views = n_query_views

        if truncate is not None:
            self.obj_paths = self.obj_paths[:truncate]
        self.obj_paths = self.obj_paths * repeat
        self.views_rng = views_rng

    def __getitem__(self, idx):
        """
        Returns:
            support | query: [imgs, rays_o, rays_d].
                imgs: shape (N, 3, H, W)
                rays_o, rays_d: shape (N, H, W, 3)
        """
        obj_path = self.obj_paths[idx]
        meta = json.load(open(os.path.join(obj_path, 'transforms.json'), 'r'))

        frames = meta['frames']
        if self.views_rng is not None:
            ql, qr = self.views_rng
            frames = frames[ql: qr]
        frames = np.random.choice(frames, self.n_support_views + self.n_query_views, replace=False)

        imgs = [os.path.join(obj_path, _['file_path'].split('/')[-1] + '.png') for _ in frames]
        imgs = torch.stack([transforms.ToTensor()(Image.open(_)) for _ in imgs])
        assert imgs.shape[1] == 4
        imgs = imgs[:, :-1, ...] * imgs[:, -1:, ...] + (1 - imgs[:, -1:, ...])

        poses = [_['transform_matrix'] for _ in frames]
        poses = torch.tensor(poses, dtype=torch.float)

        camera_angle_x = float(meta['camera_angle_x'])
        h, w = imgs.shape[-2:]
        f = (w / 2) / np.tan(camera_angle_x / 2)
        rays_o, rays_d = get_rays_batch(h, w, f, poses[:, :3, :])

        ret = {
            'support_imgs': imgs[:self.n_support_views],
            'support_rays_o': rays_o[:self.n_support_views],
            'support_rays_d': rays_d[:self.n_support_views],
            'query_imgs': imgs[self.n_support_views:],
            'query_rays_o': rays_o[self.n_support_views:],
            'query_rays_d': rays_d[self.n_support_views:],
            'near': torch.tensor(2., dtype=torch.float32),
            'far': torch.tensor(6., dtype=torch.float32),
        }
        return ret

    def __len__(self):
        return len(self.obj_paths)
