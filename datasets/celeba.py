import os.path as osp
from PIL import Image

import torch
import pandas
from torchvision import transforms

from datasets import register
from utils import make_coord


@register('celeba')
class Celeba(torch.utils.data.Dataset):

    def __init__(self, root_path, split, input_size, truncate=None, repeat=1):
        partition_csv = pandas.read_csv(osp.join(root_path, 'list_eval_partition.csv'))
        split_id = {'train': 0, 'val': 1, 'test': 2}[split]
        img_names = partition_csv[partition_csv['partition'] == split_id]['image_id']
        self.paths = [osp.join(root_path, 'img_align_celeba', 'img_align_celeba', _) for _ in img_names]
        self.center_size = 178
        self.input_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        if truncate is not None:
            self.paths = self.paths[:truncate]
        self.repeat = repeat

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        idx %= len(self.paths)
        x = transforms.ToTensor()(Image.open(self.paths[idx]))
        R = self.center_size // 2
        x = x[:, x.shape[1] // 2 - R: x.shape[1] // 2 + R, x.shape[2] // 2 - R: x.shape[2] // 2 + R]
        return {
            'support_imgs': self.input_transform(x),
            'query_coords': make_coord((x.shape[1], x.shape[2]), flatten=False),
            'gt': x.permute(1, 2, 0),
        }
