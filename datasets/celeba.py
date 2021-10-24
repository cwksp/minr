import os.path as osp
from PIL import Image

import torch
import pandas
from torchvision import transforms

from datasets import register


@register('celeba')
class CelebA(torch.utils.data.Dataset):

    def __init__(self, root_path, split):
        partition_csv = pandas.read_csv(osp.join(root_path, 'list_eval_partition.csv'))
        split_id = {'train': 0, 'val': 1, 'test': 2}[split]
        img_names = partition_csv[partition_csv['partition'] == split_id]['image_id']
        self.paths = [osp.join(root_path, 'img_align_celeba', 'img_align_celeba', _) for _ in img_names]
        self.center_size = 178

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = transforms.ToTensor()(Image.open(self.paths[idx]))
        R = self.center_size // 2
        return x[:, x.shape[1] // 2 - R: x.shape[1] // 2 + R, x.shape[2] // 2 - R: x.shape[2] // 2 + R]
