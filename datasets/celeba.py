import os.path as osp
from PIL import Image

import torch
import pandas
from torchvision import transforms

from datasets import register
from utils import make_coord


@register('celeba')
class Celeba(torch.utils.data.Dataset):
    """
        Given a support image, query INR-image.

        Item:
            support_img: (3, H, W)
            query_coord: (gt_size, gt_size, 2)
            gt: (gt_size, gt_size, 3)
    """

    def __init__(self, root_path, split, input_crop='none', truncate=None, repeat=1):
        partition_csv = pandas.read_csv(osp.join(root_path, 'list_eval_partition.csv'))
        split_id = {'train': 0, 'val': 1, 'test': 2}[split]
        img_names = partition_csv[partition_csv['partition'] == split_id]['image_id']
        self.paths = [osp.join(root_path, 'img_align_celeba', 'img_align_celeba', _) for _ in img_names]
        self.gt_size = 178

        # self.input_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(input_size, transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        # ])
        self.input_crop = input_crop

        if truncate is not None:
            self.paths = self.paths[:truncate]
        self.repeat = repeat

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        idx %= len(self.paths)
        gt = transforms.ToTensor()(Image.open(self.paths[idx]))
        R = self.gt_size // 2
        gt = gt[:, gt.shape[1] // 2 - R: gt.shape[1] // 2 + R, gt.shape[2] // 2 - R: gt.shape[2] // 2 + R]

        # x = self.input_transform(gt)
        x = gt.clone()
        if self.input_crop == 'center':
            R = x.shape[1] // 4
            x = x[:, x.shape[1] // 2 - R: x.shape[1] // 2 + R, x.shape[2] // 2 - R: x.shape[2] // 2 + R]
        elif self.input_crop == 'top':
            x = x[:, :x.shape[1] // 2, :]
        elif self.input_crop == 'left':
            x = x[:, :, :x.shape[2] // 2]

        return {
            'support_img': x,
            'query_coord': make_coord((gt.shape[1], gt.shape[2]), flatten=False),
            'gt': gt.permute(1, 2, 0),
        }
