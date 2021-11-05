import os
import os.path as osp
from PIL import Image

import torch
from torchvision import transforms

from datasets import register
from utils import make_coord


@register('imagenette')
class Imagenette(torch.utils.data.Dataset):

    def __init__(self, root_path, split, augment='none'):
        root_path = osp.join(root_path, split)
        wnids = os.listdir(root_path)
        self.paths = []
        for wnid in wnids:
            self.paths.extend([
                osp.join(root_path, wnid, _)
                for _ in os.listdir(osp.join(root_path, wnid))])
        if split == 'val':
            assert augment == 'none'
        if augment == 'flip':
            self.transform = transforms.Compose([
                transforms.Resize(178),
                transforms.CenterCrop(178),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(178),
                transforms.RandomCrop(178),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif augment == 'rawcrop':
            self.transform = transforms.Compose([
                transforms.RandomCrop(178),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif augment == 'resizedcrop':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(178),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(178),
                transforms.CenterCrop(178),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        x = self.transform(img)
        return {
            'support_img': x,
            'query_coord': make_coord((x.shape[1], x.shape[2]), flatten=False),
            'gt': x.permute(1, 2, 0),
        }
