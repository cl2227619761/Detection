"""
生成数据集
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchvision import transforms
from skimage import transform

from bbox_dataset import BboxDataset
from util import Transform
from lib.config import OPT


class Dataset:
    """训练集dataset"""

    def __init__(self, opt=OPT):
        self.opt = opt
        self.dataset = BboxDataset(
            data_dir= opt.data_dir, split=opt.train_split
        )
        self.transform = Transform(
            min_size=opt.min_size, max_size=opt.max_size
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ori_img, bbox, label, _ = self.dataset[index]
        img, bbox, label, scale = self.transform(
            in_data=(ori_img, bbox, label)
        )
        return img, bbox, label, scale


