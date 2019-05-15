"""
生成数据集
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchvision import transforms
from skimage import transform


def img_normalize(img):
    """对图片进行标准化处理"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img = torch.from_numpy(img)
    return normalize(img)


def inverse_normalize(img):
    """将标准化后的图片转回来"""
    img = img * 0.225 + 0.45
    return img.clip(min=0, max=1) * 225


def preprocess(img, min_size=600, max_size=1000):
    """对图像进行预处理"""
    channel, height, width = img.shape
    scale1 = min_size / min(height, width)
    scale2 = max_size / max(height, width)
    scale = min(scale1, scale2)
    img = img / 255.
    img = transform.resize(
        img, (channel, height * scale, width * scale), mode="reflect",
        anti_aliasing=False
    )
    return img_normalize(img)


def resize_bbox(bbox, in_size, out_size):
    """根据图像的变化对真实框做相应的变化"""

