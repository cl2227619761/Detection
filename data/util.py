"""
工具文件
1. 读取图片
"""
import random

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from skimage import transform

from lib.config import OPT


def read_image(path, dtype=np.float32, color=True):
    """读取图片"""
    img_file = Image.open(path)
    try:
        if color:
            img = img_file.convert("RGB")
        else:
            img = img_file.convert("P")
        img = np.array(img, dtype=dtype)
    finally:
        img_file.close()

    if img.ndim == 2:
        return img[np.newaxis]
    else:
        return img.transpose((2, 0, 1))


def img_normalize(img):
    """图像标准化处理"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img = torch.from_numpy(img)
    return normalize(img)


def inverse_normalize(img):
    """标准化处理的反过程"""
    img = img * 0.225 + 0.45
    return img.clip(min=0, max=1) * 255


def preprocess(img, min_size=600, max_size=1000):
    """图片预处理"""
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = transform.resize(
        img, (C, H * scale, W * scale),
        mode="reflect", anti_aliasing=False
    )
    return img_normalize(img)


def resize_bbox(bbox, in_size, out_size):
    """bbox随着图片的预处理过程做相应变化"""
    bbox = bbox.copy()
    y_scale = float(out_size[0] / in_size[0])
    x_scale = float(out_size[1] / in_size[1])
    bbox[:, 0::2] = bbox[:, 0::2] * y_scale
    bbox[:, 1::2] = bbox[:, 1::2] * x_scale
    return bbox


def random_flip(img, y_random=False, x_random=False):
    """对图片进行随机镜像操作"""
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]
    flip_param = {"y_flip": y_flip, "x_flip": x_flip}
    return img, flip_param


def flip_bbox(bbox, img_size, y_flip, x_flip):
    """对bbox做相应的镜像操作"""
    height, width = img_size
    bbox = bbox.copy()
    if y_flip:
        y_max = height - bbox[:, 0]
        y_min = height - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = width - bbox[:, 1]
        x_min = width - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


class Transform:
    """处理加增强操作"""

    def __init__(self, min_size=OPT.min_size, max_size=OPT.max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, height, width = img.shape
        img = preprocess(img, min_size=self.min_size, max_size=self.max_size)
        _, new_height, new_width = img.shape
        scale = new_height / height
        bbox = resize_bbox(
            bbox=bbox, in_size=(height, width),
            out_size=(new_height, new_width)
        )
        img, flip_param = random_flip(img=img, y_random=True, x_random=True)
        bbox = flip_bbox(
            bbox=bbox, img_size=(new_height, new_width),
            y_flip=flip_param["y_flip"], x_flip=flip_param["x_flip"]
        )
        return img, bbox, label, scale

