"""
工具文件
1. 读取图片
"""
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage import transform


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



