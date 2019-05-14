"""
工具文件
1. 读取图片
"""
import numpy as np
from PIL import Image


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
