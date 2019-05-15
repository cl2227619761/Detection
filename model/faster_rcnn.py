"""
faster rcnn模型基类
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def nograd(func):
    """修饰器，非训练阶段使用"""
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


class FasterRCNN(nn.Module):
    """Faster RCNN基类
    参数：
        extractor: 特征提取网络
        rpn: 候选框生成网络
        head: 分类和回归头
        loc_normalize_mean: 位置估计的均值
        loc_normalize_std: 位置估计的标准差
    返回：
        forward过程返回预测框的偏移量，得分，坐标，预测框的索引
    """

    def __init__(
            self, extractor, rpn, head,
            loc_normalize_mean, loc_normalize_std
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def forward(self, x):
        features = self.extractor(x)
