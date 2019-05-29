"""
本脚本是基于resnet18为特征提取网络的faster rcnn
"""
from torchvision.models import resnet18
import torch
import torch.nn as nn


def decom_resnet18():
    """resnet18为特征提取网络"""
    model = resnet18(pretrained=True)
    features = list(model.children())[:-3]
    return nn.Sequential(*features)
