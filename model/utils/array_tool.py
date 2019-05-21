"""
本脚本提供torch张量和数组之间的转换
"""
import numpy as np
import torch


def totensor(data, cuda=True):
    """将数组转变为torch张量"""
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def tonumpy(data):
    """将torch张量转变为数组"""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def scalar(data):
    """取单个值"""
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()
