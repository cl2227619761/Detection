"""
为了方便训练过程，把训练的过程写成了一个类
"""
from collections import namedtuple

import torch.nn as nn

# 总过有5个损失，用一个namedtuple来存放
LossTuple = namedtuple(
    "LossTuple",
    [
        "rpn_loc_loss", "rpn_cls_loss", "roi_loc_loss", "roi_cls_loss",
        "total_loss"
    ]
)


class FasterRCNNTrainer(nn.Module):
    """把训练过程写入类里面，方便训练"""

    def __init__(self, faster_rcnn):
        """faster_rcnn是继承了faster rcnn基类的子网络"""
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn

        # 位置估计的均值和标准差
        self.loc_normalize_mean = self.faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = self.faster_rcnn.loc_normalize_std

        # 优化器
        self.optimizer = self.faster_rcnn.get_optimizer()
