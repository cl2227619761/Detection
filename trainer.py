"""
为了方便训练过程，把训练的过程写成了一个类
"""
from collections import namedtuple
import sys
sys.path.append("./model/")
sys.path.append("./")

import torch
import torch.nn as nn
from torchnet.meter import ConfusionMeter, AverageValueMeter
import numpy as np

from utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from utils.array_tool import tonumpy, totensor
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from lib.config import OPT

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

        # 锚点框相对于真实框的真实偏移量和前景背景标签
        self.anchor_target_creator = AnchorTargetCreator()
        # 候选框相对于真实框的真实偏移量和类别标签
        self.proposal_target_creator = ProposalTargetCreator()

        # 位置估计的均值和标准差
        self.loc_normalize_mean = self.faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = self.faster_rcnn.loc_normalize_std

        # 优化器
        self.optimizer = self.faster_rcnn.get_optimizer()

        # 损失计算的超参数
        self.rpn_sigma = OPT.rpn_sigma
        self.roi_sigma = OPT.roi_sigma

        # 训练过程中的一些评估指标
        # rpn过程的评估指标--混淆矩阵
        rpn_cm = ConfusionMeter(2)  # 只有前景和背景两类
        # fast rcnn过程的评估指标--混淆矩阵
        roi_cm = ConfusionMeter(OPT.n_fg_class + 1)  # 前景类别数+背景类
        # 损失函数--average loss
        # 每个损失函数都运用一个averagevaluemeter进行求平均
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

    def forward(self, imgs, bboxes, labels, scale):
        """前向传播过程计算损失
        返回：5个损失"""
        num_batch = bboxes.shape[0]
        if num_batch != 1:
            raise ValueError("仅支持batch_size=1")

        # 得到图片的尺寸H, W
        _, _, H, W = imgs.shape
        img_size = (H, W)
        # 得到特征图
        features = self.faster_rcnn.extractor(imgs)
        # 进入rpn网络, 输出预测的锚点框预测偏移量和得分
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
            features, img_size, scale
        )
        # 由于batch size为1，所以取其中的元素为：
        bbox = bboxes[0]
        label = labels[0]
        rpn_loc = rpn_locs[0]
        rpn_score = rpn_scores[0]
        roi = rois

        # 产生锚点框的真实偏移量和标签
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox=tonumpy(data=bbox), anchor=anchor, img_size=img_size
        )

        # 产生候选框的真实偏移量和标签
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi=roi, bbox=tonumpy(bbox), label=tonumpy(label),
            loc_normalize_mean=self.loc_normalize_mean,
            loc_normalize_std=self.loc_normalize_std
        )
        # 由于batch_size=1，所以sample_roi_indice都为0
        sample_roi_index = torch.zeros(len(sample_roi))
        # 产生由候选框产生的预测框的偏移量和得分
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            x=features, rois=sample_roi, roi_indices=sample_roi_index
        )

        # ------------------------rpn loss----------------------------------#
        gt_rpn_label = totensor(data=gt_rpn_label).long()
        gt_rpn_loc = totensor(data=gt_rpn_loc)
        rpn_loc_loss = _faster_rcnn_loc_loss(
            pred_loc=rpn_loc, gt_loc=gt_rpn_loc,
            gt_label=gt_rpn_label.data, sigma=self.rpn_sigma
        )
        return rpn_loc_loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    """计算smooth_l1损失，这里之所以不用pytorch自带的，是因为加入了一些超参数。引入
    in_weight是因为背景是不参与loc_loss的计算的，所以可以乘以in_weight来控制，若是背景，
    则乘以0；否则乘以1。sigma是用来调节计算公式的
    """
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + \
        (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _faster_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    """把in_weight使用上，进行loc损失的计算"""
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # 把in_weight里面背景类对应的位置都变为0，非背景类的位置变为1
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    # 这样使用in_weight计算的时候背景类就不起作用了
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # 在rpn损失的计算时忽略-1的标签
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


def main():
    """调试用"""
    faster_rcnn = FasterRCNNVGG16().cuda()
    trainer = FasterRCNNTrainer(faster_rcnn)
    img = np.random.randn(3, 800, 800).astype(np.float32)
    img = torch.from_numpy(img[None]).cuda()
    bbox = np.array([[10, 20, 30, 40], [20, 30, 40, 50]]).astype(np.float32)
    bbox = torch.from_numpy(bbox[None]).cuda()
    label = np.array([[1], [2]], dtype=np.int32)
    label = torch.from_numpy(label[None]).cuda()
    scale = 1.
    rpn_loc_loss = trainer(img, bbox, label, scale)
    import ipdb; ipdb.set_trace()
    print("ok")


if __name__ == "__main__":
    main()
