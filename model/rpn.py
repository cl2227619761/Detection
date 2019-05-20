"""
region proposal network
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append("../")

from utils.bbox_tools import generate_anchor_base, enumerate_anchors
from utils.creator_tool import ProposalCreator
from lib.config import OPT


class RegionProposalNetwork(nn.Module):
    """候选框生成网络"""

    def __init__(
            self, feature_channels=512, mid_channels=512,
            ratios=OPT.ratios, anchor_scales=OPT.anchor_scales,
            sub_sample=OPT.sub_sample,
            proposal_creator_params=dict()
    ):
        super(RegionProposalNetwork, self).__init__()
        # 第一个锚点处生成的锚点框
        self.anchor_base = generate_anchor_base(
            ratios=ratios, anchor_scales=anchor_scales
        )
        self.sub_sample = sub_sample  # 下采样的倍数，由特征提取网络决定
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        # 一个锚点处的锚点框的个数
        n_anchor = self.anchor_base.shape[0]
        # rpn网络中的第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels=feature_channels, out_channels=mid_channels,
            kernel_size=3, stride=1, padding=1
        )
        # rpn网络背景和前景的分类层
        self.score = nn.Conv2d(
            in_channels=mid_channels, out_channels=n_anchor * 2,
            kernel_size=1, stride=1, padding=0
        )
        # rpn网络预测的锚点框偏移量
        self.loc = nn.Conv2d(
            in_channels=mid_channels, out_channels=n_anchor * 4,
            kernel_size=1, stride=1, padding=0
        )
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """
        x: 是特征图
        """
        num, _, feature_h, feature_w = x.shape
        anchors = enumerate_anchors(
            anchor_base=self.anchor_base, sub_sample=self.sub_sample,
            height=feature_h, width=feature_w
        )
        n_anchor = anchors.shape[0] // (feature_h * feature_w)
        hidden = F.relu(self.conv1(x))
        rpn_scores = self.score(hidden)
        rpn_locs = self.loc(hidden)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(num, -1, 4)
        rpn_softmax_scores = F.softmax(
            rpn_scores.view(num, feature_h, feature_w, n_anchor, 2),
            dim=4
        )
        rpn_fg_scores = rpn_softmax_scores[..., 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(num, -1)
        rpn_scores = rpn_scores.view(num, -1, 2)
        import ipdb; ipdb.set_trace()

        rois = list()
        roi_indices = list()
        for i in range(num):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchors, img_size, scale
            )
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            roi_indices.append(batch_index)
            rois.append(roi)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchors


def normal_init(layer, mean, stddev):
    """参数的初始化方式"""
    layer.weight.data.normal_(mean, stddev)
    layer.bias.data.zero_()


def main():
    feature = torch.randn(1, 512, 50, 50).cuda()
    rpn = RegionProposalNetwork().cuda()
    rpn_locs, rpn_scores, rois, roi_indices, anchors = rpn(
        feature, (600, 800)
    )
    import ipdb; ipdb.set_trace()
    print("finish")


if __name__ == "__main__":
    main()
