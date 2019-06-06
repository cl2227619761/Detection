"""
本脚本是基于resnet18为特征提取网络的faster rcnn
"""
import sys
sys.path.append("../")

from torchvision.models import resnet18
import torch
import torch.nn as nn
import numpy as np

from model.roi_module import RoIPooling2D
from model.utils.array_tool import totensor
from model.faster_rcnn import FasterRCNN
from model.rpn import RegionProposalNetwork
from lib.config import OPT


def decom_resnet18():
    """resnet18为特征提取网络"""
    model = resnet18(pretrained=True)
    features = list(model.children())[:-3]
    features = nn.Sequential(*features)
    for param in features.parameters():
        param.requires_grad = False
    classifier = nn.Linear(12544, 4096)
    return features, classifier


class ResNet18RoIHead(nn.Module):
    """最后的定位头和分类头"""

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(ResNet18RoIHead, self).__init__()

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(roi_size, roi_size, spatial_scale)
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        norm_init(self.classifier, 0, 0.01)
        norm_init(self.cls_loc, 0, 0.001)
        norm_init(self.score, 0, 0.01)

    def forward(self, x, rois, roi_indices):
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # 将y1, x1, y2, x2->x1, y1, x2, y2很重要
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc)
        roi_scores = self.score(fc)
        return roi_cls_locs, roi_scores


class FasterRCNNResNet16(FasterRCNN):
    """继承FasterRCNN"""

    feat_stride = 16

    def __init__(self, n_fg_class=OPT.n_fg_class):
        extractor, classifier = decom_resnet18()
        rpn = RegionProposalNetwork()
        head = ResNet18RoIHead(
            n_class=n_fg_class + 1, roi_size=OPT.roi_size,
            spatial_scale=1. / self.feat_stride, classifier=classifier
        )
        super(FasterRCNNResNet16, self).__init__(
            extractor=extractor, rpn=rpn, head=head
        )


def norm_init(layer, mean, std):
    """层初始化"""
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()


def main():
    """调试"""
    # faster_rcnn = FasterRCNNResNet16().cuda()
    # img = np.random.randn(3, 800, 800)
    # size = (800, 800)
    # bboxes, labels, scores = faster_rcnn.predict([img], [size])
    # print(faster_rcnn.n_class)
    extractor, classifier = decom_resnet18()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()