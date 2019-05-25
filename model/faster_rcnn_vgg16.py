"""
vgg16作为特征提取网络的faster rcnn
"""
import sys
sys.path.append("../")
import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np

from model.roi_module import RoIPooling2D
from model.faster_rcnn import FasterRCNN

from lib.config import OPT
from model.rpn import RegionProposalNetwork


def decom_vgg16():
    """特征提取网络"""
    model = vgg16(pretrained=True)
    features = list(model.features)[:30]  # vgg16的特征提取部分
    classifier = list(model.classifier)
    del classifier[6]
    # 如果不使用dropout
    if not OPT.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)
    # 冻结前4个conv层
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), classifier


class VGG16RoIHead(nn.Module):
    """最后的定位和分类头
    参数：
        n_class: 包括背景类，总共的类别数
        roi_size: 经过roi pooling后的大小
        spatial_scale: 1./sub_sample
        classifier: 分类层相关的全连接层
    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        # n_class包含了背景在内
        super(VGG16RoIHead, self).__init__()

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(roi_size, roi_size, self.spatial_scale)
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def forward(self, x, rois, roi_indices):
        # 以防出现ndarray
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


class FasterRCNNVGG16(FasterRCNN):
    """继承基类FasterRCNN定制专门的网络"""

    feat_stride = 16

    def __init__(self, n_fg_class=OPT.n_fg_class):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork()
        head = VGG16RoIHead(
            n_class=n_fg_class + 1, roi_size=7,
            spatial_scale=1. / self.feat_stride, classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor=extractor, rpn=rpn, head=head
        )


def normal_init(layer, mean, std):
    """初始化方式"""
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()


def totensor(data, cuda=True):
    """"将nd.array转为tensor"""
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def main():
    """调试用函数"""
    faster_rcnn = FasterRCNNVGG16().cuda()
    img = np.random.randn(3, 800, 800)
    size = (800, 800)
    bboxes, labels, scores = faster_rcnn.predict([img], [size])
    import ipdb; ipdb.set_trace()
    print(faster_rcnn.n_class)


if __name__ == "__main__":
    main()
