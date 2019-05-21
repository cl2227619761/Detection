"""
faster rcnn模型基类
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../")

import torch
import torch.nn as nn

from data.util import preprocess
from utils.array_tool import tonumpy, totensor


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
            loc_normalize_mean=[0., 0., 0., 0.],
            loc_normalize_std=[0.1, 0.1, 0.2, 0.2]
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor  # 特征提取器
        self.rpn = rpn  # region proposal network
        self.head = head  # 最后的分类头和定位头

        # 对偏移量和缩放量进行矫正的均值和标准差
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    @property
    def n_class(self):
        """包括背景在内的类别总数"""
        return self.head.n_class

    def forward(self, x, scale=1.):
        """
        前向传播需要的参数并不能太多，主要就是输入图像，否则会在预测的时候产生问题，因为
        预测的时候，我们所拥有的信息也就只有输入图像
        参数：
            x: 输入的图像
        """
        img_size = x.shape[2:]  # 输入图像的H, W。x是torch张量[N, C, H, W]

        features = self.extractor(x)  # 特征提取器提取到的特征
        # rpn网络产生的预测偏移量，前景vs背景得分，候选框，候选框索引和锚点框
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(
            features, img_size, scale
        )
        # 最后的分类头和定位头得到的预测偏移量缩放量，预测得分;rpn和head是联动的
        roi_cls_locs, roi_scores = self.head(features, rois, roi_indices)
        # 前向传播返回的有最终的预测偏移量缩放量，预测框的得分，候选框，候选框索引
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        """作用是在不同的过程中改变nms的阈值以及score的阈值，舍弃掉低分的框"""
        if preset == "visualize":
            self.nms_thresh = 0.3  # 进行nms的时候的iou阈值
            self.score_thresh = 0.7  # 进行最后筛选的时候的得分阈值
        elif preset == "evaluate":
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError("preset必须取visualize或者evaluate")

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        """预测过程
            参数：输入的是图像
            返回：返回的是框的坐标，框的预测类别，框的预测类别得分
        """
        self.eval()  # 调整预测时网络为eval模式
        if visualize:
            # 设置visualize时的nms_thresh和score_thresh
            self.use_preset("visualize")
            prepared_imgs = list()  # 放置要输入的图片
            sizes = list()  # 放置要输入图片的尺寸H, W
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(img=tonumpy(img))  # 对图片的H, W做缩放操作
                prepared_imgs.append(img)  # 得到处理后的图片
                sizes.append(size)  # 注意这是未经处理的原始图片的尺寸
        else:
            prepared_imgs = imgs  # 如果不做处理就使用原始图片

        bboxes = list()  # 用于放置预测框的坐标
        labels = list()  # 用于放置预测框类别标签
        scores = list()  # 用于放置预测框类别得分
        for img, size in zip(prepared_imgs, sizes):
            # 将图片数组转变为[N, C, H, W]的张量，并为float类型
            import ipdb; ipdb.set_trace()
            img = totensor(img[None]).float()
            scale = img.shape[3] / size[1]  # 处理后的图片的W除以原始图片的W
            # 将图像和缩放比例代入前向传播过程得到预测的偏移量，预测得分，候选框
            roi_cls_locs, roi_scores, rois, _ = self(x=img, scale=scale)
            # 假设batch size大小为1，则有：
            roi_score = roi_scores.data  # 一张图片的预测框得分
            roi_cls_loc = roi_cls_locs.data  # 一张图片的预测框偏移量
            # 上面得到的rois是针对处理过后的图像的，除以scale就得到了在原始图像的rois
            roi = totensor(rois) / scale

            # 将偏移量缩放量转变为坐标，注意需要用到mean和std进行调整？？？
            mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(
                self.n_class
            )[None]
            std = torch.Tensor(self.loc_normalize_std).cuda().repeat(
                self.n_class
            )[None]
            return roi_cls_loc, roi_score


