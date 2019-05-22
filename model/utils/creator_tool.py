"""
关于候选框的产生
"""
import numpy as np
import cupy as cp

from utils.bbox_tools import loc2bbox, bbox_iou
from utils.nms import non_maximum_suppression


class ProposalCreator:
    """
    候选框的产生
    这是由rpn预测出来的偏移量先换算为坐标框；
    然后利用非极大值抑制从中筛选出一些坐标框作为候选框，
    用于训练fast rcnn网络
    """

    def __init__(
            self,
            parent_model, nms_thresh=0.7,
            n_train_pre_nms=12000,
            n_train_post_nms=2000,
            n_test_pre_nms=6000,
            n_test_post_nms=300,
            min_size=16
    ):
        """
        参数：
            parent_model: 模型
            nms_thresh: nms的阈值
            min_size: 注意这是在原图上的大小，我们需要乘以scale
        """
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], a_min=0, a_max=img_size[0]
        )
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], a_min=0, a_max=img_size[1]
        )
        min_size = self.min_size * scale  # 这里转换为了处理后的图片的上面
        roi_h = roi[:, 2] - roi[:, 0]
        roi_w = roi[:, 3] - roi[:, 1]
        keep = np.where((roi_h >= min_size) & (roi_w >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        # 这是nms之前的选取，按照得分的高低进行排序，是前景的得分
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh
        )
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        for i in range(keep.size):
            keep[i] = keep[i].tolist()
        keep = np.int32(keep)
        roi = roi[keep]
        return roi


class AnchorTargetCreator:
    """为锚点框赋予真实的位置和标签信息，用于训练RPN网络"""

    def __init__(
            self, n_sample=256,
            pos_iou_thresh=0.7, neg_iou_thresh=0.3,
            pos_ratio=0.5
    ):
        """
        参数：
            n_sample: 使用的锚点框的数量
            pos_iou_thresh: 赋予标签时iou的上限
            neg_iou_thresh: 赋予标签时iou的下限
            pos_ratio: 其中正样本所占的比例
        """
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """为抽到的n_sample个框赋予偏移量和标签
        返回：赋予的偏移量和标签
        """
        img_h, img_w = img_size
        # 得到不超出图像边界的锚点框的索引
        inside_index = _get_inside_index(anchor, img_h, img_w)
        # 得到不超出图像边界的锚点框
        anchor = anchor[inside_index]
        # 然后就可以创建标签了
        argmax_iou, label = self._create_label(inside_index, anchor, bbox)


    def _create_label(self, inside_index, anchor, bbox):
        """创建标签的函数"""
        # 初始化为-1
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_iou, max_ious, gt_argmax_iou = self._calc_ious(
            anchor=anchor, bbox=bbox, inside_index=inside_index
        )
        # 先赋予负样本
        label[max_ious < self.neg_iou_thresh] = 0
        # 正样本
        label[gt_argmax_iou] = 1
        label[max_ious >= self.pos_iou_thresh] = 1
        # 如果正样本的数量多于n_sample * pos_ratio，则需要抽样。抽样过程就是把一部分
        # 正样本的标签改为-1
        n_pos = int(self.n_sample * self.pos_ratio)
        pos_index = np.where(label == 1)[0]
        # 如果正样本的数量大于n_pos，则：
        if len(pos_index) > n_pos:
            disabled_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False
            )
            label[disabled_index] = -1

        # 对于负样本进行同样的抽样操作
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disabled_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False
            )
            label[disabled_index] = -1
        return argmax_iou, label

    def _calc_ious(self, anchor, bbox, inside_index):
        """根据iou筛选锚点框索引
        返回的是：
            1. 和锚点框具有较大iou的真实框索引
            2. 这些iou是多少
            3. 和真实框具有最大iou的锚点框索引"""
        iou = bbox_iou(anchor, bbox)
        # 和锚点框具有较大iou的真实框的索引
        argmax_iou = iou.argmax(axis=1)
        # 得到inside_index个这些较大的iou
        max_ious = iou[np.arange(len(inside_index)), argmax_iou]
        # 和真实框具有较大iou的锚点框索引
        gt_argmax_iou = iou.argmax(axis=0)
        # 这些iou是
        gt_max_ious = iou[gt_argmax_iou, np.arange(iou.shape[1])]
        gt_argmax_iou = np.where(iou == gt_max_ious)[0]
        return argmax_iou, max_ious, gt_argmax_iou




def _get_inside_index(anchor, img_h, img_w):
    """得到不超出边界的锚点框的索引"""
    inside_index = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= img_h) &
        (anchor[:, 3] <= img_w)
    )[0]
    return inside_index
