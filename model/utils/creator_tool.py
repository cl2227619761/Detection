"""
关于候选框的产生
"""
import numpy as np
import cupy as cp

from utils.bbox_tools import loc2bbox
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
