"""
关于候选框的产生
"""
import numpy as np
import cupy as cp

from utils.bbox_tools import loc2bbox, bbox2loc, bbox_iou
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
        n_anchor = len(anchor)
        # 得到不超出图像边界的锚点框的索引
        inside_index = _get_inside_index(anchor, img_h, img_w)
        # 得到不超出图像边界的锚点框
        anchor = anchor[inside_index]
        # 然后就可以创建标签了
        argmax_iou, label = self._create_label(inside_index, anchor, bbox)

        # 计算锚点框相对于其对应的最大iou的真实框的偏移量作为真实偏移量
        loc = bbox2loc(src_bbox=anchor, dst_bbox=bbox[argmax_iou])
        # 将偏移量和标签整理成(count, 4)和(count,)的形式
        label = _umap(data=label, count=n_anchor, index=inside_index, fill=-1)
        loc = _umap(data=loc, count=n_anchor, index=inside_index, fill=0)
        return loc, label

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


def _umap(data, count, index, fill=0):
    """将标签整理成(count,)的形状，将偏移量整理成(count, 4)的形状"""
    # 对于标签会有：
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    # 对于偏移量会有：
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class ProposalTargetCreator:
    """为候选框产生真实的偏移量和标签，用于训练fast rcnn部分"""

    def __init__(
            self, n_sample=128,
            pos_ratio=0.25, pos_iou_thresh=0.5,
            neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
    ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(
            self, roi, bbox, label,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
    ):
        roi = np.concatenate((roi, bbox), axis=0)
        # 图片上要抽取的阳性框个数
        pos_roi_per_img = np.round(self.n_sample * self.pos_ratio)
        # 计算和真实框的iou
        iou = bbox_iou(roi, bbox)
        # 和roi的iou较大的那个真实框的索引
        gt_assignment = iou.argmax(axis=1)
        # 那个较大的iou是多少
        max_iou = iou.max(axis=1)
        # 去除背景类?
        gt_roi_label = label[gt_assignment] + 1

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # 从非背景类中选取一定数量的roi
        pos_roi_per_this_img = int(min(pos_roi_per_img, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_img, replace=False
            )
        # 从背景类中选取一定数量的背景
        neg_index = np.where(
            (max_iou >= self.neg_iou_thresh_lo) &
            (max_iou < self.neg_iou_thresh_hi)
        )[0]
        neg_roi_per_img = self.n_sample - pos_roi_per_this_img
        neg_roi_per_this_img = int(min(neg_roi_per_img, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_img, replace=False
            )
        # 选取
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_img:] = 0  # 背景类的标签设为0
        sample_roi = roi[keep_index]

        # 为这些候选框赋予真实的偏移量和标签
        gt_roi_loc = bbox2loc(
            src_bbox=sample_roi, dst_bbox=bbox[gt_assignment[keep_index]]
        )
        gt_roi_loc = (
            (gt_roi_loc - np.array(loc_normalize_mean, np.float32)) /
            np.array(loc_normalize_std, np.float32)
        )
        return sample_roi, gt_roi_loc, gt_roi_label
