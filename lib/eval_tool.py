"""
本脚本提供了ap和mAP的计算过程
"""
import itertools
from collections import defaultdict
import sys
sys.path.append("../")

from model.utils.bbox_tools import bbox_iou

import numpy as np


def calc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5
):
    """计算每一类的precision和recall
    参数：
        iou_thresh: 预测和真实框的重叠iou达到该阈值则为预测正确
    """
    # 将标签坐标等转变为迭代器
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(gt_difficults)
    else:
        gt_difficults = iter(gt_difficults)

    # 非背景类的数量
    n_pos = defaultdict(int)
    # 得分
    score = defaultdict(list)
    # 预测和真实框匹配
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
        zip(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults
        ):
            if gt_difficult is None:
                gt_difficult = np.zeros((gt_bbox.shape[0]), dtype=bool)
            for l in np.unique(
                np.concatenate((pred_label, gt_label)).astype(int)
            ):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # 按照得分进行排序
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                # 对真实框也按照类别进行整理
                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                # 剔除difficult=1的框
                n_pos[l] += np.logical_not(gt_difficult_l).sum()
                score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    match[l].extend((0,) * pred_bbox_l.shape[0])
                    continue

                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1
                iou = bbox_iou(pred_bbox_l, gt_bbox_l)
                # 较大iou对应的真实框的索引
                gt_index = iou.argmax(axis=1)
                gt_index[iou.max(axis=1) < iou_thresh] = -1
                del iou

                # 预测框是否被选中
                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        # gt_idx=-1的那些框是iou小于阈值的，舍去
                        if gt_difficult_l[gt_idx]:
                            # 这个真实框是否为difficult
                            match[l].append(-1)
                        else:
                            # 如果不是difficult
                            if not selec[gt_idx]:
                                # 如果没被选过
                                match[l].append(1)
                            else:
                                # 如果被选过
                                match[l].append(0)
                        selec[gt_idx] = True
                    else:
                        match[l].append(0)
    n_fg_class = max(n_pos.keys()) + 1  # 前景类的个数
    # prec和rec的初始值
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (tp + fp)
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec


def calc_ap(prec, rec):
    """计算每一类的ap值"""
    n_fg_class = len(prec)  # 前景的类别
    ap = np.empty(n_fg_class)  # 每一类的初始ap为0
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue
        # 将prec从添加上0,0
        mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
        # 将rec添加上0, 1
        mrec = np.concatenate(([0], rec[l], [1]))

        mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # 将mpre先反排累加在反排
        # pr曲线的拐点处
        i = np.where(mrec[1:] != mpre[:-1])[0]
        # 计算ap，其实就是矩形面积
        ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_map(
    pred_bboxes, pred_labels, pred_scores,
    gt_bboxes, gt_labels, gt_difficults=None,
    iou_thresh=0.5
):
    """计算mAP"""
    prec, rec = calc_prec_rec(
        pred_bboxes=pred_bboxes, pred_labels=pred_labels,
        pred_scores=pred_scores, gt_bboxes=gt_bboxes,
        gt_labels=gt_labels, gt_difficults=gt_difficults,
        iou_thresh=iou_thresh
    )
    ap = calc_ap(prec, rec)

    return {"ap": ap, "mAP": np.nanmean(ap)}

