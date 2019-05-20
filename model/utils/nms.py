"""
非极大值抑制的相关脚本
"""
import numpy as np
import cupy as cp


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    """非极大值抑制"""
    bbox_y1 = bbox[:, 0]
    bbox_x1 = bbox[:, 1]
    bbox_y2 = bbox[:, 2]
    bbox_x2 = bbox[:, 3]

    area = (bbox_x2 - bbox_x1 + 1) * (bbox_y2 - bbox_y1 + 1)
    n_bbox = bbox.shape[0]

    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = cp.arange(n_bbox, dtype=np.int32)
    keep = []

    # 预测框之间进行两两比较，去除重叠面积iou大于thresh的框
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = cp.maximum(bbox_x1[i], bbox_x1[order[1:]])
        yy1 = cp.maximum(bbox_y1[i], bbox_y1[order[1:]])
        xx2 = cp.minimum(bbox_x2[i], bbox_x2[order[1:]])
        yy2 = cp.minimum(bbox_y2[i], bbox_y2[order[1:]])

        width = cp.maximum(0., (xx2 - xx1 + 1))
        height = cp.maximum(0., (yy2 - yy1 + 1))
        inter = width * height
        iou = inter / (area[i] + area[order[1:]] - inter)
        index = cp.where(iou <= thresh)[0]
        order = order[index + 1]
    if limit is not None:
        keep = keep[:limit]
    return cp.asnumpy(keep)
