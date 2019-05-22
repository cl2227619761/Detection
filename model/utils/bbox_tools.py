"""
框相关的函数
"""
import sys
sys.path.append("../../")

import numpy as np

from lib.config import OPT


def generate_anchor_base(
        base_size=OPT.base_size, ratios=OPT.ratios,
        anchor_scales=OPT.anchor_scales
):
    """在原图的第一个锚点位置产生相应的9个锚点框"""
    ctr_y = base_size / 2.
    ctr_x = base_size / 2.
    anchor_base = np.zeros(
        (len(ratios) * len(anchor_scales), 4),
        dtype=np.float32
    )
    index = 0
    for i, _ in enumerate(ratios):
        for j, _ in enumerate(anchor_scales):
            height = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            width = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            anchor_base[index, 0] = ctr_y - height / 2.
            anchor_base[index, 1] = ctr_x - width / 2.
            anchor_base[index, 2] = ctr_y + height / 2.
            anchor_base[index, 3] = ctr_x + width / 2.
            index += 1
    return anchor_base


# 在原图所有锚点处产生锚点框
def enumerate_anchors(anchor_base, sub_sample, height, width):
    """产生所有的锚点框"""
    shift_y = np.arange(0, height * sub_sample, sub_sample)
    shift_x = np.arange(0, width * sub_sample, sub_sample)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack(
        (shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()),
        axis=1
    )
    n_per_anchor = anchor_base.shape[0]
    num_anchors = shift.shape[0]
    anchors = anchor_base.reshape((1, n_per_anchor, 4)) + \
        shift.reshape((1, num_anchors, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((n_per_anchor * num_anchors, 4))
    return anchors


# 由偏移量换算为坐标
def loc2bbox(base_box, locs):
    """由偏移量换算为坐标，要注意是相对于谁的偏移量"""
    base_h = base_box[:, 2] - base_box[:, 0]
    base_w = base_box[:, 3] - base_box[:, 1]
    base_ctr_y = base_box[:, 0] + 0.5 * base_h
    base_ctr_x = base_box[:, 1] + 0.5 * base_w

    d_y = locs[:, 0::4]
    d_x = locs[:, 1::4]
    d_h = locs[:, 2::4]
    d_w = locs[:, 3::4]

    ctr_y = d_y * base_h[:, np.newaxis] + base_ctr_y[:, np.newaxis]
    ctr_x = d_x * base_w[:, np.newaxis] + base_ctr_x[:, np.newaxis]
    height = np.exp(d_h) * base_h[:, np.newaxis]
    width = np.exp(d_w) * base_w[:, np.newaxis]

    dst_box = np.zeros(locs.shape, dtype=locs.dtype)
    dst_box[:, 0::4] = ctr_y - 0.5 * height
    dst_box[:, 1::4] = ctr_x - 0.5 * width
    dst_box[:, 2::4] = ctr_y + 0.5 * height
    dst_box[:, 3::4] = ctr_x + 0.5 * width

    return dst_box


# 由坐标换算为偏移量
def bbox2loc(box, base_box):
    """将坐标变为偏移量"""
    box_h = box[:, 2] - box[:, 0]
    box_w = box[:, 3] - box[:, 1]
    box_ctr_y = box[:, 0] + 0.5 * box_h
    box_ctr_x = box[:, 1] + 0.5 * box_w

    base_box_h = base_box[:, 2] - base_box[:, 0]
    base_box_w = base_box[:, 3] - base_box[:, 1]
    base_box_y = base_box[:, 0] + 0.5 * base_box_h
    base_box_x = base_box[:, 1] + 0.5 * base_box_w

    eps = np.finfo(base_box_h.dtype).eps
    base_box_h = np.maximum(base_box_h, eps)
    base_box_w = np.maximum(base_box_w, eps)

    d_y = (box_ctr_y - base_box_y) / base_box_h
    d_x = (box_ctr_x - base_box_x) / base_box_h
    d_h = np.log(box_h / base_box_h)
    d_w = np.log(box_w / base_box_w)

    res = np.stack((d_y, d_x, d_h, d_w), axis=1)
    return res


def bbox_iou(bbox_a, bbox_b):
    """计算相交框的iou"""
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError("框的坐标必须时4列")
    # 相交的框的左上角的坐标(y1, x1)
    # 这种变形的目的是用a和每个b进行比较
    t_l = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # 相交的框的右下角的坐标(y2, x2)
    b_r = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    # 相交部分的面积
    area_i = np.prod(b_r - t_l, axis=2) * (t_l < b_r).all(axis=2)
    # bbox_a的面积
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    # bbox_b的面积
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    # 计算iou
    iou = area_i / (area_a[:, None] + area_b - area_i)
    return iou


def main():
    """调试用函数"""
    box = np.random.normal(2, size=(5, 4))
    base_box = np.random.normal(2, size=(2, 4))
    # res = bbox2loc(box, base_box)
    # print("ok")
    import ipdb; ipdb.set_trace()
    iou = bbox_iou(box, base_box)


if __name__ == "__main__":
    main()
