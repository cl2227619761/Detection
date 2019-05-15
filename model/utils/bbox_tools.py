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
