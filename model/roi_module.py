"""
roi pooling的函数
"""
import torch
import torch.nn as nn


class RoIPooling2D(nn.Module):
    """roi pooling"""

    def __init__(self, out_h, out_w, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((out_h, out_w))
        self.spatial_scale = spatial_scale

    def forward(self, x, indices_and_rois):
        """x是feature"""
        output = []
        rois = indices_and_rois.data.float()
        rois[:, 1:].mul_(self.spatial_scale)
        rois = rois.long()
        num_rois = rois.size(0)
        for i in range(num_rois):
            # 对每一个roi都进行计算
            roi = rois[i]
            im_idx = roi[0]
            roi_feature = x.narrow(0, im_idx, 1)[
                ..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)
            ]
            output.append(self.adaptive_max_pool(roi_feature))
        output = torch.cat(output, 0)
        return output

