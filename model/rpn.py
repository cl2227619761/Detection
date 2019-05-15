"""
region proposal network
"""
import sys
sys.path.append("../")

import numpy as np
import torch.nn as nn

from utils.bbox_tools import generate_anchor_base, enumerate_anchors
from lib.config import OPT


class RegionProposalNetwork(nn.Module):
    """候选框生成网络"""

    def __init__(
            self, feature_channels=512, mid_channels=512,
            ratios=OPT.ratios, anchor_scales=OPT.anchor_scales,
            sub_sample=OPT.sub_sample

    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base()