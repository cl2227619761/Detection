"""
生成数据集
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../")

from data.bbox_dataset import BboxDataset
from data.util import Transform, preprocess, resize_bbox
from lib.config import OPT


class Dataset:
    """训练集dataset"""

    def __init__(self, opt=OPT):
        self.opt = opt
        self.dataset = BboxDataset(
            data_dir=opt.data_dir, split=opt.train_split
        )
        self.transform = Transform(
            min_size=opt.min_size, max_size=opt.max_size
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ori_img, bbox, label, _ = self.dataset[index]
        img, bbox, label, scale = self.transform(
            in_data=(ori_img, bbox, label)
        )
        # 这里copy的目的是为了防止出现ValueError: some of the strides of
        # a given numpy array are negative. This is currently not supported,
        # but will be added in future releases的错误
        return img.copy(), bbox.copy(), label.copy(), scale


class TestDataset:
    """验证集和测试集dataset"""

    def __init__(self, opt=OPT):
        self.opt = opt
        self.dataset = BboxDataset(
            data_dir=opt.data_dir, split=opt.valtest_split,
            use_difficult=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ori_img, gt_bbox, label, difficult = self.dataset[index]
        height, width = ori_img.shape[1:]
        img = preprocess(ori_img)
        new_height, new_width = img.shape[1:]
        bbox = resize_bbox(
            bbox=gt_bbox, in_size=(height, width),
            out_size=(new_height, new_width)
        )
        return img, ori_img.shape[1:], gt_bbox, bbox, label, difficult


def main():
    """调试用"""
    dataset = Dataset()


if __name__ == "__main__":
    main()
