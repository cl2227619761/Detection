"""
对xml文件进行解析，从中提取出图像，真实框，真实框标签，difficult等信息
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from data.util import read_image


class BboxDataset:
    """
    产生图像，真实框，真实框标签和difficult信息
    参数：
        data_dir: 数据集路径，到VOC2007文件夹路径
        split: 用来区分训练集，验证集，测试集的
        use_difficult: 是否使用difficult的图片
        return_difficult: 对应框是否被标为difficult
    """

    def __init__(
            self, data_dir, split="trainval",
            use_difficult=False, return_difficult=False
    ):
        # 获取图片编号txt的路径
        id_txt_file = os.path.join(
            data_dir, "ImageSets/Main/%s.txt" % split
        )
        # 读取txt中的图片名编号到ids列表中
        self.ids = [id_.strip() for id_ in open(id_txt_file, encoding="utf-8")]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        # import ipdb; ipdb.set_trace()  本步是为了避免检查是否出现.jpg.xml
        anno_path = os.path.join(
            self.data_dir, "Annotations", id_.split(".")[0] + ".xml"
        )
        anno = ET.parse(anno_path)
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall("object"):
            if not self.use_difficult and int(obj.find("difficult").text) == 1:
                continue
            difficult.append(int(obj.find("difficult").text))
            bndbox = obj.find("bndbox")
            bbox.append([
                int(bndbox.find(tag).text) - 1
                for tag in ("ymin", "xmin", "ymax", "xmax")
            ])
            # name = obj.find("name").text.lower().strip()
            # 这步是针对中文标签的
            name = obj.findtext("name").strip()
            label.append(BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        # ---------------加入去除ALL图片中超出边界及噪声框------------------------#
        # import ipdb; ipdb.set_trace()
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], a_min=0, a_max=1200)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], a_min=0, a_max=1920)
        keep_bbox = []
        bbox_hs = bbox[:, 2] - bbox[:, 0]
        bbox_ws = bbox[:, 3] - bbox[:, 1]
        for i in range(len(bbox_hs)):
            if bbox_hs[i] > 50 and bbox_ws[i] > 50:
                keep_bbox.append(i)
        bbox = bbox[keep_bbox]
        label = np.stack(label).astype(np.int32)
        label = label[keep_bbox]
        difficult = np.stack(difficult).astype(np.bool)
        difficult = difficult[keep_bbox]

        # img_path = os.path.join(self.data_dir, "JPEGImages", id_ + ".jpg")
        img_path = os.path.join(
            self.data_dir, "JPEGImages", id_.split(".")[0] + ".jpg"
        )
        img = read_image(img_path)
        return img, bbox, label, difficult


# 所有的类别名称，总共有20类
BBOX_LABEL_NAMES = (
    '正常',
    '异常',
)

# BBOX_LABEL_NAMES = (
#     'aeroplane',
#     'bicycle',
#     'bird',
#     'boat',
#     'bottle',
#     'bus',
#     'car',
#     'cat',
#     'chair',
#     'cow',
#     'diningtable',
#     'dog',
#     'horse',
#     'motorbike',
#     'person',
#     'pottedplant',
#     'sheep',
#     'sofa',
#     'train',
#     'tvmonitor'
# )


def main():
    """调试"""
    bbox_dataset = BboxDataset(
        data_dir="D:/code/faster_rcnn/VOCdevkit/VOC2007/"
    )
    import ipdb; ipdb.set_trace()
    image, bbox, label, difficult = bbox_dataset[0]


if __name__ == "__main__":
    main()
