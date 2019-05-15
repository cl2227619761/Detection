"""
配置文件
"""


class Config:
    """配置文件类"""
    # 数据集相关
    data_dir = "D:/code/faster_rcnn/VOCdevkit/VOC2007/"
    min_size = 600
    max_size = 1000
    train_split = "train"
    valtest_split = "valtest"
    # 锚点框生成相关
    base_size = 16
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]
    # 网络相关
    sub_sample = 16


OPT = Config()
