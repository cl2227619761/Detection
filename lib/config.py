"""
配置文件
"""


class Config:
    """配置文件类"""
    data_dir = "D:/code/faster_rcnn/VOCdevkit/VOC2007/"
    min_size = 600
    max_size = 1000
    train_split = "trainval"
    val_split = "val"


OPT = Config()
