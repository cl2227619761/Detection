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

    # 分类相关
    n_fg_class = 20  # 前景类别的总数量

    # 特征提取网络相关
    use_drop = False  # 是否使用dropout

    # 网络超参数
    lr = 1e-3
    weight_decay = 0.0005  # 权重衰减
    use_adam = False  # 是否使用adam优化器
    # 损失计算相关的超参数，这样设置的用意是什么？
    rpn_sigma = 3.
    roi_sigma = 1.


OPT = Config()
