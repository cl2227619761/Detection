"""
配置文件
"""
from pprint import pprint


class Config:
    """配置文件类"""
    # 数据集相关
    data_dir = "D:/code/ALL_data/VOC2007/"
    min_size = 600
    max_size = 1000
    train_split = "ALL_train"
    valtest_split = "ALL_valtest"

    # 线程数
    num_workers = 8
    # 锚点框生成相关
    base_size = 16
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]
    # 网络相关
    sub_sample = 16

    # 分类相关
    n_fg_class = 2  # 前景类别的总数量

    # 特征提取网络相关
    use_drop = False  # 是否使用dropout

    # 网络超参数
    lr = 1e-3
    epoch = 14
    weight_decay = 0.0005  # 权重衰减
    use_adam = False  # 是否使用adam优化器
    # 损失计算相关的超参数，这样设置的用意是什么？
    rpn_sigma = 3.
    roi_sigma = 1.

    roi_size = 7

    # 绘制损失函数图相关
    plot_every = 40

    # 计算mAP的相关参数
    test_num = 51

    def _state_dict(self):
        """"返回Config对象的属性，不包括那些以下划线开头的属性。返回的是字典"""
        return {
            k: getattr(self, k) for k, _ in Config.__dict__.items()
            if not k.startswith("_")
        }

    def _parse(self, kwargs):
        """解析参数
        其实**kwargs是要写的一些参数
        """
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError("未知参数: --%s" % k)
            setattr(self, k, v)

        print("===========用户配置文件==============")
        pprint(self._state_dict())
        print("===========配置文件加载结束===========")


OPT = Config()
