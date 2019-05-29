"""
本脚本是关于visdom可视化的
"""
import visdom


class Visualizer:
    """将可视化过程封装到了一个类里面"""

    def __init__(self, env="default", **kwargs):
        """设置visdom配置"""
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs  # 其他参数

        self.index = {}  # 比如，{"loss", 23}，表示loss的第23个值
        self.log_text = ''

    def reinit(self, env="default", **kwargs):
        """修改visdom配置"""
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self