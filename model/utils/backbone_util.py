"""
本脚本是关于fpn的添加的工具函数
"""
from collections import OrderedDict

from torchvision.models import resnet, resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenBatchNorm2d(torch.jit.ScriptModule):
    """冻结BatchNorm层"""

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    @torch.jit.script_method
    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * rv.rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class IntermediaLayerGetter(nn.ModuleDict):
    """从一个模型中提取中间层，指定要提取的层的名称
    输入：模型和层名
    输出：字典，键为层的名字；值为该层返回的特征
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([
            name for name, _ in model.named_children()
        ]):
            return ValueError("要提取的层在模型中不存在！")

        orig_return_layers = return_layers  # 要提取的层的字典
        layers = OrderedDict()  # 用来存储要提取的层的OrderedDict字典
        for name, module in model.named_children():
            layers[name] = module
            # 遍历return_layers，直到其为空为止
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        # 继承父类ModuleDict的init方法，需要传入一个OrderedDict
        super(IntermediaLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()  # 输出的结果是OrderedDict
        import ipdb; ipdb.set_trace()
        for name, module in self.named_children():  # self为提取出的层模型
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):

class BackboneWithFPN(nn.Sequential):
    """添加FPN到特征提取器"""

    def __init__(
            self, backbone, return_layers, in_channels_list, out_channels
    ):
        body = IntermediaLayerGetter(backbone, return_layers)


def resnet_fpn_backbone(backbone_name):
    """为resnet添加fpn"""
    backebone = resnet.__dict__[backbone_name](
        pretrained=True, norm_layer=FrozenBatchNorm2d
    )
    for name, param in backebone.named_paramters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in  \
                name:
            param.requires_grad_(False)

    return_layers = {"layer1": 0, "layer2": 1, "layer3": 2, "layer4": 3}

    in_channels_stage2 = 256  # 这是rpn中间层的卷积层输入通道数
    # 这是fpn特征们的通道数
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8
    ]
    out_channels = 256  # 经过rpn卷积层的输出通道数为256


def main():
    """调试用函数"""
    # backbone = resnet_fpn_backbone("resnet18")
    model = resnet18(pretrained=True)
    img = torch.rand(1, 3, 224, 224)
    return_layers = {"layer1": "feat1", "layer2": "feat2"}
    new_model = IntermediaLayerGetter(model, return_layers)
    out = new_model(img)


if __name__ == "__main__":
    main()

