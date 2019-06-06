"""
本脚本是关于fpn的添加的工具函数
"""
from collections import OrderedDict

from torchvision.models import resnet18
import resnet
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
        return_layers = {k: v for k, v in return_layers.items()}
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
        for name, module in self.named_children():  # self为提取出的层模型
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()  # 用来放置1x1卷积层
        self.layer_blocks = nn.ModuleList()  # 用来放置3x3卷积层
        for in_channels in in_channels_list:
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(
                out_channels, out_channels, 3, padding=1
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())

        # 最顶层特征图进行1x1卷积改变通道数
        last_inner = self.inner_blocks[-1](x[-1])
        results = []  # 用来放置每一层金字塔处理后得到的特征
        results.append(self.layer_blocks[-1](last_inner))

        # 遍历除了顶层特征图以外的所有特征图，进行特征图的整合
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1],
                self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            # 进行横向的1x1卷积操作
            inner_lateral = inner_block(feature)
            # 得到特征图尺寸，用于上采样插值
            feat_shape = inner_lateral.shape[-2:]
            # 上采样过程
            inner_top_down = F.interpolate(
                input=last_inner, size=feat_shape, mode="nearest"
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out  # 返回的是金字塔的每一层的经过特征整合之后的特征


class ExtraFPNBlock(nn.Module):
    """在金字塔之后接的额外的层"""

    def forward(self, results, x, names):
        pass


class LastLevelMaxPool(ExtraFPNBlock):
    """继承ExtraFPNBlock，接的是最大池化"""

    def forward(self, results, x, names):
        names.append("pool")
        results.append(F.max_pool2d(results[-1], 1, 2, 0))
        return results, names


class BackboneWithFPN(nn.Sequential):
    """添加FPN到特征提取器"""

    def __init__(
            self, backbone, return_layers, in_channels_list, out_channels
    ):
        body = IntermediaLayerGetter(backbone, return_layers)
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        # body和fpn都是OrderedDict，对其进行合并，得到加了FPN的特征提取器
        super(BackboneWithFPN, self).__init__(OrderedDict([
            ("body", body), ("fpn", fpn)
        ]))
        self.out_channels = out_channels


def resnet_fpn_backbone(backbone_name):
    """为resnet添加fpn"""
    backbone = resnet.__dict__[backbone_name](
        pretrained=True, norm_layer=FrozenBatchNorm2d
    )
    for name, param in backbone.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in  \
                name:
            param.requires_grad_(False)

    return_layers = {"layer1": 0, "layer2": 1, "layer3": 2, "layer4": 3}

    in_channels_stage2 = 64
    # 这是fpn特征们的通道数
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8
    ]
    out_channels = 256  # 经过rpn卷积层的输出通道数为256
    return BackboneWithFPN(
        backbone=backbone, return_layers=return_layers,
        in_channels_list=in_channels_list, out_channels=out_channels
    )


def main():
    """调试用函数"""
    # backbone = resnet_fpn_backbone("resnet18")
    # model = resnet.__dict__["resnet18"](pretrained=True)
    # img = torch.rand(1, 3, 224, 224)
    # return_layers = {"layer1": "feat1", "layer2": "feat2"}
    # new_model = IntermediaLayerGetter(model, return_layers)
    # import ipdb; ipdb.set_trace()
    # out = new_model(img)
    backbone = resnet_fpn_backbone("resnet18")
    img = torch.rand(1, 3, 223, 224)
    import ipdb; ipdb.set_trace()
    feat = backbone(img)


if __name__ == "__main__":
    main()

