"""
本脚本是学习过程中遇到的一些知识点
"""
# pytorch tnt meters的使用
import torch
import torchnet.meter as meter
import numpy as np

# meter.AverageValueMeter返回的是一堆数的均值和标准差
# AVE_METER = meter.AverageValueMeter()
# for i in range(1, 10):
#     AVE_METER.add(i)
# mean, std = AVE_METER.value()
# print(AVE_METER.value())
# print(mean, std)

# meter.APMeter计算每一类的ap
# ap_meter = meter.APMeter()

# target = torch.Tensor([[0, 0, 0, 1]])
# output = torch.Tensor([[0.1, 0.2, 0.3, 0.4]])
# import ipdb; ipdb.set_trace()
# ap_meter.add(output, target)
# ap = ap_meter.value()
# print(ap)

# meter.ConfusionMeter计算混淆矩阵
confusion_meter = meter.ConfusionMeter(3)
predicted = torch.Tensor([[0.1, 0.4, 0.5], [0.3, 0.2, 0.5], [0.6, 0.2, 0.2]])
target = torch.Tensor([0, 2, 1])
confusion_meter.add(predicted, target)
print(confusion_meter.value())