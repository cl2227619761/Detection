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
# confusion_meter = meter.ConfusionMeter(3)
# predicted = torch.Tensor([[0.1, 0.4, 0.5], [0.3, 0.2, 0.5], [0.6, 0.2, 0.2]])
# target = torch.Tensor([0, 2, 1])
# confusion_meter.add(predicted, target)
# print(confusion_meter.value())


# 关于*args和**kwargs的用法
# python支持可变参数，例子
# def test_args(first, *args):
#     print("我是用来测试*args的用法的函数")
#     print("必须参数:", first)
#     # 遍历*args里面的元素需要使用循环，注意循环的使用方法
#     for v in args:
#         print("可选参数:", v)


# test_args(1, 2, 3, 4)


# def test_kwargs(first, *args, **kwargs):
#     print("我是用来测试**kwargs的函数")
#     print("必须参数:", first)
#     for v in args:
#         print("可选参数(*args):", v)
#     for k, v in kwargs.items():
#         # 注意遍历kwargs需要用items，一次包含两个元素
#         print("可选参数 %s (*kwargs): %s" % (k, v))


# test_kwargs(1, 2, 3, k1="张三", k2="李四")


# python中的hasattr(), getattr(), setattr()函数的使用
# hasattr(object, name): 判断一个对象里是否有name属性或者name方法，返回的是布尔值
class Test:

    name = "张三"

    def run(self):
        print("hello hasattr!")


test = Test()
# 判断test对象是否有name属性
print("test是否有name属性:", hasattr(test, "name"))
# 判断test对象是否有run方法
print("test是否有run方法:", hasattr(test, "run"))
# getattr(object, name[,default])获取对象的属性或者方法，如果存在就打印出来；如果不存在
# 打印出默认值，默认值可选。注意，如果返回方法，返回的是方法的内存地址，如果要运行该方法，可以
# 在后面加上括号
print("获取name属性:", getattr(test, "name"))  # 获取name属性，如果存在，就打印出来
print("获取run方法地址:", getattr(test, "run"))
getattr(test, "run")()  # 获取run方法并运行
print("获取一个属性，若不存在，返回默认值:", getattr(test, "age", 18))

# setattr(object, name, value): 给对象的属性赋值，如果不存在，先创建再赋值
print("age属性是否存在:", hasattr(test, "age"))
print("若age属性不存在，先创建再赋值:", setattr(test, "age", 18))
print("age属性现在是否存在:", hasattr(test, "age"))
print("age的值为:", getattr(test, "age"))
print(getattr(test, "name", None))  # 获取name属性，若不存在，则返回None; 若存在。。。
print(getattr(test, "name2", None))