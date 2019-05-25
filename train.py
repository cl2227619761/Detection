"""
训练和测试的脚本
"""
import sys
sys.path.append("./")

from torch.utils.data import DataLoader
import progressbar as pb

from lib.config import OPT
from data.dataset import Dataset, TestDataset
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.utils.array_tool import scalar
from trainer import FasterRCNNTrainer


def train(**kwargs):
    """训练过程"""
    # 加载配置文件中的各种参数设置
    OPT._parse(kwargs)

    # 数据集
    dataset = Dataset(opt=OPT)
    print("加载数据集")
    dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=True,
        num_workers=OPT.num_workers
    )
    # 测试集
    testset = TestDataset(opt=OPT)
    test_dataloader = DataLoader(
        dataset=testset, batch_size=1, shuffle=False,
        num_workers=OPT.num_workers, pin_memory=True
    )
    # 模型
    faster_rcnn = FasterRCNNVGG16()
    print("模型加载完成")
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    best_map = 0  # 最好的map
    lr_ = OPT.lr  # 学习率
    for epoch in range(OPT.epoch):
        trainer.reset_meters()  # 每次epoch的开始将损失函数清零
        for ii, (img, bbox_, label_, scale) in pb.progressbar(
            enumerate(dataloader), max_value=len(dataloader)
        ):
            scale = scalar(scale)  # 原图和处理后的图片之间的一个缩放比例
            img, bbox, label = img.cuda(), bbox_.cuda(), label_.cuda()
            trainer.train_step(
                imgs=img, bboxes=bbox, labels=label, scale=scale
            )
            if (ii + 1) % OPT.plot_every == 0:
                print(trainer.get_meter_data())


def main():
    """调试"""
    train()


if __name__ == "__main__":
    main()

