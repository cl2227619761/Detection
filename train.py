"""
训练和测试的脚本
"""
import sys
sys.path.append("./")

import torch
from torch.utils.data import DataLoader
import progressbar as pb

from lib.config import OPT
from data.dataset import Dataset, TestDataset
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.utils.array_tool import scalar
from trainer import FasterRCNNTrainer
from lib.eval_tool import calc_map


def evaluate(dataloader, faster_rcnn, test_num=10000):
    """测试过程"""
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    # 遍历测试集或者验证集
    for ii, (
        imgs, sizes, gt_bboxes_, _, gt_labels_, gt_difficults_
    ) in pb.progressbar(enumerate(dataloader), max_value=len(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(
            imgs, [sizes]
        )
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break
    results = calc_map(
        pred_bboxes=pred_bboxes, pred_labels=pred_labels,
        pred_scores=pred_scores, gt_bboxes=gt_bboxes,
        gt_labels=gt_labels, gt_difficults=gt_difficults
    )
    return results


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
        print("Epoch: %s/%s" % (epoch, OPT.epoch - 1))
        print("-" * 10)
        trainer.reset_meters()  # 每次epoch的开始将损失函数清零
        for ii, (img, bbox_, label_, scale) in pb.progressbar(
            enumerate(dataloader), max_value=len(dataloader)
        ):
            scale = scalar(scale)  # 原图和处理后的图片之间的一个缩放比例
            img, bbox, label = img.cuda(), bbox_.cuda(), label_.cuda()
            trainer.train_step(
                imgs=img, bboxes=bbox, labels=label, scale=scale
            )
        print("train:", trainer.get_meter_data())
            # if (ii + 1) % OPT.plot_every == 0:
            #     print(trainer.get_meter_data())
        trainer.eval()
        for jj, (img, size, _, bbox, label, _) in pb.progressbar(
            enumerate(test_dataloader), max_value=len(test_dataloader)
        ):
            img, bbox, label = img.cuda(), bbox.cuda(), label.cuda()
            trainer.val_step(img, size, bbox, label)
        print("val:", trainer.get_meter_data())
        eval_result = evaluate(
            dataloader=test_dataloader, faster_rcnn=faster_rcnn,
            test_num=OPT.test_num
        )
        print("mAP: %.4f" % eval_result["mAP"])
        print()
        trainer.train()


# def main():
#     """调试"""
#     train()
    # testset = TestDataset(opt=OPT)
    # test_dataloader = DataLoader(
    #     dataset=testset, batch_size=1, shuffle=False,
    #     num_workers=OPT.num_workers, pin_memory=True
    # )
    # faster_rcnn = FasterRCNNVGG16()
    # print("模型加载完成")
    # trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    # result = evaluate(test_dataloader, faster_rcnn, test_num=10)
    # import ipdb; ipdb.set_trace()


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    import fire

    fire.Fire()
