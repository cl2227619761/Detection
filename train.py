"""
训练和测试的脚本
"""
import sys
sys.path.append("./")
import csv

import torch
from torch.utils.data import DataLoader
import progressbar as pb
import numpy as np

from lib.config import OPT
from lib.csv_tool import loss_writer, map_writer
from data.dataset import Dataset, TestDataset
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.faster_rcnn_resnet import FasterRCNNResNet16
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
    valtestset = TestDataset(opt=OPT)
    np.random.seed(1234)
    split_seq = np.random.choice(
        range(len(valtestset)), size=int(0.5 * len(valtestset)), replace=False
    )
    valset = torch.utils.data.Subset(valtestset, indices=split_seq)
    testset = torch.utils.data.Subset(
        valtestset, indices=[
            i for i in range(len(valtestset))
            if i not in split_seq
        ]
    )
    val_dataloader = DataLoader(
        dataset=valset, batch_size=1, shuffle=False,
        num_workers=OPT.num_workers, pin_memory=True
    )
    test_dataloader = DataLoader(
        dataset=testset, batch_size=1, shuffle=False,
        num_workers=OPT.num_workers, pin_memory=True
    )
    # 模型
    # faster_rcnn = FasterRCNNVGG16()
    faster_rcnn = FasterRCNNResNet16()
    print("模型加载完成")
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    train_loss = []
    val_loss = []
    val_mAP = []

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
        train_loss.append(trainer.get_meter_data())
        print("train:", trainer.get_meter_data())
            # if (ii + 1) % OPT.plot_every == 0:
            #     print(trainer.get_meter_data())
        trainer.eval()
        for jj, (img, size, _, bbox, label, _) in pb.progressbar(
            enumerate(val_dataloader), max_value=len(val_dataloader)
        ):
            img, bbox, label = img.cuda(), bbox.cuda(), label.cuda()
            trainer.val_step(img, size, bbox, label)
        val_loss.append(trainer.get_meter_data())
        print("val:", trainer.get_meter_data())
        eval_result = evaluate(
            dataloader=val_dataloader, faster_rcnn=faster_rcnn,
            test_num=len(val_dataloader)
        )
        val_mAP.append(eval_result["mAP"])
        print("mAP: %.4f" % eval_result["mAP"])
        print()
        trainer.train()

        # 将损失函数，map写入csv文件
        loss_writer(train_loss, out_path=OPT.train_loss_csv_path)
        loss_writer(val_loss, OPT.val_loss_csv_path)
        map_writer(val_mAP, out_path=OPT.map_csv_path)

        if eval_result["mAP"] > best_map:
            best_map = eval_result["mAP"]
            best_path = trainer.save(
                save_path=OPT.model_save_path, best_map=best_map
            )
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(OPT.lr_decay)
    print("===========训练结束============")
    print("===========开始测试=============")
    trainer.load(best_path)
    trainer.eval()
    test_result = evaluate(
        dataloader=test_dataloader, faster_rcnn=trainer.faster_rcnn,
        test_num=len(test_dataloader)
    )
    print("==========测试结束===========")
    print("best mAP: %.4f" % best_map)
    print("test mAP: %.4f" % test_result["mAP"])


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


if __name__ == "__main__":
    import fire

    fire.Fire()
