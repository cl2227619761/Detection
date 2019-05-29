"""
本脚本将损失函数和mAP写入csv中
"""
import csv

import pandas as pd
import numpy as np


def loss_writer(loss_dict, out_path):
    """将产生的loss写入到csv文件中
    loss_dict：字典构成的列表
    """
    print("====写入损失函数====")
    header = list(loss_dict[0].keys())  # 这是csv的header
    with open(out_path, "w", newline="") as loss_file:
        writer = csv.DictWriter(loss_file, header)
        writer.writeheader()
        for row in loss_dict:
            writer.writerow(row)


def map_writer(map_data, out_path):
    """将产生的mAP写入到csv文件中
    map_data: mAP值组成的列表
    """
    print("====写入mAP====")
    map_array = np.array(map_data)
    header = ["mAP"]
    dataframe = pd.DataFrame(data=map_array, columns=header)
    dataframe.to_csv(out_path, index=None)


def main():
    """调试用"""
    map_data = [10, 20, 30]
    map_writer(map_data, "./test.csv")
    loss_dict = [
        {"rpn_loc_loss": 0.2, "rpn_cls_loss": 0.3},
        {"rpn_loc_loss": 0.1, "rpn_cls_loss": 0.4},
    ]
    loss_writer(loss_dict, "./loss.csv")


if __name__ == "__main__":
    main()
