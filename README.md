# Detection

faster rcnn pytorch版本，对自定义数据进行训练

## 文件树：

-data  

    --bbox_dataset.py: 从xml文件提取出bbox, label, difficult等信息
    --dataset.py: 利用bbox_dataset.py以及util.py里面的工具，生成数据集的函数
    --util.py: 对图片进行变换以及预处理的工具函数
-lib  

    --config.py: 配置文件
    --csv_tool.py: 用来将损失函数，mAP指标读入csv的工具函数
    --eval_tool.py: 用来计算mAP的工具函数
    --vis_tool.py: visdom相关的可视化工具
-model  

    --utils
        ---array_tool.py: numpy和tensor之间进行转换的函数工具
        ---bbox_tools.py: iou等与bbox有关的工具函数
        ---creator_tool.py: 锚点框生成等框生成有关的工具函数
        ---nms.py: 非极大值抑制函数
    --faster_rcnn.py: faster_rcnn基类，供扩展
    --rpn.py: rpn网络函数
    --roi_module.py: roi pooling函数
    --faster_rcnn_vgg16.py: 特征提取网络为VGG16的faster_rcnn，继承于faster_rcnn

-trainer.py: 将训练过程封装进了一个类里面  
-train.py: 最终启动训练，验证，测试的脚本  
-results  

    --saved_model: 用来放训练好的模型

## 训练过程(...为需要配置的参数，参考config.py)：

python train.py train --...
