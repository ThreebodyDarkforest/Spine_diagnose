# 脊柱识别与诊断

## 简介

### 项目背景

本项目的背景介绍如下：

> 脊柱退化性疾病，如腰椎间盘突出等症状正呈现出年轻化的趋势，困扰着老年人群体和办公族，正确的预防和干预能够有效防止疾病的恶化。核磁（MRI）作为非侵入式检查手段，对软组织成像好，无辐射，对肌肉骨骼疾病的特异性和敏感度较高，适合对普通人群的常规检查，是预防脊柱退化性疾病的可靠检查手段。同时临床上，对脊柱退化性疾病的诊断的一致性有待提高，人工智能算法在临床流程中可以帮助提高诊断的一致性和可量化性，对量化评估针对脊柱退化性疾病的干预效果有着很高的价值。本次大赛将召集全球开发者利用人工智能技术探索高效准确的脊柱退化性疾病自动诊断，并积极推动相关技术的临床应用。
>
> 本次挑战的目标是通过人工智能算法自动分析核磁共振影像来检测和分类脊柱的退行性改变，包括正常椎体、退变椎体、正常椎间盘、椎间盘突出、膨出等特征。参与者需要提供全自动算法来定位椎体和椎间盘的位置和相应分类。

如果需要更多详细信息，请参阅[Spark“数字人体”AI挑战赛-脊柱疾病智能诊断大赛](https://tianchi.aliyun.com/competition/entrance/531796/introduction?lang=zh-cn)。

本项目待解决的问题可以简要描述任务如下：

- 给定一张T2失状位脊柱CT图像，完成对脊椎骨的定位识别和病变情况诊断
    - 定位识别：CT图像含有11块待识别的椎骨，它们又可细分为6块椎间盘和5块椎骨
    - 病变诊断：除了需要检测和定位CT图像中11块椎骨，还需要分别对它们的病变情况进行多分类

Spinal Disease Dataset脊柱疾病数据集详情请见[链接](https://tianchi.aliyun.com/dataset/dataDetail?dataId=79463)。

### 项目介绍

本项目的目录树如下：

```
.
├── configs
│   ├── yolov6s6_finetune.py
│   └── yolov6s.py
├── data
│   ├── images
│   └── spine.yaml
├── LICENSE
├── modules
│   ├── adapter.py
│   ├── classifier.py
│   ├── components
│   ├── detector.py
│   ├── resnest
│   ├── resnet
│   ├── util.py
│   ├── vit
│   └── yolov6
├── README.md
├── requirements.txt
├── test
│   ├── middle_test.ipynb
│   ├── models
│   ├── Resnet_infer.py
│   ├── resnet_test.ipynb
│   ├── Resnet_train.py
│   ├── swin_train.py
│   └── yolo_infer.py
├── tools
│   └── gen_yolo_data.py
└── weights
```

其中 `modules` 模块中实现了项目的核心功能。

## 快速开始

由于本项目暂不完善，您需要通过以下方式运行测试代码。

- 克隆仓库

```
git clone https://www.github.com/ThreebodyDarkforest/Spine_diagnose
cd Spine_diagnose
```

- 下载模型文件

您可以通过以下链接下载本项目需要的文件，并将它们存放在 `Spine_diagnose/weights` 目录下。

- 运行测试代码

需要注意的是，我们尚未完成真正用于测试的代码。一个可运行的测试代码位于 `Spine_diagnose/modules/adapter.py`，您只需要直接运行它：

```
python modules/adapter.py
```

- 获取结果

如果没有发生错误，您将看到类似下面的内容。

```
$ python modules/adapter.py

Loading checkpoint from /home/marcus/Learn/DL/Spine_diagnose/weights/detect.pt

Fusing model...
Detection done. start classifying now...
/home/marcus/Learn/DL/Spine_diagnose/modules/classifier.py:41: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  return label, (torch.max(F.softmax(logits[:, :6]), dim=1)[0].item(), torch.max(F.softmax(logits[:, 6:]), dim=1)[0].item())
Classification done.
[{'xyxy': ((125, 179), (151, 191)), 'label': 'L5-S1 v2', 'class_num': 10, 'confidence': 0.5239071628414889}, {'xyxy': ((117, 162), (149, 184)), 'label': 'L5 v2', 'class_num': 4, 'confidence': 0.7836284251675887}, {'xyxy': ((118, 154), (144, 166)), 'label': 'L4-L5 v2', 'class_num': 9, 'confidence': 0.5585730742222171}, {'xyxy': ((113, 134), (147, 159)), 'label': 'L4 v1', 'class_num': 3, 'confidence': 0.4727259708481813}, {'xyxy': ((118, 128), (144, 140)), 'label': 'L3-L4 v2', 'class_num': 8, 'confidence': 0.6851188511207023}, {'xyxy': ((116, 109), (149, 135)), 'label': 'L3 v2', 'class_num': 2, 'confidence': 0.8005590789324253}, {'xyxy': ((123, 102), (148, 114)), 'label': 'L2-L3 v2', 'class_num': 7, 'confidence': 0.7152538218377865}, {'xyxy': ((122, 84), (153, 107)), 'label': 'L2 v2', 'class_num': 1, 'confidence': 0.7543958538411826}, {'xyxy': ((128, 78), (153, 91)), 'label': 'L1-L2 v1', 'class_num': 6, 'confidence': 0.4898075383329039}, {'xyxy': ((127, 61), (160, 84)), 'label': 'L1 v2', 'class_num': 0, 'confidence': 0.7516732347548739}, {'xyxy': ((132, 55), (160, 68)), 'label': 'T12-L1 v2', 'class_num': 5, 'confidence': 0.78073688683909}]
done.
```