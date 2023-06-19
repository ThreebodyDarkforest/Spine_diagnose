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

## 快速开始

本项目提供了快速部署模型推理的方式。

- 下载模型文件

您需要首先通过以下链接下载本项目需要的文件，并将它们存放在 `Spine_diagnose/weights` 目录下。

权重文件下载链接：https://mega.nz/file/sHVQGQLI#oiD6etmKEnG4ZQZBztNcoa3WWRYIcsbpT2I3CNsOh4s

- 获取结果

本项目提供了一键运行的网页Demo，您可以直接运行 `webui.py` 并在浏览器打开 `http://localhost:7860` 获取交互式页面。

```
python webui.py
```

## 推理

我们建议您通过如下方式运行模型推理。

- 克隆仓库

```
git clone https://www.github.com/ThreebodyDarkforest/Spine_diagnose
cd Spine_diagnose
```

- 安装依赖

```
pip install -r requirements.txt
```

- 下载模型文件

您可以通过以下链接下载本项目需要的文件，并将它们存放在 `Spine_diagnose/weights` 目录下。

权重文件下载链接：https://mega.nz/file/sHVQGQLI#oiD6etmKEnG4ZQZBztNcoa3WWRYIcsbpT2I3CNsOh4s

- 运行推理获取结果

直接运行推理（无结果）：

```
python predict.py --source img_path / img_dir --device 0
```

如果您希望获取推理结果的详细内容，我们提供了如下两种方式

```
# 保存推理结果文本
python predict.py --source img_path / img_dir --save-dir runs/inference --save-txt --device 0
```

```
# 保存推理结果图像（包含检测框和分类结果）
python predict.py --source img_path / img_dir --save-dir runs/inference --save-img --device 0
```

当然，您也可以组合上述两种方法，同时保存两种推理结果。

> 如果需要查看运行时结果，您可以查看 `Spine_diagnose/log` 下的日志。同时，您也可以通过设置环境变量 `RANK` 为 `1` 来将日志模式设为 `DEBUG` 以获取更丰富的信息。

## 训练

本项目需要两个模型权重才能完整运行，它们分别对应检测和分类。下面分别是训练检测器（YOLOv6）和分类器（Resnet50）的例子，您也可以尝试修改代码并替换其他模型。

- 准备数据集

在开始训练之前，需要先准备数据集。我们推荐您从[网盘链接]()下载已经预处理过的开箱即用的数据集。下载好后进行解压，您将看到以下内容。

```
.
├── eval
├── detect
├── classify
└── views
```

- `eval`：用于性能评估的数据，在 `eval.py` 使用
- `detect`：yolo格式的训练数据，用于训练检测器
- `classify`：用于训练分类器
- `views`：数据集的可视化结果，您可以查看它们

将它们移动到项目的任意目录下，并记下它。

接下来，按照之前记下的信息修改 `Spine_diagnose/data/spine.yaml` 中的内容。注意，您只需要修改文件中的路径。

> 您也可以从[链接](https://tianchi.aliyun.com/dataset/79463)下载原始数据集并使用我们提供的工具提取有效数据（需要补充网盘链接，官网数据集对不上）。具体方法请参阅[从原始数据集中提取数据](./doc/data_extract.md)。

### 训练 YOLOv6

通过以下代码训练 yolov6：

```
python train.py --model yolov6 --batch 32 --conf configs/yolov6s_finetune.py --device 0
```

其中，`--conf` 选项的内容是可另外配置的，您可以查看 `Spine_diagnose/configs` 目录来了解更多信息。

需要指出，本项目的 yolov6 训练配置参数与 YOLOv6 官方代码仓相同（包括 `train.py` 的参数）。

训练完毕后，您可以在 `Spine_diagnose/runs/train` 查看您的训练结果。

> 您可以使用 `python train.py --help` 查看参数的具体配置。

### 训练 Resnet50

通过以下代码训练 resnet50：

```
python train.py --model resnet50 --batch 32 --epochs 10 --conf configs/resnet50.py
```

训练完毕后，您可以在 `Spine_diagnose/runs/train` 查看您的训练结果。

值得一提的是，您可以通过配置 `--pretrained` 选项来指定是否使用预训练模型。

> 注意：当您希望使用自己训练的模型进行推理或性能评估时，需要在代码运行参数中配置 `--detect` 和 `--classify` 为您的模型存放路径。

## 性能评估

您可以通过以下方式进行性能评估：

```
python eval.py --source eval_dir --device 0
```

运行完毕后，您将获取模型在测试数据上的准确率、召回率、f1分数指标和AP值。

> 注意：此处的 `--source` 与 `predict.py` 中略有不同。请将运行参数中的 `--source` 设置为您存放性能评估数据 `eval` 的路径。除此之外，**`eval.py` 的运行参数与 `predict.py` 一致**。

## 项目介绍

本项目的大致思路如下：

- 通过YOLO/Faster-RCNN等目标检测模型对11块椎骨进行定位和检测
- 将识别到的椎骨切分成11张图片
- 通过Resnet/VIT/swin transformer等模型对椎骨的图片进行疾病情况分类

如果需要了解具体的实现思路，请移步[详细实现过程](./doc/detail.md)。

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
│   ├── __init__.py
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

其中 `modules` 包中实现了项目的核心功能：

- `adapter.py`: 对接不同检测和分类模型的适配器，用于将检测和分类任务集成到一个模块中方便调用。
- `classifier.py`: 实现了分类器的模型推理部分
- `detector.py`: 实现了检测器的模型推理部分
- `util.py`: 放置一些常用函数
- `yolov6/resnet/vit`: 定义不同的检测/分类模型结构的代码或github仓库